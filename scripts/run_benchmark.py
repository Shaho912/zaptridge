#!/usr/bin/env python3
"""Run the end-to-end benchmark for one dataset directory under ``data/``."""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Parse --model-id before importing cartridge modules so CARTRIDGES_MODEL_ID is
# set before config.py evaluates DEFAULT_MATRIX (which happens at import time).
_early = argparse.ArgumentParser(add_help=False)
_early.add_argument("--model-id", default=None)
_early_args, _ = _early.parse_known_args()
if _early_args.model_id:
    # Only override the LOCAL HF model — leave CARTRIDGES_MODEL_ID (vLLM server) unchanged.
    os.environ["CARTRIDGES_HF_MODEL_ID"] = _early_args.model_id

import httpx
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cartridges.benchmarks import (  # noqa: E402
    build_training_dataset,
    build_retrieval_index,
    generate_bootstrap_questions,
    generate_teacher_answers,
    route_eval_questions,
    write_budget_report,
    write_run_report,
)
from cartridges.data import (  # noqa: E402
    build_eval_rows_from_spec,
    build_text_manifest,
    load_corpus_slices,
    load_experiment_inputs,
)
from cartridges.data.common import write_json  # noqa: E402
from cartridges.eval import run_cartridge_eval, run_local_hf_matched_eval  # noqa: E402
from cartridges.train import train_cartridge  # noqa: E402


def _start_vllm_server(
    *,
    gpu_index: int,
    port: int,
    max_model_len: int,
    log_path: Path,
) -> subprocess.Popen[str]:
    """Launch the local vLLM teacher server used during bootstrap synthesis."""
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    command = [
        sys.executable,
        str(ROOT / "scripts" / "serve_vllm.py"),
        "--port",
        str(port),
        "--max-model-len",
        str(max_model_len),
        "--gpu-memory-utilization",
        "0.60",
        "--max-logprobs",
        "8",
    ]
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_file = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )


def _wait_for_server(base_url: str, api_key: str, timeout_seconds: float = 240.0) -> None:
    """Block until the managed vLLM server responds to health and model probes."""
    deadline = time.time() + timeout_seconds
    with httpx.Client(timeout=5.0) as client:
        while time.time() < deadline:
            try:
                health = client.get(base_url.replace("/v1", "/health"))
                models = client.get(
                    f"{base_url}/models",
                    headers={"Authorization": f"Bearer {api_key}"},
                )
                if health.status_code == 200 and models.status_code == 200:
                    return
            except httpx.HTTPError:
                pass
            time.sleep(2.0)
    raise TimeoutError(f"Timed out waiting for vLLM server at {base_url}.")


def _gpu_uuid_for_index(gpu_index: int) -> str:
    """Resolve a physical GPU index to the UUID used by ``nvidia-smi`` process queries."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    for line in result.stdout.splitlines():
        index_text, uuid = [item.strip() for item in line.split(",", maxsplit=1)]
        if int(index_text) == gpu_index:
            return uuid
    raise RuntimeError(f"Could not resolve GPU UUID for index {gpu_index}.")


def _kill_lingering_vllm_gpu_workers(gpu_index: int) -> None:
    """Kill orphaned vLLM worker processes that still hold memory on the managed GPU."""
    gpu_uuid = _gpu_uuid_for_index(gpu_index)
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    candidate_pids: list[int] = []
    for line in result.stdout.splitlines():
        gpu_uuid_text, pid_text, process_name = [item.strip() for item in line.split(",", maxsplit=2)]
        if gpu_uuid_text != gpu_uuid:
            continue
        if "vllm" not in process_name.lower():
            continue
        candidate_pids.append(int(pid_text))
    for pid in candidate_pids:
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            continue
    deadline = time.time() + 10.0
    while time.time() < deadline:
        alive = []
        for pid in candidate_pids:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                continue
            alive.append(pid)
        if not alive:
            return
        time.sleep(0.5)
    for pid in candidate_pids:
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            continue


def _stop_vllm_server(process: subprocess.Popen[str], port: int, gpu_index: int) -> None:
    """Stop the managed vLLM process, clear leaked workers, and verify the port is gone."""
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=120)
    _kill_lingering_vllm_gpu_workers(gpu_index)
    with httpx.Client(timeout=2.0) as client:
        try:
            client.get(f"http://127.0.0.1:{port}/health")
        except httpx.HTTPError:
            return
    raise RuntimeError(f"vLLM server on port {port} is still reachable after shutdown.")


def _copy_inputs(
    *,
    inputs: dict[str, object],
    destination_dir: Path,
) -> dict[str, str]:
    """Copy dataset inputs into the run directory for reproducible outputs."""
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied: dict[str, str] = {}
    for key in ("data_path", "eval_spec_path", "metadata_path"):
        source = inputs.get(key)
        if source is None:
            continue
        source_path = Path(str(source))
        target_path = destination_dir / source_path.name
        shutil.copy2(source_path, target_path)
        copied[key] = str(target_path.resolve())
    return copied


def _update_latest_pointer(run_dir: Path, *, output_root: Path, experiment_name: str) -> None:
    """Point ``outputs/<experiment>/latest`` at the run that just completed."""
    latest_path = output_root / experiment_name / "latest"
    if latest_path.is_symlink() or latest_path.exists():
        latest_path.unlink()
    latest_path.symlink_to(Path("runs") / run_dir.name)


def _cuda_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def _cuda_snapshot() -> dict[str, float]:
    """Return current CUDA memory stats in MB, or zeros if CUDA unavailable."""
    if not torch.cuda.is_available():
        return {"allocated_mb": 0.0, "peak_mb": 0.0, "reserved_mb": 0.0}
    return {
        "allocated_mb": torch.cuda.memory_allocated() / 1e6,
        "peak_mb":      torch.cuda.max_memory_allocated() / 1e6,
        "reserved_mb":  torch.cuda.memory_reserved() / 1e6,
    }


def _record_memory(
    phase_timings: dict,
    key: str,
    snap_before: dict[str, float],
    snap_after: dict[str, float],
) -> None:
    """Store peak and delta memory for a phase and print a [memory] line."""
    peak_mb  = snap_after["peak_mb"]
    delta_mb = snap_after["allocated_mb"] - snap_before["allocated_mb"]
    phase_timings[f"{key}_peak_memory_mb"]  = peak_mb
    phase_timings[f"{key}_memory_delta_mb"] = delta_mb
    print(f"[memory] {key}: peak={peak_mb:.0f}MB  delta={delta_mb:+.0f}MB  reserved={snap_after['reserved_mb']:.0f}MB")


def _profile_phase(fn, *, key: str, phase_timings: dict, enabled: bool):
    """Run fn() under torch.profiler when enabled; record FLOPs/bandwidth/roofline stats.

    Memory bytes are summed from self_cuda_memory_usage across all profiled kernels.
    This approximates output-tensor allocations per kernel — a lower bound on true DRAM
    traffic, but useful for relative comparison and arithmetic intensity estimation.
    """
    if not (enabled and torch.cuda.is_available()):
        return fn()
    import torch.profiler as _tp
    with _tp.profile(
        activities=[_tp.ProfilerActivity.CUDA],
        with_flops=True,
        profile_memory=True,
    ) as _prof:
        result = fn()
    avgs      = _prof.key_averages()
    flops     = sum(e.flops for e in avgs if e.flops > 0)
    mem_bytes = sum(
        abs(getattr(e, "cuda_memory_usage", None) or getattr(e, "self_cuda_memory_usage", None) or 0)
        for e in avgs
    )
    cuda_us   = sum(e.self_cuda_time_total for e in avgs)
    eff_tflops = flops / max(cuda_us * 1e-6, 1e-9) / 1e12
    eff_bw_gbs = mem_bytes / max(cuda_us * 1e-6, 1e-9) / 1e9
    ai         = flops / mem_bytes if mem_bytes > 0 else 0.0
    phase_timings.update({
        f"{key}_flops":                   flops,
        f"{key}_mem_bytes":               mem_bytes,
        f"{key}_cuda_time_us":            cuda_us,
        f"{key}_effective_tflops":        eff_tflops,
        f"{key}_effective_bandwidth_gbs": eff_bw_gbs,
        f"{key}_arithmetic_intensity":    ai,
    })
    mem_str = f"mem≈{mem_bytes/1e9:.2f}GB  AI={ai:.1f}  BW≈{eff_bw_gbs:.0f}GB/s" if mem_bytes > 0 else "mem=N/A"
    print(f"[profile] {key}:  flops={flops/1e12:.3f}T  {mem_str}  TFLOPS={eff_tflops:.2f}")
    return result


def main() -> int:
    """Execute the full benchmark pipeline for one experiment directory."""
    parser = argparse.ArgumentParser(
        description="Run the generic single-corpus benchmark for full context vs cartridges.",
    )
    parser.add_argument("experiment_name")
    parser.add_argument("--data-root", default=str(ROOT / "data"))
    parser.add_argument("--output-root", default=str(ROOT / "outputs"))
    parser.add_argument("--run-name")
    parser.add_argument("--gpu", type=int, default=3)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--base-url")
    parser.add_argument("--api-key", default="cartridges-local")
    parser.add_argument("--bootstrap-count", type=int, default=120)
    parser.add_argument("--cartridge-tokens", nargs="+", type=int, default=[1024])
    parser.add_argument("--train-steps", type=int, default=240)
    parser.add_argument("--train-learning-rate", type=float, default=3e-3)
    parser.add_argument("--train-max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-validation-examples", type=int, default=16)
    parser.add_argument("--train-validation-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-completion-tokens", type=int, default=48)
    parser.add_argument("--chunk-tokens", type=int, default=8192)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--semantic-judge", action="store_true")
    parser.add_argument("--judge-device")
    parser.add_argument("--kvzip-init", action="store_true",
                        help="Initialize cartridges from KVzip+ importance-scored tokens instead of the first p tokens.")
    parser.add_argument("--kvzip-prefix-tokens", type=int, default=256,
                        help="Number of leading tokens kept unconditionally as prefix in hybrid KVzip+ init (default: 256).")
    parser.add_argument("--kvzip-chunk-size", type=int, default=1024,
                        help="Reconstruction chunk size for KVzip+ scoring (default: 1024). Larger = better scores but more VRAM.")
    parser.add_argument("--kvzip-prune", action="store_true",
                        help="After training, prune the cartridge to --kvzip-prune-tokens slots using KVzip+ importance scores.")
    parser.add_argument("--kvzip-prune-tokens", type=int, default=None,
                        help="Target slot count after KVzip+ pruning (default: half of --cartridge-tokens).")
    parser.add_argument("--profile-flops", action="store_true",
                        help="Profile FLOPs during cartridge training and report TOPs.")
    parser.add_argument("--model-id", default=None,
                        help="Override the model ID (default: Qwen/Qwen3-4B). "
                             "E.g. --model-id Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    inputs = load_experiment_inputs(args.experiment_name, data_root=args.data_root)
    output_root = Path(args.output_root)
    run_id = args.run_name or datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = output_root / args.experiment_name / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    copied_inputs = _copy_inputs(inputs=inputs, destination_dir=run_dir / "input")
    manifest_path = run_dir / "input" / "corpus_manifest.json"
    eval_rows_path = run_dir / "eval" / "rows.jsonl"
    bootstrap_dir = run_dir / "bootstrap"
    retrieval_dir = run_dir / "retrieval"
    baseline_dir = run_dir / "baseline"
    baseline_predictions_path = baseline_dir / "predictions.jsonl"

    # Materialize the dataset into a manifest plus exact-match eval rows before any model work starts.
    stride_tokens = max(1, int(args.chunk_tokens * 0.85))
    build_text_manifest(
        source_path=inputs["data_path"],
        output_path=manifest_path,
        chunk_tokens=args.chunk_tokens,
        stride_tokens=stride_tokens,
        corpus_id=args.experiment_name,
    )
    slices = load_corpus_slices(manifest_path)
    eval_spec = json.loads(Path(str(inputs["eval_spec_path"])).read_text(encoding="utf-8"))
    eval_rows = build_eval_rows_from_spec(
        corpus_path=inputs["data_path"],
        spec_path=inputs["eval_spec_path"],
        output_path=eval_rows_path,
        sample_id=args.experiment_name,
    )
    retrieval_index = build_retrieval_index(slices=slices, output_dir=retrieval_dir)
    retrieval_routes = route_eval_questions(
        eval_rows=eval_rows,
        slices=slices,
        retrieval_dir=retrieval_dir,
    )

    managed_server = args.base_url is None
    base_url = args.base_url or f"http://127.0.0.1:{args.port}/v1"
    server = None
    server_max_model_len = args.chunk_tokens + max(args.max_completion_tokens + 256, 512)

    prepare_started = time.perf_counter()
    phase_timings: dict[str, float] = {}
    bootstrap_artifacts: list[dict[str, object]] = []
    teacher_answers_by_slice: dict[str, list[dict[str, str]]] = {}
    training_dataset_paths: dict[str, str] = {}
    bootstrap_question_total = 0
    _bootstrap_q_seconds = 0.0
    _teacher_ans_seconds = 0.0
    if managed_server:
        server = _start_vllm_server(
            gpu_index=args.gpu,
            port=args.port,
            max_model_len=server_max_model_len,
            log_path=run_dir / "logs" / "vllm.log",
        )
    try:
        if managed_server:
            _wait_for_server(base_url=base_url, api_key=args.api_key)
        # Generate teacher questions and exact answers independently for each chunk.
        for slice_record in slices:
            slice_id = str(slice_record["chunk_id"])
            slice_dir = bootstrap_dir / slice_id
            bootstrap_questions_path = slice_dir / "questions.txt"
            teacher_answers_path = slice_dir / "teacher_answers.jsonl"
            _t0 = time.perf_counter()
            bootstrap_examples = generate_bootstrap_questions(
                corpus_text=str(slice_record["text"]),
                eval_spec=eval_spec,
                output_path=bootstrap_questions_path,
                base_url=base_url,
                api_key=args.api_key,
                num_questions=args.bootstrap_count,
            )
            _dt = time.perf_counter() - _t0
            _bootstrap_q_seconds += _dt
            print(f"[timing] generate_bootstrap_questions slice={slice_id}: {_dt:.1f}s")
            _t0 = time.perf_counter()
            teacher_answers = generate_teacher_answers(
                corpus_text=str(slice_record["text"]),
                bootstrap_examples=bootstrap_examples,
                output_path=teacher_answers_path,
                base_url=base_url,
                api_key=args.api_key,
                max_completion_tokens=args.max_completion_tokens,
            )
            _dt = time.perf_counter() - _t0
            _teacher_ans_seconds += _dt
            print(f"[timing] generate_teacher_answers slice={slice_id}: {_dt:.1f}s")
            teacher_answers_by_slice[slice_id] = teacher_answers
            bootstrap_question_total += len(bootstrap_examples)
            bootstrap_artifacts.append(
                {
                    "slice_id": slice_id,
                    "questions_path": str(bootstrap_questions_path.resolve()),
                    "teacher_answers_path": str(teacher_answers_path.resolve()),
                    "bootstrap_question_count": len(bootstrap_examples),
                }
            )
    finally:
        if managed_server and server is not None:
            _stop_vllm_server(server, port=args.port, gpu_index=args.gpu)

    phase_timings["generate_bootstrap_questions_seconds"] = _bootstrap_q_seconds
    phase_timings["generate_teacher_answers_seconds"] = _teacher_ans_seconds
    print(f"[timing] generate_bootstrap_questions total: {_bootstrap_q_seconds:.1f}s")
    print(f"[timing] generate_teacher_answers total: {_teacher_ans_seconds:.1f}s")

    # Convert teacher answers into token-level supervision after vLLM is gone so the local
    # HF teacher can reuse the GPU safely on single-card machines.
    _build_dataset_seconds = 0.0
    _build_dataset_snap_before = _cuda_snapshot()
    _cuda_reset()

    def _do_build_training_dataset():
        nonlocal _build_dataset_seconds
        for slice_record in slices:
            slice_id = str(slice_record["chunk_id"])
            training_dataset_path = bootstrap_dir / slice_id / "train_dataset.jsonl"
            _t0 = time.perf_counter()
            build_training_dataset(
                corpus_text=str(slice_record["text"]),
                slice_id=slice_id,
                answer_records=teacher_answers_by_slice[slice_id],
                output_path=training_dataset_path,
                device=args.device,
                top_logprobs=5,
            )
            _dt = time.perf_counter() - _t0
            _build_dataset_seconds += _dt
            print(f"[timing] build_training_dataset slice={slice_id}: {_dt:.1f}s")
            training_dataset_paths[slice_id] = str(training_dataset_path.resolve())

    _profile_phase(
        _do_build_training_dataset,
        key="build_training_dataset",
        phase_timings=phase_timings,
        enabled=args.profile_flops,
    )
    phase_timings["build_training_dataset_seconds"] = _build_dataset_seconds
    _record_memory(phase_timings, "build_training_dataset", _build_dataset_snap_before, _cuda_snapshot())
    print(f"[timing] build_training_dataset total: {_build_dataset_seconds:.1f}s")
    preparation_seconds = time.perf_counter() - prepare_started

    _cuda_reset()
    _baseline_snap_before = _cuda_snapshot()
    _t0 = time.perf_counter()
    baseline_records = _profile_phase(
        lambda: run_local_hf_matched_eval(
            eval_path=eval_rows_path,
            output_path=baseline_predictions_path,
            device=args.device,
            max_completion_tokens=args.max_completion_tokens,
        ),
        key="baseline_eval",
        phase_timings=phase_timings,
        enabled=args.profile_flops,
    )
    phase_timings["baseline_eval_seconds"] = time.perf_counter() - _t0
    _record_memory(phase_timings, "baseline_eval", _baseline_snap_before, _cuda_snapshot())
    print(f"[timing] run_local_hf_matched_eval: {phase_timings['baseline_eval_seconds']:.1f}s")

    budget_summaries: list[dict[str, object]] = []
    for cartridge_tokens in args.cartridge_tokens:
        budget_label = f"cartridge_{cartridge_tokens}"
        budget_dir = run_dir / budget_label
        slice_train_summaries: dict[str, dict[str, object]] = {}
        cartridge_paths: dict[str, str] = {}
        train_seconds = 0.0
        _cuda_reset()
        _train_snap_before = _cuda_snapshot()
        for slice_record in slices:
            slice_id = str(slice_record["chunk_id"])
            train_started = time.perf_counter()
            # Each slice trains its own compact cache for the requested budget.
            train_summary = train_cartridge(
                dataset_path=training_dataset_paths[slice_id],
                output_dir=budget_dir / "train" / slice_id,
                slice_id=slice_id,
                device=args.device,
                cartridge_tokens=cartridge_tokens,
                learning_rate=args.train_learning_rate,
                steps=args.train_steps,
                max_grad_norm=args.train_max_grad_norm,
                seed=args.seed,
                validation_examples=args.train_validation_examples,
                validation_interval=args.train_validation_interval,
                kvzip_init=args.kvzip_init,
                kvzip_prefix_tokens=args.kvzip_prefix_tokens,
                kvzip_chunk_size=args.kvzip_chunk_size,
                kvzip_prune=args.kvzip_prune,
                kvzip_prune_tokens=args.kvzip_prune_tokens,
                profile_flops=args.profile_flops,
            )
            _dt = time.perf_counter() - train_started
            train_seconds += _dt
            _tops = train_summary.get("train_tops")
            _flops_msg = f"  TOPs={_tops:.2f}" if _tops else ""
            print(f"[timing] train_cartridge {budget_label} slice={slice_id}: {_dt:.1f}s{_flops_msg}")
            if args.profile_flops:
                _fp   = train_summary.get("flops_per_step") or 0.0
                _mb   = train_summary.get("mem_bytes_per_step") or 0.0
                _nsteps = train_summary.get("steps", 0)
                _loop_s = train_summary.get("train_loop_seconds") or 1e-9
                _ai   = _fp / _mb if _mb > 0 else 0.0
                _bw   = (_mb * _nsteps / _loop_s) / 1e9 if _mb > 0 else 0.0
                print(
                    f"[profile] train_cartridge_{budget_label} slice={slice_id}:  "
                    f"flops={_fp * _nsteps / 1e12:.3f}T  "
                    f"mem≈{_mb * _nsteps / 1e9:.2f}GB  AI={_ai:.1f}  "
                    f"TFLOPS={_tops or 0.0:.2f}  BW≈{_bw:.0f}GB/s"
                )
            slice_train_summaries[slice_id] = train_summary
            cartridge_paths[slice_id] = str(train_summary["cartridge_path"])
        phase_timings[f"train_cartridge_seconds_{budget_label}"] = train_seconds
        _record_memory(phase_timings, f"train_cartridge_{budget_label}", _train_snap_before, _cuda_snapshot())
        print(f"[timing] train_cartridge {budget_label} total: {train_seconds:.1f}s")
        if args.profile_flops and slice_train_summaries:
            _last = next(iter(slice_train_summaries.values()))
            _fp   = _last.get("flops_per_step") or 0.0
            _mb   = _last.get("mem_bytes_per_step") or 0.0
            _nsteps = _last.get("steps", 0)
            _loop_s = _last.get("train_loop_seconds") or 1e-9
            _key = f"train_cartridge_{budget_label}"
            phase_timings.update({
                f"{_key}_flops":                   _fp * _nsteps,
                f"{_key}_mem_bytes":               _mb * _nsteps,
                f"{_key}_effective_tflops":        _last.get("train_tops") or 0.0,
                f"{_key}_effective_bandwidth_gbs": (_mb * _nsteps / _loop_s) / 1e9 if _mb else 0.0,
                f"{_key}_arithmetic_intensity":    _fp / _mb if _mb > 0 else 0.0,
            })
        cartridge_predictions_path = budget_dir / "predictions.jsonl"
        # Cartridge inference uses the same HF backend as the matched baseline, with top-1
        # chunk routing selecting which slice-specific cache to inject for each question.
        _cuda_reset()
        _eval_snap_before = _cuda_snapshot()
        _t0 = time.perf_counter()
        _profile_phase(
            lambda: run_cartridge_eval(
                eval_path=eval_rows_path,
                cartridge_paths=cartridge_paths,
                retrieval_routes=retrieval_routes,
                output_path=cartridge_predictions_path,
                device=args.device,
                max_completion_tokens=args.max_completion_tokens,
            ),
            key=f"cartridge_eval_{budget_label}",
            phase_timings=phase_timings,
            enabled=args.profile_flops,
        )
        phase_timings[f"cartridge_eval_seconds_{budget_label}"] = time.perf_counter() - _t0
        _record_memory(phase_timings, f"cartridge_eval_{budget_label}", _eval_snap_before, _cuda_snapshot())
        print(f"[timing] run_cartridge_eval {budget_label}: {phase_timings[f'cartridge_eval_seconds_{budget_label}']:.1f}s")
        summary = write_budget_report(
            experiment_name=args.experiment_name,
            budget_label=budget_label,
            baseline_path=baseline_predictions_path,
            cartridge_path=cartridge_predictions_path,
            output_dir=budget_dir / "report",
            build_seconds=preparation_seconds + train_seconds,
            bootstrap_question_count=bootstrap_question_total,
            train_steps=args.train_steps,
            cartridge_tokens=cartridge_tokens,
            semantic_judge=args.semantic_judge,
            judge_device=args.judge_device or args.device,
        )
        write_json(
            budget_dir / "manifest.json",
            {
                "experiment_name": args.experiment_name,
                "cartridge_tokens": cartridge_tokens,
                "slice_train_summaries": slice_train_summaries,
                "cartridge_paths": cartridge_paths,
                "predictions_path": str(cartridge_predictions_path.resolve()),
                "report_path": summary["report_path"],
                "semantic_judge": args.semantic_judge,
            },
        )
        budget_summaries.append(summary)

        # If pruning was requested, collect the pruned cartridge paths from each
        # slice's train summary and run a separate eval + report for the pruned budget.
        if args.kvzip_prune:
            pruned_paths: dict[str, str] = {}
            for slice_id, ts in slice_train_summaries.items():
                pruned_path = ts.get("pruned_cartridge_path")
                if pruned_path:
                    pruned_paths[slice_id] = pruned_path
            if pruned_paths:
                prune_tokens = args.kvzip_prune_tokens or cartridge_tokens // 2
                pruned_label = f"cartridge_{cartridge_tokens}_pruned_{prune_tokens}"
                pruned_budget_dir = run_dir / pruned_label
                pruned_predictions_path = pruned_budget_dir / "predictions.jsonl"
                run_cartridge_eval(
                    eval_path=eval_rows_path,
                    cartridge_paths=pruned_paths,
                    retrieval_routes=retrieval_routes,
                    output_path=pruned_predictions_path,
                    device=args.device,
                    max_completion_tokens=args.max_completion_tokens,
                )
                pruned_summary = write_budget_report(
                    experiment_name=args.experiment_name,
                    budget_label=pruned_label,
                    baseline_path=baseline_predictions_path,
                    cartridge_path=pruned_predictions_path,
                    output_dir=pruned_budget_dir / "report",
                    build_seconds=preparation_seconds + train_seconds,
                    bootstrap_question_count=bootstrap_question_total,
                    train_steps=args.train_steps,
                    cartridge_tokens=prune_tokens,
                    semantic_judge=args.semantic_judge,
                    judge_device=args.judge_device or args.device,
                )
                write_json(
                    pruned_budget_dir / "manifest.json",
                    {
                        "experiment_name": args.experiment_name,
                        "cartridge_tokens": prune_tokens,
                        "trained_tokens": cartridge_tokens,
                        "pruned_from": budget_label,
                        "slice_train_summaries": slice_train_summaries,
                        "cartridge_paths": pruned_paths,
                        "predictions_path": str(pruned_predictions_path.resolve()),
                        "report_path": pruned_summary["report_path"],
                        "semantic_judge": args.semantic_judge,
                    },
                )
                budget_summaries.append(pruned_summary)

    # The run manifest is the stable index for everything generated during one invocation.
    aggregate_report = write_run_report(
        experiment_name=args.experiment_name,
        run_dir=run_dir,
        budget_summaries=budget_summaries,
    )
    run_manifest = {
        "experiment_name": args.experiment_name,
        "run_id": run_id,
        "run_dir": str(run_dir.resolve()),
        "input_files": copied_inputs,
        "corpus_manifest_path": str(manifest_path.resolve()),
        "eval_rows_path": str(eval_rows_path.resolve()),
        "retrieval_index_path": str((retrieval_dir / "index.json").resolve()),
        "retrieval_routes_path": str((retrieval_dir / "routes.jsonl").resolve()),
        "retrieval_summary_path": str((retrieval_dir / "summary.json").resolve()),
        "slice_count": len(slices),
        "chunk_tokens": args.chunk_tokens,
        "stride_tokens": stride_tokens,
        "bootstrap_slices": bootstrap_artifacts,
        "training_datasets": training_dataset_paths,
        "baseline_predictions_path": str(baseline_predictions_path.resolve()),
        "baseline_records": len(baseline_records),
        "retrieval_index": retrieval_index,
        "budget_reports": [
            {
                "budget_label": item["budget_label"],
                "report_path": item["report_path"],
                "summary_path": str(
                    (run_dir / item["budget_label"] / "report" / "summary.json").resolve()
                ),
            }
            for item in budget_summaries
        ],
        "aggregate_report": aggregate_report,
        "phase_timings": phase_timings,
    }
    write_json(run_dir / "run_manifest.json", run_manifest)
    print("\n[timing] === phase summary ===")
    for phase, seconds in phase_timings.items():
        print(f"[timing]   {phase}: {seconds:.1f}s")
    _update_latest_pointer(run_dir, output_root=output_root, experiment_name=args.experiment_name)

    print(json.dumps(run_manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
