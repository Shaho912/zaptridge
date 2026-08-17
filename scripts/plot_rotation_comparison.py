#!/usr/bin/env python3
"""Compare key/value rotation trajectories: independent training vs CAS.

Loads {name}_movement.json from Exp 1 (independent) and Exp 2 (CAS) output dirs,
converts cosine similarity to rotation angle (degrees), and prints a comparison
table averaged across all named patients.

The key result: if CAS amplifies value rotation relative to independent training,
that's direct evidence CAS achieves routing by pushing values into more differentiated
regions — not by changing what keys encode.

Usage:
    python scripts/plot_rotation_comparison.py \
        --exp1-dir outputs/exp_rotation_1 \
        --exp2-dir outputs/exp_rotation_2 \
        --names lh_p01 lh_p02 lh_p03 lh_p04 \
        [--output-html rotation_comparison.html]
"""

import argparse
import json
import math
import sys
from pathlib import Path


def cosim_to_degrees(cosim: float) -> float:
    cosim = max(-1.0, min(1.0, cosim))
    return math.degrees(math.acos(cosim))


def load_movement_log(path: Path) -> list[dict]:
    """Load movement log; normalize step key differences between Exp1/Exp2."""
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    normalized = []
    for entry in data:
        step = entry.get("step") or entry.get("cart_step")
        normalized.append({
            "step": step,
            "key_deg": cosim_to_degrees(entry["mean_key_cosim"]),
            "val_deg": cosim_to_degrees(entry["mean_val_cosim"]),
            "raw_key_cosim": entry["mean_key_cosim"],
            "raw_val_cosim": entry["mean_val_cosim"],
        })
    return normalized


def average_logs(logs: list[list[dict]]) -> list[dict]:
    """Average multiple per-patient logs at matching steps."""
    if not logs:
        return []
    steps = [e["step"] for e in logs[0]]
    result = []
    for i, step in enumerate(steps):
        key_degs = [l[i]["key_deg"] for l in logs if i < len(l)]
        val_degs = [l[i]["val_deg"] for l in logs if i < len(l)]
        result.append({
            "step": step,
            "key_deg": sum(key_degs) / len(key_degs),
            "val_deg": sum(val_degs) / len(val_degs),
        })
    return result


def print_table(exp1_avg: list[dict], exp2_avg: list[dict]) -> None:
    print("\n" + "=" * 72)
    print("  Key/Value Rotation: Independent vs CAS  (degrees per 50-step window)")
    print("=" * 72)
    print(f"  {'Step':>6}  {'Indep Key°':>10}  {'CAS Key°':>10}  "
          f"{'Indep Val°':>10}  {'CAS Val°':>10}  {'ΔVal° (CAS−Ind)':>16}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*10}  {'-'*16}")

    e2_by_step = {e["step"]: e for e in exp2_avg}
    for e1 in exp1_avg:
        step = e1["step"]
        e2 = e2_by_step.get(step, {})
        k1, v1 = e1["key_deg"], e1["val_deg"]
        k2, v2 = e2.get("key_deg", float("nan")), e2.get("val_deg", float("nan"))
        dv = v2 - v1 if not math.isnan(v2) else float("nan")
        dv_str = f"+{dv:>7.3f}" if dv >= 0 else f"{dv:>8.3f}"
        print(f"  {step:>6}  {k1:>10.3f}  {k2:>10.3f}  {v1:>10.3f}  {v2:>10.3f}  {dv_str:>16}")

    print()
    k1_mean = sum(e["key_deg"] for e in exp1_avg) / len(exp1_avg)
    k2_mean = sum(e["key_deg"] for e in exp2_avg) / len(exp2_avg)
    v1_mean = sum(e["val_deg"] for e in exp1_avg) / len(exp1_avg)
    v2_mean = sum(e["val_deg"] for e in exp2_avg) / len(exp2_avg)
    dv_mean = v2_mean - v1_mean
    print(f"  {'MEAN':>6}  {k1_mean:>10.3f}  {k2_mean:>10.3f}  "
          f"{v1_mean:>10.3f}  {v2_mean:>10.3f}  "
          f"{'+'if dv_mean>=0 else ''}{dv_mean:>+9.3f}°")
    print("=" * 72)

    print("\nInterpretation:")
    if dv_mean > 0.5:
        print(f"  ✓ CAS values rotate {dv_mean:.2f}° MORE per window than independent.")
        print(f"    CAS is pushing values into more differentiated regions.")
        print(f"    Routing improvement is encoded in value-space, not key-space.")
    elif dv_mean < -0.5:
        print(f"  ✗ CAS values rotate {-dv_mean:.2f}° LESS than independent.")
        print(f"    CAS appears to regularize values (unexpected).")
    else:
        print(f"  ~ CAS value rotation ≈ independent (Δ={dv_mean:+.2f}°).")
        print(f"    CAS routing improvement is achieved without amplifying value movement.")

    dk_mean = k2_mean - k1_mean
    print(f"\n  Keys: CAS Δ={dk_mean:+.3f}° (expected near 0 — keys are corpus-agnostic in both)")


def write_html(exp1_avg: list[dict], exp2_avg: list[dict], out_path: Path) -> None:
    e2_by_step = {e["step"]: e for e in exp2_avg}
    steps = [e["step"] for e in exp1_avg]

    def js_list(values):
        return "[" + ", ".join(f"{v:.4f}" for v in values) + "]"

    k1 = [e["key_deg"] for e in exp1_avg]
    v1 = [e["val_deg"] for e in exp1_avg]
    k2 = [e2_by_step[s]["key_deg"] if s in e2_by_step else 0 for s in steps]
    v2 = [e2_by_step[s]["val_deg"] if s in e2_by_step else 0 for s in steps]

    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Key/Value Rotation: Independent vs CAS</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 860px; margin: 40px auto; padding: 0 20px;
         background: #fff; color: #111; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background: #1a1a1a; color: #e8e8e8; }}
    canvas {{ filter: invert(0); }}
  }}
  h1 {{ font-size: 1.2em; font-weight: 600; }}
  p.sub {{ color: #666; font-size: 0.9em; }}
  canvas {{ width: 100%; height: 360px; display: block; border: 1px solid #ddd; border-radius: 4px; }}
  .legend {{ display: flex; gap: 24px; font-size: 0.85em; margin-top: 12px; }}
  .legend span {{ display: flex; align-items: center; gap: 6px; }}
  .swatch {{ display: inline-block; width: 28px; height: 3px; border-radius: 2px; }}
  .finding {{ background: #f0f7ff; border-left: 3px solid #3b82f6; padding: 12px 16px;
              margin-top: 24px; font-size: 0.9em; border-radius: 2px; }}
  @media (prefers-color-scheme: dark) {{
    .finding {{ background: #1e2d40; border-color: #60a5fa; }}
    p.sub {{ color: #aaa; }}
    canvas {{ border-color: #333; }}
  }}
</style>
</head>
<body>
<h1>Key vs Value Rotation: Independent Training vs CAS</h1>
<p class="sub">Rotation angle (°) per 50-step window, averaged across 4 patients.
Keys expected flat (~0°) in both; values expected larger in CAS if CAS routes via value differentiation.</p>

<canvas id="chart"></canvas>
<div class="legend">
  <span><span class="swatch" style="background:#2563eb; border-top:2px dashed #2563eb"></span>Key° — Independent</span>
  <span><span class="swatch" style="background:#2563eb"></span>Key° — CAS</span>
  <span><span class="swatch" style="background:#dc2626; border-top:2px dashed #dc2626"></span>Value° — Independent</span>
  <span><span class="swatch" style="background:#dc2626"></span>Value° — CAS</span>
</div>

<div class="finding" id="finding"></div>

<script>
const steps = {js_list(steps)};
const k1    = {js_list(k1)};
const v1    = {js_list(v1)};
const k2    = {js_list(k2)};
const v2    = {js_list(v2)};

const canvas = document.getElementById("chart");
const ctx = canvas.getContext("2d");

function resize() {{
  canvas.width  = canvas.offsetWidth * devicePixelRatio;
  canvas.height = canvas.offsetHeight * devicePixelRatio;
}}
resize();
window.addEventListener("resize", () => {{ resize(); draw(); }});

function draw() {{
  const W = canvas.width, H = canvas.height;
  const PAD = {{ t: 20, r: 20, b: 40, l: 50 }};
  const w = W - PAD.l - PAD.r;
  const h = H - PAD.t - PAD.b;
  ctx.clearRect(0, 0, W, H);

  const allVals = [...k1, ...k2, ...v1, ...v2];
  const yMax = Math.max(...allVals) * 1.15;
  const yMin = 0;
  const xMin = steps[0], xMax = steps[steps.length - 1];

  function px(x) {{ return PAD.l + (x - xMin) / (xMax - xMin) * w; }}
  function py(y) {{ return PAD.t + (1 - (y - yMin) / (yMax - yMin)) * h; }}

  // Grid
  ctx.strokeStyle = "#e5e7eb"; ctx.lineWidth = 1;
  const isDark = window.matchMedia("(prefers-color-scheme: dark)").matches;
  ctx.strokeStyle = isDark ? "#333" : "#e5e7eb";
  for (let y = 0; y <= yMax; y += 2) {{
    ctx.beginPath(); ctx.moveTo(PAD.l, py(y)); ctx.lineTo(PAD.l + w, py(y)); ctx.stroke();
    ctx.fillStyle = isDark ? "#aaa" : "#666";
    ctx.font = `${{10 * devicePixelRatio}}px system-ui`;
    ctx.fillText(y + "°", 2, py(y) + 4 * devicePixelRatio);
  }}

  // X axis labels
  ctx.fillStyle = isDark ? "#aaa" : "#666";
  for (const s of steps) {{
    ctx.fillText("step " + s, px(s) - 18 * devicePixelRatio, H - 4);
  }}

  // Draw lines
  function line(data, color, dashed) {{
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = 2 * devicePixelRatio;
    if (dashed) ctx.setLineDash([6 * devicePixelRatio, 4 * devicePixelRatio]);
    else ctx.setLineDash([]);
    for (let i = 0; i < steps.length; i++) {{
      const x = px(steps[i]), y = py(data[i]);
      i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
    }}
    ctx.stroke();
    // dots
    ctx.setLineDash([]);
    for (let i = 0; i < steps.length; i++) {{
      ctx.beginPath();
      ctx.arc(px(steps[i]), py(data[i]), 4 * devicePixelRatio, 0, 2 * Math.PI);
      ctx.fillStyle = color; ctx.fill();
    }}
  }}

  line(k1, "#2563eb", true);   // key independent — dashed blue
  line(k2, "#2563eb", false);  // key CAS — solid blue
  line(v1, "#dc2626", true);   // val independent — dashed red
  line(v2, "#dc2626", false);  // val CAS — solid red
}}

draw();

// Finding text
const v1mean = v1.reduce((a,b)=>a+b,0)/v1.length;
const v2mean = v2.reduce((a,b)=>a+b,0)/v2.length;
const dv = (v2mean - v1mean).toFixed(2);
const k1mean = k1.reduce((a,b)=>a+b,0)/k1.length;
const k2mean = k2.reduce((a,b)=>a+b,0)/k2.length;
const dk = (k2mean - k1mean).toFixed(2);
const el = document.getElementById("finding");
if (parseFloat(dv) > 0.5) {{
  el.innerHTML = `<strong>Finding:</strong> CAS values rotate <strong>+${{dv}}° more</strong> per window than independent training.
  CAS achieves its routing improvement by pushing values into a more differentiated region of representation space.
  Keys are nearly unchanged between conditions (Δ=${{dk}}°), confirming the routing mechanism is entirely value-space.`;
}} else if (parseFloat(dv) < -0.5) {{
  el.innerHTML = `<strong>Finding:</strong> CAS values rotate <strong>${{Math.abs(dv)}}° less</strong> than independent.
  CAS appears to regularize values, not amplify them.`;
}} else {{
  el.innerHTML = `<strong>Finding:</strong> CAS and independent value rotation are similar (Δ=${{dv}}°).
  Routing improvement is achieved without measurably larger value movement (Δkey=${{dk}}°).`;
}}
</script>
</body>
</html>
"""
    out_path.write_text(html, encoding="utf-8")
    print(f"\nHTML chart → {out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare key/value rotation: independent vs CAS")
    parser.add_argument("--exp1-dir", required=True, help="Output dir from train_paper_cartridges.py")
    parser.add_argument("--exp2-dir", required=True, help="Output dir from train_joint_cartridges.py")
    parser.add_argument(
        "--names", nargs="+",
        default=["lh_p01", "lh_p02", "lh_p03", "lh_p04"],
        help="Patient/corpus names to average over",
    )
    parser.add_argument("--output-html", default=None, help="Optional path for HTML output")
    args = parser.parse_args()

    exp1_dir = Path(args.exp1_dir)
    exp2_dir = Path(args.exp2_dir)

    exp1_logs, exp2_logs = [], []
    for name in args.names:
        p1 = exp1_dir / f"{name}_movement.json"
        p2 = exp2_dir / f"{name}_movement.json"
        if not p1.exists():
            print(f"Missing Exp 1 movement log: {p1}", file=sys.stderr)
            print("Re-run train_paper_cartridges.py with --movement-log-interval 50", file=sys.stderr)
            return 1
        if not p2.exists():
            print(f"Missing Exp 2 movement log: {p2}", file=sys.stderr)
            print("Re-run train_joint_cartridges.py with --movement-log-interval 50", file=sys.stderr)
            return 1
        exp1_logs.append(load_movement_log(p1))
        exp2_logs.append(load_movement_log(p2))
        print(f"  [{name}] loaded: {len(exp1_logs[-1])} Exp1 windows, {len(exp2_logs[-1])} Exp2 windows")

    exp1_avg = average_logs(exp1_logs)
    exp2_avg = average_logs(exp2_logs)

    print_table(exp1_avg, exp2_avg)

    if args.output_html:
        write_html(exp1_avg, exp2_avg, Path(args.output_html))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
