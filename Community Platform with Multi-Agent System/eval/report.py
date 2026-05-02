"""
Evaluation report generator — produces HTML and JSON reports from eval results.
"""

import argparse
import json
import os


def generate_report(input_dir: str = "reports/eval", output: str = "reports/eval_report.html"):
    """Generate an HTML evaluation report from the latest eval output."""
    summary_path = os.path.join(input_dir, "summary.json")
    if not os.path.exists(summary_path):
        print(f"[report] No eval results found at {summary_path}")
        print("[report] Run `make eval` first to generate results.")
        return {"error": "no_results", "overall_score": 0.0}

    with open(summary_path) as f:
        data = json.load(f)

    summary = data.get("summary", {})
    scenarios = data.get("scenarios", [])
    avg_score = summary.get("average_score", 0)
    passed = summary.get("scenarios_passed", 0)
    total = summary.get("scenarios_run", 0)

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Evaluation Report</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 800px; margin: 40px auto; padding: 0 20px; }}
h1 {{ color: #333; }}
.score {{ font-size: 48px; font-weight: bold; }}
.pass {{ color: #22c55e; }}
.fail {{ color: #ef4444; }}
table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
th, td {{ padding: 10px 12px; text-align: left; border-bottom: 1px solid #e5e7eb; }}
th {{ background: #f9fafb; font-weight: 600; }}
.bar {{ height: 20px; border-radius: 4px; }}
.summary {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin: 20px 0; }}
.card {{ padding: 16px; border: 1px solid #e5e7eb; border-radius: 8px; text-align: center; }}
.card-value {{ font-size: 28px; font-weight: bold; }}
</style>
</head>
<body>
<h1>Multi-Agent System — Evaluation Report</h1>
<p>Generated: {__import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")}</p>

<div class="summary">
<div class="card">
<div class="card-value {'pass' if avg_score >= 0.75 else 'fail'}">{avg_score:.0%}</div>
<div>Average Score</div>
</div>
<div class="card">
<div class="card-value">{passed}/{total}</div>
<div>Scenarios Passed</div>
</div>
<div class="card">
<div class="card-value {'pass' if summary.get('pass_rate', 0) >= 0.75 else 'fail'}">{summary.get('pass_rate', 0):.0%}</div>
<div>Pass Rate</div>
</div>
</div>

<table>
<tr><th>Scenario</th><th>Score</th></tr>"""

    for s in scenarios:
        cls = "pass" if s["overall_score"] >= 0.75 else "fail"
        html += f'<tr><td>{s["scenario_id"]}</td><td class="{cls}">{s["overall_score"]:.0%}</td></tr>'

    html += """
</table>
</body>
</html>"""

    with open(output, "w") as f:
        f.write(html)

    print(f"[report] Report generated: {output}")
    return {"overall_score": avg_score}


def main():
    parser = argparse.ArgumentParser(description="Evaluation Report Generator")
    parser.add_argument("--input-dir", default="reports/eval")
    parser.add_argument("--output", default="reports/eval_report.html")
    parser.add_argument("--json", action="store_true",
                        help="Output JSON summary to stdout")
    args = parser.parse_args()

    result = generate_report(args.input_dir, args.output)

    if args.json:
        print(json.dumps(result))


if __name__ == "__main__":
    main()
