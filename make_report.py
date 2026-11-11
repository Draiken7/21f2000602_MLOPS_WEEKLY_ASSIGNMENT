"""Build report.md for CML from pytest's JUnit XML and evaluate.py's metrics.

Run from the repo root:  python3 make_report.py
Inputs : reports/junit.xml, reports/eval_metrics.json
Output : report.md
"""
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

from tests.settings import MIN_SCORES

JUNIT = Path("reports/junit.xml")
METRICS = Path("reports/eval_metrics.json")


def read_tests():
    """Return a list of (icon, test name, message) from the JUnit XML."""
    if not JUNIT.exists():
        return []
    rows = []
    for case in ET.parse(JUNIT).getroot().iter("testcase"):
        name = f"{case.get('classname', '').split('.')[-1]}::{case.get('name')}"
        problem = case.find("failure")
        if problem is None:
            problem = case.find("error")
        if problem is not None:
            msg = (problem.get("message") or "").splitlines()[0][:120]
            rows.append(("❌", name, msg.replace("|", "/")))
        elif case.find("skipped") is not None:
            rows.append(("⏭️", name, "skipped"))
        else:
            rows.append(("✅", name, ""))
    return rows


def main():
    tests = read_tests()
    failed = [t for t in tests if t[0] == "❌"]
    if not tests:
        status = "⚠️ NO TEST RESULTS - check the workflow log"
    elif failed:
        status = f"❌ FAILED ({len(failed)} of {len(tests)} tests)"
    else:
        status = f"✅ PASSED (all {len(tests)} tests)"

    branch = os.getenv("GITHUB_HEAD_REF") or os.getenv("GITHUB_REF_NAME", "local")
    sha = os.getenv("GITHUB_SHA", "local")[:7]
    run_url = ""
    if os.getenv("GITHUB_RUN_ID"):
        run_url = (f"{os.getenv('GITHUB_SERVER_URL')}/{os.getenv('GITHUB_REPOSITORY')}"
                   f"/actions/runs/{os.getenv('GITHUB_RUN_ID')}")

    lines = [
        "## 🌸 IRIS CI report",
        "",
        f"**Status:** {status}  ",
        f"**Branch:** `{branch}` · **Commit:** `{sha}` · **Event:** `{os.getenv('GITHUB_EVENT_NAME', 'local')}`",
        "",
        "### 📊 Model evaluation",
        "",
    ]

    if METRICS.exists():
        results = json.loads(METRICS.read_text())
        model_info = results.pop("model", None)
        if model_info:
            lines += [f"**Model:** `models:/{model_info['name']}/{model_info['version']}` "
                      "(fetched from the MLflow Model Registry)", ""]
        names = list(results)
        lines.append("| Metric | Minimum | " + " | ".join(names) + " |")
        lines.append("|---" * (len(names) + 2) + "|")
        for metric, minimum in MIN_SCORES.items():
            cells = []
            for n in names:
                value = results[n][metric]
                cells.append(f"{value:.3f} {'✅' if value >= minimum else '❌'}")
            lines.append(f"| {metric} | {minimum} | " + " | ".join(cells) + " |")
        lines.append("| rows evaluated | | " + " | ".join(str(results[n]["n_samples"]) for n in names) + " |")
        lines += ["", "_iris_test = 40% held-out split of data/iris.csv · "
                      "feast = Feast training data the model never saw_"]
    else:
        lines.append("_No metrics - evaluate.py did not run (see the workflow log)._")

    lines += ["", "<details><summary>🧪 All test results</summary>", "",
              "| | Test | Message |", "|---|---|---|"]
    lines += [f"| {icon} | `{name}` | {msg} |" for icon, name, msg in tests]
    lines += ["", "</details>"]
    if run_url:
        lines += ["", f"[Full workflow log]({run_url})"]

    Path("report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
