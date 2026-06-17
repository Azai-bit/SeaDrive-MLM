#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import math
import os
from collections import Counter


def _as_float(value):
    try:
        parsed = float(value)
    except Exception:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def _as_int(value):
    try:
        return int(float(value))
    except Exception:
        return None


def _as_bool(value):
    text = str(value or "").strip().lower()
    if text in ("1", "true", "yes", "y"):
        return True
    if text in ("0", "false", "no", "n", ""):
        return False
    return False


def _mean(values):
    valid = [v for v in values if v is not None]
    if not valid:
        return None
    return sum(valid) / float(len(valid))


def _first_float(row, *keys):
    for key in keys:
        value = _as_float(row.get(key))
        if value is not None:
            return value
    return None


def _fmt_float(value, digits=3):
    if value is None:
        return "N/A"
    return ("%." + str(digits) + "f") % value


def load_results(results_csv):
    rows = []
    with open(results_csv, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row:
                continue
            rows.append(row)
    return rows


def analyze_rows(rows):
    total = len(rows)
    success_count = sum(1 for row in rows if str(row.get("result", "")).strip().upper() == "SUCCESS")
    fail_count = total - success_count
    min_dcpa_values = [_first_float(row, "min_truth_dcpa_m", "min_dcpa_m") for row in rows]
    truth_cpa_time_values = [
        _first_float(row, "min_truth_tcpa_s", "truth_cpa_time_s", "min_tcpa_s")
        for row in rows
    ]
    nmf_values = [_as_int(row.get("nmf")) for row in rows]
    nmf_values = [v for v in nmf_values if v is not None]
    colreg_count = sum(1 for row in rows if _as_bool(row.get("colreg_violation")))
    reasons = Counter((row.get("reason") or "unknown").strip() or "unknown" for row in rows)

    scenario = next((row.get("scenario") for row in rows if row.get("scenario")), "N/A")
    mode = next((row.get("mode") for row in rows if row.get("mode")), "N/A")

    return {
        "scenario": scenario,
        "mode": mode,
        "total": total,
        "success_count": success_count,
        "fail_count": fail_count,
        "success_rate": (success_count / float(total)) if total else None,
        "avg_min_truth_dcpa_m": _mean(min_dcpa_values),
        "avg_truth_cpa_time_s": _mean(truth_cpa_time_values),
        "nmf_total": sum(nmf_values) if nmf_values else 0,
        "nmf_avg_per_trial": _mean(nmf_values),
        "colreg_violation_count": colreg_count,
        "colreg_violation_rate": (colreg_count / float(total)) if total else None,
        "reasons": reasons,
    }


def build_report(results_csv, stats):
    lines = []
    lines.append("Monte Carlo Results Analysis")
    lines.append("=" * 28)
    lines.append("results_csv: %s" % os.path.abspath(results_csv))
    lines.append("scenario: %s" % stats["scenario"])
    lines.append("mode: %s" % stats["mode"])
    lines.append("")
    lines.append("Trials")
    lines.append("- total: %d" % stats["total"])
    lines.append("- success: %d" % stats["success_count"])
    lines.append("- fail: %d" % stats["fail_count"])
    lines.append("- success_rate: %s%%" % _fmt_float(stats["success_rate"] * 100.0 if stats["success_rate"] is not None else None, 2))
    lines.append("")
    lines.append("Safety Metrics")
    lines.append("- avg_min_truth_dcpa_m: %s" % _fmt_float(stats["avg_min_truth_dcpa_m"], 3))
    lines.append("- avg_truth_cpa_time_s: %s" % _fmt_float(stats["avg_truth_cpa_time_s"], 3))
    lines.append("- nmf_total: %d" % stats["nmf_total"])
    lines.append("- nmf_avg_per_trial: %s" % _fmt_float(stats["nmf_avg_per_trial"], 3))
    lines.append("- colreg_violation_count: %d" % stats["colreg_violation_count"])
    lines.append("- colreg_violation_rate: %s%%" % _fmt_float(stats["colreg_violation_rate"] * 100.0 if stats["colreg_violation_rate"] is not None else None, 2))
    lines.append("")
    lines.append("Reasons")
    if stats["reasons"]:
        for reason, count in stats["reasons"].most_common():
            rate = count / float(stats["total"]) if stats["total"] else 0.0
            lines.append("- %s: %d (%.2f%%)" % (reason, count, rate * 100.0))
    else:
        lines.append("- N/A")
    lines.append("")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Analyze Monte Carlo results.csv")
    parser.add_argument(
        "results_csv",
        help="path to results.csv",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="output report path. Default: analysis_report.txt next to results.csv",
    )
    args = parser.parse_args()

    rows = load_results(args.results_csv)
    stats = analyze_rows(rows)
    report = build_report(args.results_csv, stats)

    out_path = args.out or os.path.join(os.path.dirname(args.results_csv), "analysis_report.txt")
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print("saved:", out_path)


if __name__ == "__main__":
    main()
