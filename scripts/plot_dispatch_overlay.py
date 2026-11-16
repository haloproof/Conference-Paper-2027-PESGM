"""Replot Figure 5 as one dual-axis panel from saved CSV; never simulate.

Git Bash, from the project root:
  source "scripts/activate.sh"
  timeout --signal=KILL 55s python "results/plot_dispatch_overlay.py" --force
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pypdf import PdfReader

from plot_training import (
    CASE_STUDY_COLORS, STYLE_PATH, figure_font, figure_style,
    setup_publication_style,
)

METHODS = ("M1", "M2", "M3", "M4", "M5")


def render(csv_path, output, *, max_seconds=50):
    started = time.monotonic()
    if not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError("positive finite work limit required")

    def guard():
        if time.monotonic() - started > max_seconds:
            raise TimeoutError("Figure 5 plotting exceeded its work limit")

    raw = csv_path.read_bytes()
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) not in (96, 672):
        raise ValueError("Figure 5 requires one or seven complete quarter-hour days")
    window_days = len(rows) // 96
    first_day = int(rows[0]["simulation_day"])
    days = {int(r["simulation_day"]) for r in rows}
    if any(int(r["simulation_day"]) != first_day + i // 96 or
           int(r["interval_in_day"]) != i % 96 for i, r in enumerate(rows)):
        raise ValueError("complete, consecutive simulation days required")
    start = np.array([float(r["time_start_h"]) for r in rows])
    end = np.array([float(r["time_end_h"]) for r in rows])
    price = np.array([float(r["price_eur_per_mwh"]) for r in rows])
    power = {m: np.array([float(r[m + "_command_ac_kw"]) for r in rows]) for m in METHODS}
    if not all(np.isfinite(a).all() for a in [start, end, price, *power.values()]):
        raise ValueError("finite saved coordinates required")
    if not np.array_equal(start, np.arange(len(rows)) / 4) or not np.array_equal(end, np.arange(1, len(rows) + 1) / 4):
        raise ValueError("expected unchanged consecutive quarter-hour time edges")
    guard()

    font = setup_publication_style()
    width = figure_style()["canvas_width_inches"]
    height = width * 0.4  # A compact single panel; no change to the common style file.
    fig, price_ax = plt.subplots(figsize=(width, height))
    power_ax = price_ax.twinx()
    fig.subplots_adjust(left=0.17, right=0.85, bottom=0.25, top=0.79)
    power_ax.spines["right"].set_visible(True)
    edges = np.r_[start, end[-1]]
    price_line, = price_ax.step(
        edges, np.r_[price, price[-1]], where="post", color="#333333",
        linestyle=(0, (4, 2)), linewidth=1.7, label="Price", zorder=2,
    )
    price_ax.fill_between(
        edges, np.r_[price, price[-1]], 0, step="post", color="#777777",
        alpha=0.055, linewidth=0, zorder=1,
    )

    styles = {
        "M1": ("-", 1.25),
        "M2": ((0, (4, 2)), 1.7),
        "M3": ((0, (5, 2, 1, 2)), 1.8),
        "M4": ("-", 2.1),
        "M5": ((0, (3, 1.5)), 1.9),
    }
    lines = [price_line]
    for method in METHODS:
        linestyle, linewidth = styles[method]
        line, = power_ax.step(
            edges, np.r_[power[method], power[method][-1]], where="post",
            color=CASE_STUDY_COLORS[method], linestyle=linestyle,
            linewidth=linewidth, alpha=1.0, label=method,
            zorder=3 + METHODS.index(method),
        )
        lines.append(line)
    power_ax.axhline(0, color="#A8A8A8", linewidth=0.65, zorder=1)

    price_ax.set_xlabel("Time from dispatch start (h)")
    price_ax.set_ylabel("Price (EUR/MWh)")
    power_ax.set_ylabel("Power (kW)", rotation=90, labelpad=24)
    price_ax.set_xlim(0, end[-1])
    price_ax.set_xticks(np.arange(0, end[-1] + 0.1, 4 if window_days == 1 else 24))
    low = min(0.0, 50 * math.floor(float(price.min()) / 50))
    high = max(50.0, 50 * math.ceil(float(price.max()) / 50))
    price_ax.set_ylim(low, high)
    price_ax.set_yticks(np.arange(low, high + 1, 50))
    power_ax.set_ylim(-10.7, 10.7)
    power_ax.set_yticks([-10, -5, 0, 5, 10])
    for axis in (price_ax, power_ax):
        axis.tick_params(direction="out", length=3, width=0.7, pad=3)
        axis.spines["top"].set_visible(False)
    fig.legend(
        handles=lines, labels=["Price", *METHODS], loc="upper center",
        bbox_to_anchor=(0.5, 0.98), ncol=6, frameon=False,
        handlelength=1.15, handletextpad=0.3, columnspacing=0.7,
        fontsize=figure_font("legend"), borderaxespad=0,
    )
    guard()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output, format="pdf",
        metadata={
            "Title": "Figure 5. Single-panel price and M1-M5 dispatch",
            "Subject": (
                "Saved quarter-hour commands starting on day %d; left price (EUR/MWh), "
                "right AC power (kW), positive discharge. No interpolation, "
                "smoothing, model inference or simulation." % first_day
            ),
            "Author": "Pengfei He",
            "Creator": "results/plot_dispatch_overlay.py (Matplotlib)",
        },
    )
    plt.close(fig)
    pdf = PdfReader(output)
    box = pdf.pages[0].mediabox
    if len(pdf.pages) != 1 or not math.isclose(float(box.width) / float(box.height), width / height, abs_tol=1e-10):
        raise ValueError("Figure 5 PDF dimensions differ from the single-panel canvas")
    guard()
    return {
        "figure": 5, "simulation_day": first_day, "window_days": window_days, "rows": len(rows),
        "canvas_inches": [width, height], "font": font["family"],
        "axis_font_final_tex_pt": figure_style()["font"]["axis_label_pt"],
        "legend_tick_font_final_tex_pt": figure_style()["font"]["legend_pt"],
        "csv_sha256": hashlib.sha256(raw).hexdigest(),
        "style_sha256": hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest(),
        "pdf_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "elapsed_seconds": time.monotonic() - started,
        "simulation_executed": False,
    }


def plot_dispatch_overlay(csv_path, target):
    """CSV-only compatibility entry point for the existing result plotter."""
    return render(Path(csv_path), Path(target))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path(__file__).with_name("dispatch_results.csv"))
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("fig5_dispatch_comparison.pdf"))
    parser.add_argument("--max-seconds", type=float, default=50)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.out.exists() and not args.force:
        raise FileExistsError(str(args.out) + " exists; use --force for intentional replacement")
    print(json.dumps(render(args.csv, args.out, max_seconds=args.max_seconds), indent=2))


if __name__ == "__main__":
    main()
