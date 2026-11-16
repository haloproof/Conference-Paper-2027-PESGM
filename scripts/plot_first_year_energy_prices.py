"""First-year delivered energy and energy-weighted prices; saved reports only.

Git Bash (project root):
  source "scripts/activate.sh"
  timeout --signal=KILL 55s python "results/plot_first_year_energy_prices.py" --max-seconds 50
  timeout --signal=KILL 55s python "results/plot_first_year_energy_prices.py" --csv "results/first_year_energy_prices.csv" --force
No teacher, model, optimizer, training, or simulation is imported or executed.
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
from plot_training import setup_publication_style, figure_font, STYLE_PATH

RUNS = {
    "M1": "m0-da-1031", "M2": "b1-da-1031", "M3": "b2-da-1031",
    "M4": "b5-v4-eol", "M5": "b3-v4-eol",
}
FIELDS = [
    "method", "source_run", "days", "intervals", "charge_energy_kwh",
    "discharge_energy_kwh", "average_charge_price_eur_per_mwh",
    "average_discharge_price_eur_per_mwh", "charge_expenditure_eur",
    "discharge_revenue_eur", "arbitrage_revenue_eur",
]
WIDTH, HEIGHT = 8.0, 3.4


def close(a, b, label):
    if not math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-7):
        raise ValueError(f"{label}: {a} != {b}")


def collect(root, guard):
    rows, hashes = [], {}
    common_prices = None
    for method, run in RUNS.items():
        ch, dis, buy, sell, cash = [], [], [], [], []
        ordered = hashlib.sha256()
        prices_hash = hashlib.sha256()
        for number in range(1, 366):
            guard()
            path = root/"runs"/run/"days"/f"{number:05d}.json"
            raw = path.read_bytes()
            ordered.update(path.name.encode("utf-8"))
            ordered.update(hashlib.sha256(raw).digest())
            day = json.loads(raw)
            if day["index"] != number-1 or not day["complete_day"] or len(day["intervals"]) != 96:
                raise ValueError(f"{method}: incomplete or out-of-order first-year day")
            ec, ed, bc, sd = [], [], [], []
            for interval in day["intervals"]:
                p, dt, price = (interval["delivered_ac_kw"], interval["dt_h"],
                                interval["price_eur_kwh"])
                if not all(math.isfinite(v) for v in (p, dt, price)) or dt != .25:
                    raise ValueError("Finite saved quarter-hour observations required")
                q_ch, q_dis = max(-p, 0.)*dt, max(p, 0.)*dt
                ec.append(q_ch); ed.append(q_dis)
                # Negative prices remain signed in the numerators.
                bc.append(price*q_ch); sd.append(price*q_dis)
                prices_hash.update(json.dumps(
                    [interval["start_utc"], interval["end_utc"], dt, price],
                    separators=(",", ":")).encode("utf-8"))
            qch, qdis, cost, revenue = map(math.fsum, (ec, ed, bc, sd))
            close(qch, day["delivered_charge_kwh"], method+" delivered charge")
            close(qdis, day["delivered_discharge_kwh"], method+" delivered discharge")
            close(revenue-cost, day["revenue_eur"], method+" actual cash")
            ch.append(qch); dis.append(qdis); buy.append(cost); sell.append(revenue)
            cash.append(day["revenue_eur"])
        qch, qdis, cost, revenue = map(math.fsum, (ch, dis, buy, sell))
        if min(qch, qdis) <= 0:
            raise ValueError("Energy-weighted mean undefined without directional energy")
        close(revenue-cost, math.fsum(cash), method+" annual cash")
        row = dict(zip(FIELDS, [method, run, 365, 35040, qch, qdis,
            1000*cost/qch, 1000*revenue/qdis, cost, revenue, math.fsum(cash)]))
        rows.append(row)
        hashes[method] = ordered.hexdigest()
        if common_prices is None:
            common_prices = prices_hash.hexdigest()
        elif prices_hash.hexdigest() != common_prices:
            raise ValueError("Methods do not share the same first-year price sequence")
    return rows, dict(ordered_daily_reports_sha256=hashes,
                      common_first_year_price_sha256=common_prices)


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if [r["method"] for r in rows] != list(RUNS):
        raise ValueError("CSV must contain exactly one row for each of M1-M5")
    for row in rows:
        if row["source_run"] != RUNS[row["method"]]:
            raise ValueError("CSV source mapping changed")
        for key in FIELDS[2:]:
            row[key] = float(row[key])
        if row["days"] != 365 or row["intervals"] != 35040:
            raise ValueError("CSV must represent 365 full days")
        if not all(math.isfinite(row[k]) for k in FIELDS[2:]):
            raise ValueError("Nonfinite CSV value")
        close(row["average_charge_price_eur_per_mwh"],
              1000*row["charge_expenditure_eur"]/row["charge_energy_kwh"], "charge mean")
        close(row["average_discharge_price_eur_per_mwh"],
              1000*row["discharge_revenue_eur"]/row["discharge_energy_kwh"], "discharge mean")
        close(row["arbitrage_revenue_eur"],
              row["discharge_revenue_eur"]-row["charge_expenditure_eur"], "cash")
    return rows


def plot(rows, target):
    setup_publication_style()
    fig, energy_ax = plt.subplots(figsize=(WIDTH, HEIGHT))
    price_ax = energy_ax.twinx()
    fig.subplots_adjust(left=.145, right=.855, bottom=.16, top=.75)
    price_ax.spines["right"].set_visible(True)
    x = np.arange(len(rows), dtype=float)
    width, offset = .22, .25
    charge = np.array([r["charge_energy_kwh"] for r in rows])
    discharge = np.array([r["discharge_energy_kwh"] for r in rows])
    charge_price = np.array([r["average_charge_price_eur_per_mwh"] for r in rows])
    discharge_price = np.array([r["average_discharge_price_eur_per_mwh"] for r in rows])
    edges = dict(edgecolor="#303030", linewidth=.65)
    a = energy_ax.bar(x-offset, charge, width, color="#4C78A8", label="Charge", **edges)
    b = energy_ax.bar(x-offset, discharge, width, bottom=charge,
                      color="#A6CEE3", label="Discharge", **edges)
    c = price_ax.bar(x, charge_price, width, color="#F2A541",
                     label="Charge price", **edges)
    d = price_ax.bar(x+offset, discharge_price, width, color="#59A14F",
                     label="Discharge price", **edges)
    energy_ax.set_xlim(-.6, len(rows)-.4)
    energy_ax.set_xticks(x, list(RUNS))
    energy_ax.set_ylabel("Energy (kWh)")
    price_ax.set_ylabel("Price (EUR/MWh)")
    energy_max = math.ceil(float(np.max(charge+discharge))/5000)*5000
    energy_ax.set_ylim(0, energy_max)
    step = 5000 if energy_max <= 20000 else 10000
    energy_ax.set_yticks(np.arange(0, energy_max+.5*step, step))
    low = min(0., math.floor(float(min(charge_price.min(), discharge_price.min()))/20)*20)
    high = math.ceil(float(max(charge_price.max(), discharge_price.max()))/20)*20
    price_ax.set_ylim(low, high)
    price_step = 50 if high-low >= 150 else 20
    price_ax.set_yticks(np.arange(low, high+1, price_step))
    energy_ax.grid(axis="y", color="#D8D8D8", linewidth=.55, alpha=.7)
    energy_ax.set_axisbelow(True)
    for ax in (energy_ax, price_ax):
        ax.tick_params(direction="out", length=3, width=.7)
    fig.legend([a, b, c, d], ["Charge", "Discharge", "Charge price", "Discharge price"],
               loc="upper center", bbox_to_anchor=(.5,.995), ncol=2,
               frameon=False, columnspacing=1.1, handlelength=1.35,
               handletextpad=.45, labelspacing=.18, fontsize=figure_font("legend"))
    fig.savefig(target, format="pdf", metadata={
        "Title": "Figure 6. First-year energy and energy-weighted transaction prices",
        "Subject": "Saved teacher-delivered AC energy, days 1-365. Energy-weighted means retain negative prices. Energy and prices use separate y axes.",
        "Author": "Pengfei He", "Creator": "results/plot_first_year_energy_prices.py",
    })
    plt.close(fig)
    pdf = PdfReader(target)
    box = pdf.pages[0].mediabox
    if len(pdf.pages) != 1 or not math.isclose(float(box.width), WIDTH*72) or not math.isclose(float(box.height), HEIGHT*72):
        raise ValueError("Unexpected PDF canvas")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path)
    parser.add_argument("--csv", type=Path, help="Replot from the public five-row CSV only")
    parser.add_argument("--max-seconds", type=float, default=50.)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    began = time.monotonic()
    def guard():
        if time.monotonic()-began > args.max_seconds:
            raise TimeoutError("First-year figure export exceeded its time limit")
    if not math.isfinite(args.max_seconds) or args.max_seconds <= 0:
        raise ValueError("Positive finite time limit required")
    root = args.root.resolve()
    out = (args.out_dir or root/"results").resolve()
    out.mkdir(parents=True, exist_ok=True)
    pdf_path = out/"fig6_first_year_energy_prices.pdf"
    csv_path = out/"first_year_energy_prices.csv"
    targets = [pdf_path] + ([] if args.csv else [csv_path])
    if not args.force:
        for path in targets:
            if path.exists():
                raise FileExistsError(str(path)+" already exists; use --force intentionally")
    if args.csv:
        rows = read_csv(args.csv)
        source = dict(csv_source=str(args.csv), csv_sha256=hashlib.sha256(args.csv.read_bytes()).hexdigest())
    else:
        rows, source = collect(root, guard)
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader(); writer.writerows(rows)
        read_csv(csv_path)
    guard()
    plot(rows, pdf_path)
    guard()
    print(json.dumps(dict(
        scope="Saved first-year records only; no model or teacher execution",
        source=source, statistics=rows, elapsed_seconds=time.monotonic()-began,
        weighting="sum(price_EUR_per_kWh * directional_delivered_energy_kWh) / sum(directional_delivered_energy_kWh), multiplied by 1000 for EUR/MWh",
        energy_basis="actual delivered AC power integrated using saved dt_h",
        negative_prices="retained without clipping",
        axes=dict(left="Energy (kWh)", right="Price (EUR/MWh)"),
        pdf=str(pdf_path), pdf_sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest(),
        style_sha256=hashlib.sha256(STYLE_PATH.read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    ), indent=2))


if __name__ == "__main__":
    main()
