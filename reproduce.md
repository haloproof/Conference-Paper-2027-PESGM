# Reproduction

Run from the repository root in **Git Bash on Windows**. Python 3.12 and a COPT 8.0.7 license/mode supporting the full models are required.

## 1. Environment setup

~~~bash
py -3.12 -m venv .venv
source .venv/Scripts/activate
python -c 'import sys; print(sys.executable); assert sys.version_info[:2] == (3, 12)'
python -m pip install -r requirements.txt
python -m pip install --no-deps --no-build-isolation -e .
python -m pip check

# Verify inputs and create the required local teacher certificate.
sha256sum -c reproducibility-sha256.txt
mkdir -p reports
python scripts/bounded.py 55 python -m pytest -p no:cacheprovider \
  tests/test_teacher.py::test_native_accounting_restore_convergence
~~~

For later sessions, activate the existing environment with source .venv/Scripts/activate.

## 2. Run M0, B1, B2, B3 and B5 (src/)

These are full experiments, using the supplied prices and frozen model. All five start from the initial battery state. Use new run directories; finish each command before continuing.

~~~bash
model="models/hybrid-fit-1020/stage2-best.pt"
log_dir="logs/$(date +%Y%m%d-%H%M%S)"
mkdir -p "$log_dir"
set -o pipefail

# M0 = paper M1
python -m dad.benchmark_m0_da run \
  --config configs/benchmark-m0-da-multi.json --out runs/m0-da-1031 \
  2>&1 | tee -a "$log_dir/M0.log"

# B1 = paper M2
python -m dad.benchmark_b1_da run \
  --config configs/benchmark-b1-da-multi.json --out runs/b1-da-1031 \
  2>&1 | tee -a "$log_dir/B1.log"

# B2 = paper M3
python -m dad.benchmark_b2_da run \
  --config configs/benchmark-b2-da-multi.json --out runs/b2-da-1031 \
  2>&1 | tee -a "$log_dir/B2.log"

# B3-v4 = paper M5
python -m dad.benchmark_b3_v4_da run \
  --config configs/benchmark-b3-v4-eol.json --model "$model" --out runs/b3-v4-eol \
  2>&1 | tee -a "$log_dir/B3.log"

# B5-v4 = paper M4
python -m dad.benchmark_b5_v4_eol run \
  --config configs/benchmark-b5-v4-eol.json --model "$model" --out runs/b5-v4-eol \
  2>&1 | tee -a "$log_dir/B5.log"
~~~

Check each run's summary.json before exporting. Retirement is native SOH <= 0.8; protective horizons are 7300 days for M0/B1/B2 and 4000 for B3. B5 has no day limit. A paused run is incomplete.

To resume, repeat its command with --resume before the logging pipe. If work allowance is exhausted, also add --add-seconds 604800 --add-steps 100000000 --add-queries 10000. Keep the code, configuration, inputs and certificate unchanged.

## 3. Export tables and figures (results/)

The scripts read the completed runs and write new files under reproduced/, preserving the reference results/. Figure 2 uses the supplied training CSV; Figures 3–6 use the new dispatch runs.

~~~bash
out="reproduced/$(date +%Y%m%d-%H%M%S)"

# Annual tables and lifetime comparison table.
python scripts/bounded.py 60 python scripts/build_tables.py \
  --out-dir "$out/tables" --max-seconds 55

# Figure 2: training curves from the published CSV.
python scripts/bounded.py 60 python scripts/plot_training.py \
  --csv results/training_results.csv --out-dir "$out/fig2" --max-seconds 55

# Figure 3: evaluate the frozen model on the first year of the new M0 run.
python scripts/plot_results.py --figure3-only --evaluate-m1 \
  --out-dir "$out/fig3" --max-seconds 540

# Figures 4–5: lifecycle and daily dispatch.
python scripts/bounded.py 60 python scripts/plot_results.py --dispatch-only \
  --out-dir "$out/fig4-5" --max-seconds 55

# Figure 6: first-year energy and average prices.
python scripts/bounded.py 60 python scripts/plot_first_year_energy_prices.py \
  --out-dir "$out/fig6" --max-seconds 55
~~~

To redraw Figures 2–6 directly from the published CSVs, without running Section 2:

~~~bash
out="reproduced/published-$(date +%Y%m%d-%H%M%S)"
python scripts/bounded.py 55 python scripts/plot_results.py \
  --from-csv results --out-dir "$out" --max-seconds 50
python scripts/bounded.py 55 python scripts/plot_first_year_energy_prices.py \
  --csv results/first_year_energy_prices.csv --out-dir "$out" --max-seconds 50
~~~
