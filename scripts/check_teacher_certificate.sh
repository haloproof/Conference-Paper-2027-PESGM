#!/usr/bin/env bash
# Explicit bounded audit only; never launch production dispatch.
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/activate.sh
mkdir -p reports
audit_dir="$(mktemp -d reports/teacher-audit.XXXXXXXX)"
printf 'Audit output: %s\n' "$audit_dir"
python scripts/bounded.py 55 python -m pytest -q tests/test_teacher.py::test_native_accounting_restore_convergence --junitxml="$audit_dir/tests.xml"
python scripts/bounded.py 10 python -c 'from dad.data import check_certificate; check_certificate("reports/teacher_certificate.json"); print("Teacher certificate valid for current adapter and native SimSES")'
