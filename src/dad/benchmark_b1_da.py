"""B1-DA: explicit 24 h planning and execution, native feedback only at day end.

Shares the audited MILP and execution/accounting engine with historical B1.
No terminal SOC target or inventory value; no market job starts on import.
"""
from .benchmark_b1 import main as shared_main


def main(argv=None):
    return shared_main(argv,require_day_ahead=True)


if __name__=='__main__':
    raise SystemExit(main())
