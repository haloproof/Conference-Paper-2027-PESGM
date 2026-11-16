"""Hard wall-clock watchdog for foreground short checks; no production launch."""
import subprocess
import sys

seconds=float(sys.argv[1])
if not 0 < seconds <= 60:
    raise SystemExit('short command timeout must be <=60 seconds')
try:
    command=sys.argv[2:]
    if command[0] == 'python':
        command[0]=sys.executable  # Windows executable search can otherwise find base Python.
    result=subprocess.run(command,timeout=seconds)
except subprocess.TimeoutExpired:
    raise SystemExit('STOPPED: short command exceeded its wall-clock budget')
raise SystemExit(result.returncode)
