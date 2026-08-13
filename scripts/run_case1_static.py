import subprocess,sys
raise SystemExit(subprocess.call([sys.executable,"scripts/run_grid.py","--case","1",*sys.argv[1:]]))
