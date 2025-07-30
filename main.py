import subprocess
import sys

# Set start and end years here
START_YEAR = 1936
END_YEAR = 2044

# Import and call modularized functions
from import_csv import run_import_csv
from fourier_FUTURE import run_fourier_future

print(f"Running import_csv.run_import_csv({START_YEAR})...")
run_import_csv(START_YEAR)

print(f"Running fourier_FUTURE.run_fourier_future({END_YEAR})...")
run_fourier_future(END_YEAR)

# Run future_EC.py as a script
print("Running future_EC.py ...")
result = subprocess.run([sys.executable, "future_EC.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("Error running future_EC.py:")
    print(result.stderr)

# Run plot_state_trends.py as a script
print("Running plot_state_trends.py ...")
result = subprocess.run([sys.executable, "plot_state_trends.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("Error running plot_state_trends.py:")
    print(result.stderr)
