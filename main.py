import subprocess
import sys

# Set start and end years here
START_YEAR = 1972
END_YEAR = 2024
REDO_EACH_STEP = False
USE_FFT2 = True  # Set to True to use FFT2 for national margin and relative margins

# Import and call modularized functions
from import_csv import run_import_csv
from fourier_FUTURE import run_fourier_future
from calculate_pvi import calculate_pvi

print(f"Running import_csv.run_import_csv({START_YEAR})...")
run_import_csv(START_YEAR)

print(f"Running fourier_FUTURE.run_fourier_future({END_YEAR})...")
run_fourier_future(
    END_YEAR, 
    REDO_EACH_STEP=REDO_EACH_STEP, 
    INCLUDE_HISTORICAL=True, 
    USE_FFT2=USE_FFT2
    )

# Run PVI calculation
#print(f"Calculating PVI from {START_YEAR} to {END_YEAR}...")
#calculate_pvi(START_YEAR, END_YEAR)

# Run hierarchical_clustering.py as a script
print("Running hierarchical_clustering.py ...")
result = subprocess.run([sys.executable, "hierarchical_clustering.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("Error running hierarchical_clustering.py:")
    print(result.stderr)

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

# Run create_state_tables.py as a script
print("Running create_state_tables.py ...")
result = subprocess.run([sys.executable, "create_state_tables.py"], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print("Error running create_state_tables.py:")
    print(result.stderr)