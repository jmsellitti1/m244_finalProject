import subprocess
# Run all python scripts to load data, build models, and analyze results
subprocess.run(["python3", "scripts/load_data.py"], check=True)
print()
subprocess.run(["python3", "scripts/player_xba.py"], check=True)
print()
subprocess.run(["python3", "scripts/team_xba.py"], check=True)
print()
subprocess.run(["python3", "scripts/xba_classification.py"], check=True)
print()
subprocess.run(["python3", "scripts/xba_regression.py"], check=True)