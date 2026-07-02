"""
Timing test for the sequential HighLevel variant.

Runs HighLevel on a whole directory of cases and reports the total wall time
for the entire process, plus an average per case for reference. Useful for
comparing against the parallel worker version.

Lives in Test/TempTest/. Imports the class through the package path, same as
TempTest2Copy imports EasyMedicalImages, so run it from anywhere the package
is importable.

Point it at a parent folder of case folders when prompted.
"""

import time as tym
from HighLevelPythonVarient import HighLevel


# 1) Load the whole directory, timing the full run end to end.
parent = input("Path to the FOLDER OF cases: ").strip()

total_start = tym.perf_counter()
batch = HighLevel(path_to_folder=parent, verbose=False).run_on_directory()
total_elapsed = tym.perf_counter() - total_start

# 2) Report.
print("\n==== RESULT ====")
print(f"Cases loaded: {len(batch)}")
print(f"Total time:   {total_elapsed:.2f} s")
if batch:
    print(f"Avg per case: {total_elapsed / len(batch):.2f} s")
else:
    print("No cases loaded. Is the parent path right?")