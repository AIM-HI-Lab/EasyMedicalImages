"""Smoke test for an installed EasyMedicalImages wheel.

cibuildwheel runs this against every wheel it builds (see test-command in
pyproject.toml), and you can run it yourself against any environment:

    python Test/wheel_smoke.py

Running it as a script puts Test/ (not the repo root) on sys.path, so it imports
the *installed* package rather than the source tree. The printed paths show
which copy was actually loaded.
"""

import sys
import sysconfig

import EasyMedicalImages as emi

# HighLevel is loaded lazily from the compiled _core extension; touching it
# proves the .pyd/.so in this wheel loads, not just the pure-Python files.
emi.HighLevel
core = sys.modules["EasyMedicalImages._core"]

if sysconfig.get_config_var("Py_GIL_DISABLED"):
    gil = "enabled (something re-enabled it)" if sys._is_gil_enabled() else "disabled"
else:
    gil = "n/a (standard build)"

print(f"EasyMedicalImages {emi.__version__}")
print(f"  package: {emi.__file__}")
print(f"  _core:   {core.__file__}")
print(f"  Python {sys.version.split()[0]} on {sysconfig.get_platform()}, free-threaded GIL: {gil}")
