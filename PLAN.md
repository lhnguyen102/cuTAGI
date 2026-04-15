# PLAN

- [x] Identify existing test suites and current baseline behavior.
- [x] Confirm Python unit-test baseline currently fails in this environment due missing runtime deps/build artifacts (`numpy`, `cutagi`).
- [x] Fix correctness issues found in test code:
  - [x] `test/py_unit/test_mnist.py`: compute average error rate using the actual number of collected samples.
  - [x] `test/cpp_unit/main.cpp`: fix incorrect `--cpu` mode status message.
- [x] Validate changed test files with targeted checks.

## Verification commands

- Baseline (before fixes):  
  `python -m unittest discover -s test/py_unit -p 'test_*.py'`  
  Result: fails in this environment because `numpy` and built module `cutagi` are unavailable.
- Post-change targeted checks:
  - `python -m py_compile test/py_unit/test_mnist.py`
  - `python -m py_compile test/py_unit/main.py`
  Result: pass.
