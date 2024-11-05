import argparse
import os
import sys
import unittest

# path to binding code
sys.path.append(
    os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "build"))
)


def set_cpu_only_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Run only CPU tests")
    args, unknown = parser.parse_known_args()

    # Set an environment variable to indicate if only CPU tests should run
    os.environ["TEST_CPU_ONLY"] = "1" if args.cpu else "0"

    # Pass any remaining args to unittest
    return unknown


if __name__ == "__main__":
    remaining_args = set_cpu_only_flag()

    # Load test files start with `test_` and run them
    test_suite = unittest.defaultTestLoader.discover(start_dir=".", pattern="test_*.py")

    runner = unittest.TextTestRunner(verbosity=2, failfast=True)
    result = runner.run(test_suite)

    # Exit with a non-zero code if there were any test failures
    sys.exit(not result.wasSuccessful())
