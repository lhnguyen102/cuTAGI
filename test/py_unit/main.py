import argparse
import os
import unittest


def set_cpu_only_flag():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Run only CPU tests")
    args, unknown = parser.parse_known_args()

    # Set an environment variable to indicate if only CPU tests should run
    os.environ["CPU_ONLY"] = "1" if args.cpu else "0"

    # Pass any remaining args to unittest
    return unknown


if __name__ == "__main__":
    remaining_args = set_cpu_only_flag()

    test_suite = unittest.defaultTestLoader.discover(start_dir=".", pattern="test_*.py")
    runner = unittest.TextTestRunner()
    runner.run(test_suite)
