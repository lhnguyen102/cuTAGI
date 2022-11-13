#cutagi/python_src/__version__.py
import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))
with open(os.path.join(ROOT_DIR, "CMakeLists.txt"), "r") as cmakelists:
    for line in cmakelists.readlines():
        if line.strip().startswith("VERSION"):
            VERSION = line.split("VERSION")[-1].strip()
            break

__version__ = VERSION