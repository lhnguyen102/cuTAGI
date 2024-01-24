# cutagi/python_src/__version__.py
import os

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__)))
with open(os.path.join(ROOT_DIR, "version.txt"), "r") as f:
    version = f.readlines()
VERSION = version[0]

__version__ = VERSION
