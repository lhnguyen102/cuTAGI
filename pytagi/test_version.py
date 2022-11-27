import os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..'))

def main():
    with open(os.path.join(ROOT_DIR, "version.txt"), "r") as f:
        version = f.readlines()

    VERSION = version[0]
    print(VERSION)

if __name__== "__main__":
    main()