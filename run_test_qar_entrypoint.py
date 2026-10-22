import pathlib
import runpy
import sys


SCRIPT_PATH = pathlib.Path(__file__).with_name("test_qar.py")
sys.argv[0] = str(SCRIPT_PATH)
runpy.run_path(str(SCRIPT_PATH), run_name="__main__")
