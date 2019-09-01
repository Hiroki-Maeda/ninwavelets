from pathlib import Path
from os import chdir
from subprocess import run

chdir(Path(__file__).parent.parent)
run(['python', 'setup.py', 'build_ext', '--inplace'])
