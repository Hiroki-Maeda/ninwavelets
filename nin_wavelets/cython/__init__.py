from pathlib import Path
from os import chdir, getcwd
from subprocess import run
from sys import argv


def compile_cy() -> None:
    current = getcwd()
    chdir(Path(__file__).parent.parent)
    run(['python', 'setup_cy.py', 'build_ext', '--inplace'])
    chdir(current)
    print('Compile completed')


try:
    from .base import WaveletBase, WaveletMode
except ModuleNotFoundError as error:
    print(error)
    print('''
Is it first time you run the script?
This package need to compiled by cython.
I am going to compile. (Y/n)
''')
    key = input()
    if key.lower != 'n':
        compile_cy()
        print('''
Compiled.

Compile has been done.
If you want to compile again, run below.
=======================================
python -m nin_wavelets.compile clear
python -m nin_wavelets.compile compile
=======================================
''')
from .wavelets import Morse, MorseMNE, Morlet
