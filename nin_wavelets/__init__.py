from pathlib import Path
from os import chdir, getcwd
from subprocess import run

current = getcwd()
if (Path(__file__).parent / 'base.c').exists() is False:
    chdir(Path(__file__).parent.parent)
    print('Is it first time import?')
    print('Compiling')
    run(['python', 'setup_cy.py', 'build_ext', '--inplace'])


from .wavelets import WaveletBase
from .morse import Morse, MorseMNE, Morlet
