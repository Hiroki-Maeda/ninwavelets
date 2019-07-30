from pathlib import Path
from os import chdir, getcwd
from subprocess import run
from sys import argv
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument('command', type=str)
args = parser.parse_args()


def compile_cy() -> None:
    current = getcwd()
    chdir(Path(__file__).parent.parent)
    run(['python', 'setup_cy.py', 'build_ext', '--inplace'])
    chdir(current)
    print('Compile completed')


def clear() -> None: (Path(__file__).parent / 'base.c').unlink()


if __name__ == '__main__':
    if args.command == 'compile':
        compile_cy()
    elif args.command == 'clear':
        clear()
