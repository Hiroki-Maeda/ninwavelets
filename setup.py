from numpy import get_include
from setuptools import setup, find_packages

setup(
    name='nin_wavelets',
    version='0.0.2',
    install_requires=['scipy', 'numpy', 'cupy'],
    package_dir={'nin_wavelets': 'nin_wavelets'},
    packages=find_packages('nin_wavelets'),
    description='Wavelets package',
    long_description='''Analystic wavelets package, which can perform Generalized Morse and Morlet based cwt.''',
    url='https://github.com/uesseu/nin_wavelets',
    author='ninja',
    author_email='sheepwing@kyudai.jp',
    license='MIT',
)

