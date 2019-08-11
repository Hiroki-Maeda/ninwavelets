from numpy import get_include
from setuptools import setup, find_packages

setup(
    name='nin_wavelets',
    version='0.0.1',
    install_requires=['scipy', 'numpy'],
    package_dir={'nin_wavelets': 'nin_wavelets'},
    packages=find_packages('nin_wavelets'),
    description='My wavelets package',
    long_description='My wavelets package',
    url='https://github.com/uesseu/nin_wavelets',
    author='ninja',
    author_email='sheepwing@kyudai.jp',
    license='MIT',
)

