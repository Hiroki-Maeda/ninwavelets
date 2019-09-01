from setuptools import setup, find_packages

setup(
    name='ninwavelets',
    version='0.0.2',
    install_requires=['scipy', 'numpy', 'cupy'],
    package_dir={'ninwavelets': 'ninwavelets'},
    packages=find_packages('ninwavelets'),
    description='Wavelets package',
    long_description='''Analystic wavelets package,
which can perform Generalized Morse and Morlet based cwt.''',
    url='https://github.com/uesseu/ninwavelets',
    author='ninja',
    author_email='sheepwing@kyudai.jp',
    license='MIT',
)
