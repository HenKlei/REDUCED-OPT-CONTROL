from setuptools import setup, find_packages


dependencies = [
    'numpy',
    'scipy',
    'matplotlib',
    'pymor',
    'scikit-learn',
    'VKOGA @ git+https://github.com/GabrieleSantin/VKOGA.git',
    'torch',
]

setup(
    name='ml-control',
    version='0.1.0',
    description='Python implementation of a neural network for optimal control of parameter-dependent LTI systems',
    author='Hendrik Kleikamp, Martin Lazar, Cesare Molinari',
    author_email='hendrik.kleikamp@uni-muenster.de, mlazar@unidu.hr, cecio.molinari@gmail.com',
    maintainer='Hendrik Kleikamp',
    maintainer_email='hendrik.kleikamp@uni-muenster.de',
    packages=find_packages(),
    install_requires=dependencies,
)
