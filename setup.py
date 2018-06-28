
from setuptools import setup, find_packages

setup(
    name='ustools',
    version='0.1',
    packages=find_packages(exclude=['tests*']),
    license='Apache License v.2',
    description='An example python package',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas'],
    url='https://github.com/UltraSuite/ultrasuite-tools.git',
    author='Aciel Eshky',
    author_email='aeshky@ed.ac.uk'
)
