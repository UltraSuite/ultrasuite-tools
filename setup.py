
from setuptools import setup, find_packages

setup(
    name='ustools',
    version='0.2dev',
    packages=find_packages(exclude=['tests*']),
    license='Apache License v.2',
    description='Tools to process the UltraSuite data',
    long_description=open('README.md').read(),
    install_requires=['numpy', 'scipy', 'matplotlib', 'pandas', 'skimage', 'python_speech_feature', 'webrtcvad',
                      'samplerate'],
    url='https://github.com/UltraSuite/ultrasuite-tools.git',
    author='Aciel Eshky',
    author_email='aeshky@ed.ac.uk'
)
