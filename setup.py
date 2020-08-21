# -*- coding: utf-8 -*-

from setuptools import setup, find_packages


with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='povalt',
    version='0.0.1',
    description='PoValT - Package for potantial validation and training',
    long_description=readme,
    author='Jan Kloppenburg',
    author_email='jan.kloppenburg@aalto.fi',
    url='https://github.com/janklinux/povalt',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)

