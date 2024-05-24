# -*- coding: utf-8 -*-

# Learn more: https://github.com/kennethreitz/setup.py

from setuptools import setup, find_packages


with open('README.md') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='traderl',
    version='0.1.0',
    description='Trade Reinforcement Learning project',
    long_description=readme,
    author='Luiz Moura',
    author_email='muniz.mourafelipe@gmail.com',
    url='https://github.com/LuizFelipeM/traderl',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)