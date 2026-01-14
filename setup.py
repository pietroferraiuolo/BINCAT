#!/usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="bincatsim",
    version="0.1.0",
    description="Binary star observation simulator for GAIA",
    author="Pietro Ferraiuolo",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "astropy",
        "matplotlib",
        "poppy",
        "xupy",
    ],
    include_package_data=True,
)
