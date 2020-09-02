#!/usr/bin/env python3

# -*- coding: utf-8 -*-

from distutils.core import setup

setup(
    name="lareg",
    version="0.1",
    description="""Linear least squares fit""",
    long_description="""Function for linear least squares fit with
    arbitrary functions and calculation of parameter errors.
    """,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
    python_requires='>=3.6',
    license="GNU GPLv3",
    install_requires=["numpy"],
    author="Finn Krein",
    author_email="finn@krein.moe",
    url="https://github.com/sents/lareg",
    packages=["lareg"],
    entry_points={},
)
