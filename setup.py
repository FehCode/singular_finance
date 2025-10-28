#!/usr/bin/env python3
"""
Setup script for Singular Finance library
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="singular-finance",
    version="1.0.0",
    author="Singular Finance Team",
    author_email="contact@singularfinance.com",
    description="Uma biblioteca Python para análise financeira corporativa, valuation e indicadores",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/singular-finance/singular-finance",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Office/Business :: Financial :: Investment",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
        ],
    },
    keywords="finance, valuation, financial-analysis, corporate-finance, indicators",
    project_urls={
        "Bug Reports": "https://github.com/singular-finance/singular-finance/issues",
        "Source": "https://github.com/singular-finance/singular-finance",
        "Documentation": "https://singular-finance.readthedocs.io/",
    },
)
