from setuptools import setup, find_packages

setup(
    name="bioimpedancia-bayesiana",
    version="1.0.0",
    description=(
        "Bayesian Parameter Estimation and Tissue State Classification "
        "for Bioelectrical Impedance Spectroscopy"
    ),
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="PPGEEL-UFSC",
    author_email="guilherme.pintarelli@ufsc.br",
    url="https://github.com/[USERNAME]/bioimpedancia-bayesiana",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24",
        "scipy>=1.10",
        "matplotlib>=3.7",
    ],
    extras_require={
        "dev": ["pytest", "pytest-cov", "black", "flake8"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Medical Science Apps.",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
    entry_points={
        "console_scripts": [
            "bioeis-run=main:main",
            "bioeis-real=main_real_data:main",
        ],
    },
)
