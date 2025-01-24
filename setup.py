from setuptools import setup, find_packages
import os

setup(
    name="polarbert",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=[
        "torch",
        "numpy",
        "pandas",
        "pyarrow",
        "tqdm",
        "yaml",
    ],
    author="Inar Timiryasov",
    author_email="timinar@gmail.com",
    description="PolarBERT - A foundation model for IceCube",
    long_description=open("README.md").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/timinar/PolarBERT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
