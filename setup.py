from setuptools import find_packages, setup

setup(
    name="arcmentations",
    author="mo",
    version="0.1",
    install_requires=[
        "numpy",
        "arc",
        "matplotlib",
    ],
    packages=["arcmentations"] + ["arcmentations." + pkg for pkg in find_packages("arcmentations")]
)