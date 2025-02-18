from setuptools import setup, find_packages

setup(
    name="CoMOcG",
    version="0.1",
    packages=find_packages(where="App"),
    package_dir={"": "App"},
    include_package_data=True,
)
