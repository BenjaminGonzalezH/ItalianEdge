from setuptools import setup, find_packages

setup(
    name="ParetoInsight",
    version="0.8",
    packages=find_packages(where="App"),
    package_dir={"": "App"},
    include_package_data=True,
)
