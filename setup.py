from setuptools import setup, find_packages

setup(
    name="vsign",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[],
    include_package_data=True,
)
