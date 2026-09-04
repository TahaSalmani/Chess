import setuptools


with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


__version__ = "0.0.0"
Repo_Name = "Chess"
Author_NAME = "tahasalmani"
SRC_REPO = "Chess_Deeplearning"
AUTHOR_EMAIL = "the.taha.salmani@gmail.com"

setuptools.setup(
    name=SRC_REPO,
    version=__version__,
    author=Author_NAME,
    author_email=AUTHOR_EMAIL,
    description="A modular MLOps package for Chess AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/TahaSalmani/Chess",
    package_dir={"" :"src"},
    packages=setuptools.find_packages(where="src"),


)