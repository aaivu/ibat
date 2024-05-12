from setuptools import find_packages, setup
from codecs import open
from os import path


HERE = path.abspath(path.dirname(__file__))

with open(path.join(HERE, "README.md"), "r", encoding="utf-8") as f:
    long_description = f.read()

with open(path.join(HERE, "requirements.txt"), "r", encoding="utf-8") as f:
    requirements = f.read().splitlines()

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development",
]

setup(
    name="ibat",
    version="0.1.0-rc1",
    description="Python framework designed to improve the robustness of real-time bus arrival/dwell time \
prediction models in heterogeneous traffic conditions by addressing real concept drift.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaivu/ibat/",
    keywords=[
        "Bus arrival/dwell time prediction",
        "Hybrid batch processing",
        "Concept drift handling",
        "Active strategy",
        "Incremental learning",
    ],
    author="Aaivu",
    author_email="helloaaivu@gmail.com",
    license="MIT",
    python_requires=">=3.9",
    classifiers=classifiers,
    packages=find_packages(exclude=["tests", "tests.*"]),
    include_package_data=True,
    package_data={"": ["datasets/_datasets/*.csv"]},
    install_requires=requirements,
    project_urls={
        "Source Code": "https://github.com/aaivu/ibat",
        "Download": "https://github.com/aaivu/ibat/releases",
        "Documentation": "https://github.com/aaivu/ibat/blob/master/README.md",
        "Bug Tracker": "https://github.com/aaivu/ibat/issues",
    },
)
