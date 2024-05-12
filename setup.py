from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Intended Audience :: Science/Research",
    "Programming Language :: Python :: 3 :: Only",  # Specify Python 3 only
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "License :: OSI Approved :: MIT License",
    "Operating System :: OS Independent",
    "Topic :: Scientific/Engineering",
    "Topic :: Software Development",
]

setup(
    name="iBAT",
    packages=find_packages(),
    version="0.1.0-rc1",
    description="A Python framework to enhance the real-time bus arrival time prediction in heterogeneous traffic condition by addressing concept drift.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aaivu/Incremental-Online-Learning-for-BAT-Prediction/",
    keywords=['GPS', 'Travel Time', 'Public Transit', 'Heterogeneous Traffic Condition', 'ITS (Intelligent Transportation System)', 'Dwell Time', 'Incremental Learning', 'Concept Drift'],
    author="Aaivu",
    author_email='helloaaivu@gmail.com',
    license='MIT',
    classifiers=classifiers,
    python_requires=">=3.6",
    install_requires=['pandas', 'numpy', 'Cython', 'matplotlib', 'scikit-learn', 'xgboost', 'frouros', 'scikit-multiflow'],
    project_urls={
        "Homepage": "https://github.com/aaivu/Incremental-Online-Learning-for-BAT-Prediction/",
        "Source": "https://github.com/aaivu/Incremental-Online-Learning-for-BAT-Prediction/",
        "Download": "https://github.com/aaivu/Incremental-Online-Learning-for-BAT-Prediction/",
        "Documentation": "https://github.com/aaivu/Incremental-Online-Learning-for-BAT-Prediction/blob/master/PACKAGE_DESCRIPTION.md",
        "Bug Tracker": "https://github.com/aaivu/Incremental-Online-Learning-for-BAT-Prediction/issues",
    }
)
