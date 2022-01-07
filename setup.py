from setuptools import setup, find_packages

package = "tensorflow_time_series_dataset"
version = "0.1"

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name=package,
    version=version,
    author="Marcel Arpogaus",
    author_email="marcel.arpogaus@gmail.com",
    packages=find_packages("src"),
    package_dir={"": "src"},
    install_requires=["tensorflow==2.6.*", "pandas==1.3.*"],
    description="A TensorFlow dataset from time-series data.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        "Operating System :: OS Independent",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Topic :: Scientific/Engineering",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development",
    ],
    python_requires=">=3.7",
)
