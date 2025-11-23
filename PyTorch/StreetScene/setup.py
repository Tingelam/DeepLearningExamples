"""
Setup script for Street Scene Optimization package.
"""

from setuptools import setup, find_packages
import os

# Read README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="nvidia-street-scene",
    version="0.1.0",
    author="NVIDIA",
    author_email="deeplearning@nvidia.com",
    description="Street Scene Optimization Framework for PyTorch",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/StreetScene",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
        "docs": [
            "sphinx>=4.0",
            "sphinx-rtd-theme>=0.5",
            "myst-parser>=0.15",
        ],
        "monitoring": [
            "tensorboard>=2.5",
            "wandb>=0.10",
            "psutil>=5.8",
        ]
    },
    entry_points={
        "console_scripts": [
            "street-scene-train=scripts.train:main",
            "street-scene-eval=scripts.evaluate:main",
            "street-scene-infer=scripts.inference:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.txt"],
    },
    zip_safe=False,
)