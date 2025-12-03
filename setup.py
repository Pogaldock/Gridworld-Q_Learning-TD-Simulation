"""Setup configuration for Gridworld Q-Learning TD Simulation."""
from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="gridworld-qlearning-td",
    version="1.0.0",
    author="Gridworld Q-Learning Contributors",
    description="Interactive gridworld environment for Q-learning reinforcement learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Pogaldock/Gridworld-Q_Learning-TD-Simulation",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
    ],
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "gridworld-qlearning=Gridworld_Qlearning:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.png"],
    },
)
