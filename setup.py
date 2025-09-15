"""Setup configuration for NPE_LV project."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="npe-lv",
    version="0.1.0",
    author="NPE_LV Team",
    description="Neural Posterior Estimation for Viral Dynamics (TEIRV Model)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-repo/NPE_LV",
    packages=["src"],
    package_dir={"src": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.10",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest",
            "black",
            "flake8",
            "mypy",
        ],
    },
    entry_points={
        "console_scripts": [
            "teirv=src.main:main",
            "npe-lv-train=src.teirv_inference:main",
            "npe-lv-data=src.teirv_data_generation:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)