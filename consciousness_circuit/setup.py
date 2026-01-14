#!/usr/bin/env python3
"""Setup script for consciousness-circuit package."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text(encoding="utf-8") if readme_path.exists() else ""

setup(
    name="consciousness-circuit",
    version="3.0.0",
    author="VFD-Org",
    author_email="research@vfd.org",
    description="Measure and analyze consciousness-like patterns in transformer LLMs",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vfd-org/consciousness-circuit",
    project_urls={
        "Bug Reports": "https://github.com/vfd-org/consciousness-circuit/issues",
        "Source": "https://github.com/vfd-org/consciousness-circuit",
        "Documentation": "https://github.com/vfd-org/consciousness-circuit#readme",
    },
    packages=find_packages(),
    package_data={
        "consciousness_circuit": [
            "discovered_circuits/*.json",
        ],
    },
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.35.0",
        "numpy>=1.21.0",
        "bitsandbytes>=0.41.0",
    ],
    extras_require={
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "plotly>=5.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
        ],
        "all": [
            "matplotlib>=3.5.0",
            "seaborn>=0.12.0",
            "plotly>=5.0.0",
            "pytest>=7.0.0",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    keywords=[
        "consciousness",
        "llm",
        "transformer",
        "interpretability",
        "mechanistic-interpretability",
        "ai-safety",
        "neural-networks",
    ],
    entry_points={
        "console_scripts": [
            "consciousness-discover=consciousness_circuit.cli:discover_circuit",
            "consciousness-measure=consciousness_circuit.cli:measure_prompt",
        ],
    },
)
