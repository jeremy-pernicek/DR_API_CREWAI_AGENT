#!/usr/bin/env python3
"""
Setup script for DataRobot AI Code Generation Agent
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="dr-api-crewai-agent",
    version="1.0.0",
    author="Jeremy Pernicek",
    author_email="jeremy.pernicek@datarobot.com",
    description="Multi-LLM DataRobot API code generation agent with hybrid synthesis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jeremy-pernicek/DR_API_CREWAI_AGENT",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Data Scientists",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Code Generators",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
            "mypy>=1.0.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "dr-generate=examples.quick_generate:main",
            "dr-interactive=examples.interactive_session:main",
            "dr-test=examples.system_test:main",
        ],
    },
    include_package_data=True,
    package_data={
        "agent_crewai": [
            "data/indexes/*",
            "data/scraped/*",
        ],
    },
    keywords=[
        "datarobot",
        "machine-learning",
        "code-generation", 
        "ai-agents",
        "multi-llm",
        "claude",
        "gpt-4",
        "gemini",
        "synthesis",
        "automation"
    ],
    project_urls={
        "Bug Reports": "https://github.com/jeremy-pernicek/DR_API_CREWAI_AGENT/issues",
        "Source": "https://github.com/jeremy-pernicek/DR_API_CREWAI_AGENT",
        "Documentation": "https://github.com/jeremy-pernicek/DR_API_CREWAI_AGENT/wiki",
    },
)