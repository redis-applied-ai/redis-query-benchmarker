"""Setup configuration for Redis Query Benchmarker."""

from setuptools import setup, find_packages
import redis_benchmarker

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="redis-query-benchmarker",
    version=redis_benchmarker.__version__,
    author="Redis Labs",
    author_email="support@redis.com",
    description="A production-ready benchmarking tool for Redis search queries",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/redis/redis-query-benchmarker",
    packages=find_packages(exclude=["tests*", "examples*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Software Development :: Testing",
        "Topic :: System :: Benchmark",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-asyncio>=0.21.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "flake8>=6.0.0",
            "mypy>=1.5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "redis-benchmarker=redis_benchmarker.__main__:main",
            "redis-data-generator=redis_benchmarker.data_generator:main",
        ],
    },
    include_package_data=True,
    keywords="redis search benchmark vector database performance testing",
    project_urls={
        "Bug Reports": "https://github.com/redis/redis-query-benchmarker/issues",
        "Source": "https://github.com/redis/redis-query-benchmarker",
        "Documentation": "https://github.com/redis/redis-query-benchmarker#readme",
    },
)