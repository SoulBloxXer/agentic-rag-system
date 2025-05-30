from setuptools import setup, find_packages

setup(
    name="retrieval_tool",
    version="0.1.0",
    packages=find_packages(),  # Look for packages in current directory
    install_requires=[
        "numpy>=1.21.0",
        "psutil>=5.9.0",
    ],
    python_requires=">=3.8",
) 