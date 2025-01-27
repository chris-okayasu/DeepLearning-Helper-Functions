from setuptools import setup, find_packages

setup(
    name="ai_helpers",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0.0",
        # Add other dependencies if needed
    ],
)