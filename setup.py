from setuptools import setup

with open("requirements.txt") as f:
    requirements = f.readlines()

if __name__ == "__main__":
    setup(install_requires=requirements)
