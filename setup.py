from setuptools import setup, find_packages


INSTALL_REQUIRES = [
    "datasets==3.2.0",
    "torch==1.12.1",
    "transformers==4.30.0",
    "editdistance",
    "camel-tools==1.5.2",
]

setup(
    name="text_editing",
    version="0.1",
    author="Bashar Alhafni",
    author_email="alhafni@nyu.edu",
    maintainer="Bashar Alhafni",
    maintainer_email="alhafni@nyu.edu",
    packages=find_packages(),
    install_requires=INSTALL_REQUIRES,
    license="MIT",
    python_requires=">=3.10"
)
