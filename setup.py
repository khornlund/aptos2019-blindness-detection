import io
import os
import re

from setuptools import find_packages
from setuptools import setup


def read(filename):
    filename = os.path.join(os.path.dirname(__file__), filename)
    text_type = type(u"")
    with io.open(filename, mode="r", encoding='utf-8') as fd:
        return re.sub(text_type(r':[a-z]+:`~?(.*?)`'), text_type(r'``\1``'), fd.read())

requirements = [
    # use environment.yaml
]

setup(
    name="aptos",
    version="0.0.1",
    url="https://github.com/khornlund/aptos2019-blindness-detection",
    license='MIT',
    author="Karl Hornlund",
    author_email="karlhornlund@gmail.com",
    description="https://www.kaggle.com/c/aptos2019-blindness-detection/overview",
    long_description=read("README.rst"),
    packages=find_packages(exclude=('tests',)),
    entry_points={
        'console_scripts': [
            'aptos=aptos.cli:cli'
        ]
    },
    install_requires=requirements,
    classifiers=[
        'Programming Language :: Python',
        'Programming Language :: Python :: 3.6'
    ],
)
