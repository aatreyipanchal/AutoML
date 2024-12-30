from setuptools import setup, find_packages
from pathlib import Path

def parse_requirements():
    requirements_path = Path(__file__).parent / 'requirements.txt'
    with open(requirements_path) as f:
        return [line.strip() for line in f.readlines() if line.strip()]

setup(
    name='AutoML',
    version='0.1.0',
    packages=find_packages(),
    description='A FastAPI project integrating regression and classification functionalities.',
    author='Aatreyi',
    author_email='aatreyijpanchal@gmail.com',
    install_requires=parse_requirements(),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',  
)
