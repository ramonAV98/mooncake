from setuptools import setup, find_packages

name = 'mooncake'
version = '1.0'
author = 'chookity'
description = 'Ai system'

setup(
   name=name,
   version=version,
   author=author,
   packages=find_packages(),
   description=description,
   python_requires='<3.10',
   install_requires=[
      'numpy>=1.22.0',
      'pandas>=1.2.4',
      'scikit-learn==0.24.1',
      'skorch>=0.10.0',
      'torch>=1.5.0',
      'pytorch-forecasting>=0.9.0'
    ]
)