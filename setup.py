from setuptools import setup

setup(
   name='mooncake',
   version='1.0',
   description='Ai system',
   author='chookity',
   packages=['mooncake'],  # same as name
   install_requires=[
      'numpy>=1.22.0',
      'pandas>=1.2.4',
      'scikit-learn==0.24.1',
      'skorch>=0.10.0',
      'torch>=1.5.0',
      'pytorch-forecasting>=0.9.0'
    ]
)