from setuptools import setup, find_packages

setup(
    name='rtx',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy', 'torch', 'torchmetrics', 'tqdm', 'transformers', 'statsmodels'
    ],
)
