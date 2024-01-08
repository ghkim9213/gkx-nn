from setuptools import setup, find_packages

setup(
    name='gkx-nn',
    version='0.1.0',
    packages=find_packages(include=['gkx_nn', 'gkx_nn.*']),
    install_requires=[
        "torch",
        "pandas",
        "requests",
        "tqdm",
        "optuna",
        "mlflow",
    ]
)