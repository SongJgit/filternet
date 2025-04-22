from setuptools import find_packages, setup

setup(
    name='filternet',
    version='0.0.1',
    packages=find_packages(),
    author='Jian Song',
    install_requires=[
        'mmengine', 'wandb', 'pandas', 'numpy', 'matplotlib', 'pycocotools',
        'faster-coco-eval', 'jax', 'torchmetrics'
    ],
    python_requires='>=3.10',
)
