from setuptools import setup, find_packages

setup(
    name='protein_embed_eval',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'scikit-learn',
        'matplotlib',
        'biopython',
        'torch',
        'transformers'
    ],
    author='Leilani Hoffmann',
    description='A package to evaluate biological feature representation in protein embeddings',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown'
)
