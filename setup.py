from setuptools import setup, find_packages

setup(
    name='gammon',
    version='1.0.0',
    description='Gammon package',
    author='Olivier Nadeau',
    author_email='oliviernadeau97@gmail.com',
    packages=find_packages(),
    install_requires=[
        # 'ase',
        # 'numpy',
        # 'pyace',
        # 'tensorflow',
        # 'mace-torch',
        # 'matplotlib',
    ],
    entry_points={
        'console_scripts': [
            'gammon=gammon.cli.cli:main',
        ],
    },
)
