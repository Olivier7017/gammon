from setuptools import setup, find_packages

setup(
    name='gammon',
    version='1.0.0',
    description='Gammon package',
    author='Olivier Nadeau',
    author_email='oliviernadeau97@gmail.com',
    packages=find_packages(),
    include_package_data=True,  # <--- include non-code files
    package_data={'gammon.utilities': ['nd3_data/*.npy']},
    install_requires=[
        'ase',
        'numpy',
        # 'pyace',
        # 'tensorflow',
        # 'mace-torch',
        'matplotlib',
        'spglib',
    ],
    entry_points={
        'console_scripts': [
            'gammon=gammon.cli.cli:main',
        ],
    },
)
