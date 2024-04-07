from setuptools import setup, find_packages
__version__='0.0'
setup(
    name='biotsavartlaw',
    author='Matthew P. Griffiths',
    author_email='mpg@geo.au.dk',
    packages=find_packages('src'),
    package_dir={'':'src'},
    install_requires=['numpy', 'matplotlib', 'contextily', 'pandas', 'utm', 'geopandas', 'textwrap'],
    version=__version__,
    license='MIT',
    description='generate magnetic fields in free space using biot savart law',
    python_requires=">=3.8",
)