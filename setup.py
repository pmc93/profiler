from setuptools import setup, find_packages
__version__='0.0'
setup(
    name='profiler',
    author='Paul McLachlan',
    author_email='pm@geo.au.dk',
    packages=find_packages('src'),
    package_dir={'':'src'},
    install_requires=['numpy', 'matplotlib', 'contextily', 'pandas', 'utm', 'geopandas', 'textwrap'],
    version=__version__,
    license='MIT',
    description='generate magnetic fields in free space using biot savart law',
    python_requires=">=3.8",
)