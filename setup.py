from setuptools import setup


with open("README.md", "r") as fh:
    long_description = fh.read()


setup(
    name="mfmodel",
    version="0.0.1",
    packages=["mfmodel"],
    license="GPLv3",
    description="Fitting multilevel factor models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    # install_requires=[
    #     "numpy >= 1.22.2",
    #     "scipy >= 1.8.0",
    #     "cvxpy >= 1.2.0",
    #     "matplotlib >= 3.5.0"],
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
 
# conda create -n mfm python=3.9
# conda activate mfm
# conda install numpy scipy matplotlib seaborn pandas numba networkx dask
# pip install torch==2.0.0 ipython
# python setup.py install
# conda config --prepend channels conda-forge
# pip install osmnx
# conda install scipy matplotlib seaborn pandas numba networkx dask
# conda install profilehooks memory_profiler