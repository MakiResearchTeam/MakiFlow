from setuptools import setup
import setuptools

setup(
    name='MakiFlow',
    packages=setuptools.find_packages(),
    version='0.0.1',
    description='todo',
    long_description='todo',
    author='todo',
    author_email='todo',
    url='https://github.com/oKatanaaa/MakiFlow',                            # URL to GitHub repo
    # download_url='https://github.com/mdbloice/Augmentor/tarball/0.1.1',   # Get this using git tag
    include_package_data=True,  # This will include all files in MANIFEST.in in the package when installing.
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    # zip_safe=False # Check this later.
)