from setuptools import setup
import setuptools

setup(
    name='MakiFlow',
    packages=setuptools.find_packages(),
    version='1.0.1',
    description='Machine learning framework made by students of Samara National Research University.'
                'It is made primarily to extend our knowledge and get an instrument that accelerates out studies.',
    long_description='...',
    author='Kilbas Igor, Mukhin Artem, Gribanov Danil',
    author_email='igor.kilbas@mail.ru',
    url='https://github.com/oKatanaaa/MakiFlow',
    include_package_data=True,  # This will include all files in MANIFEST.in in the package when installing.
    classifiers=[
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ], install_requires=[]
)