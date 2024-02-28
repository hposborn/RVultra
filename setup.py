from setuptools import setup, find_packages

setup(
   name='RVultra',
   version='0.1',
   description='An RV+ultranest nested sampling code for exoplanet modelling',
   author='Hugh P. Osborn',
   author_email='hugh.osborn@unibe.ch',
   packages = find_packages(),  #same as name
   install_requires=['radvel', 'ultranest', 'scipy', 'numpy'], #external packages as dependencies
)