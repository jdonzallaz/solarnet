from setuptools import find_packages, setup

with open('README.md') as readme_file:
    readme = readme_file.read()

setup(name='SolarNet',
      version='0.1.0',
      description='Deep Learning for Solar Physics Prediction',
      long_description=readme,
      url='https://gitlab.com/jdonzallaz/solarnet',
      author='Jonathan Donzallaz',
      author_email='jonathan.donzallaz@hefr.ch',
      license='MIT',
      packages=find_packages(),
      entry_points={
          'console_scripts': ['solarnet=solarnet.main:app'],
      },
      # install_requires=[], # Not specified, to be installed from requirements.txt
      zip_safe=False)
