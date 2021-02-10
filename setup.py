# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

### dependencies

install_requires = [        
            'numpy>=1.16.5',
            'scipy>=1.3.1',
            'pandas>=0.25.1',
            'pytest>=5.2.1',
            #'matplotlib>=3.1.1'
            #'seaborn>=0.11.1'
            ]

#extras_require = [
#            'bokeh>=1.4.0',
#            'matplotlib>=3.1.1',
#            'numpydoc>=0.9.1',
#            ]
#
setup(
      name='votesim',
      version='1.0.1',
      packages=find_packages(),
      install_requires=install_requires,
      zip_safe=False,
#      extras_require=extras_require,
      )