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
      version='1.0.12',   # Mar 4, 2021 update
      # version='1.0.11',   # Mar 4, 2021 update
      # version='1.0.10',   # Mar 4, 2021 update
      # version='1.0.9',   # Mar 3, 2021 update, Seq Monroe Update
      # version='1.0.8',   # Mar 3, 2021 update
      # version='1.0.7',   # Mar 2, 2021 update
      # version='1.0.6',   # Mar 1, 2021 update
      # version='1.0.5',   # Feb 26, 2021 update - fix to STAR
      # version='1.0.4',   # Feb 26, 2021 update
      # version='1.0.3',   # Feb 25, 2021 update
      # version='1.0.2',   # Feb 23, 2021 update
      # version='1.0.1',
      packages=find_packages(),
      install_requires=install_requires,
      zip_safe=False,
#      extras_require=extras_require,
      )