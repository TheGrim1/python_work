#!/usr/bin/env python
#from distutils.core import setup
import os
#https_proxy="http://proxy.esrf.fr:3128"
os.environ["https_proxy"] = "http://proxy.esrf.fr:3128" # for pip
from setuptools import setup


setup(
    name='id01sware',
    version='0.1',
    description='Collection of libraries, scripts and apps from beamline ID01 (ESRF).',
    author='Steven Leake, Marie-Ingrid Richard, Carsten Richter, Edoardo Zatterin, Tao Zhou, et al.',
    author_email='id01@esrf.fr',
    url='https://gitlab.esrf.fr/zatterin/id01sware',
    packages = ['id01lib',
                'id01lib.ptycho',
                'id01lib.plot',
                'id01lib.process',
                'id01lib.xrd'
                ],
    package_data = {
        "id01lib": ["media/*"]},
#    entry_points={ #later
#        'console_scripts': [
#            'id01_microscope_contrast=id01lib.camtools:get_microscope_contrast',
#        ],
#    },
    install_requires=[
                      'numpy',
                      'matplotlib',
                      'scipy',
                      'h5py',
                      'silx>=0.5.0',
                      'xrayutilities',
                      'Pillow',
                      #'SpecClient',
                     ],
    scripts = [
                'bin/CamView',
                'bin/pscan_live',
                'bin/pscan_detector_average',
                'bin/kmap_showroi',
                'bin/id01_microscope_contrast',
                'bin/id01_microscope_cofm',
                ],
    long_description = """
                        Collection of libraries, scripts and apps from beamline ID01 (ESRF).
                     """
     )


