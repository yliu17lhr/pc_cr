#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 18 23:27:15 2024

@author: yiyanliu
"""

from setuptools import setup, find_packages

setup(
    name="pc_cr",
    version="0.1.0",
    author="Yiyan Liu, Harvey John Burd, Sinan Acikgoz",
    author_email="dimo.liu@gmail.com",
    description="A python package to detect and measure cracks using point clouds as described in Liu's thesis on Displacement and Damage Monitoring for Masonry Buildings Subjected to Ground Movements Induced by Underground Construction.",
    packages=find_packages(),
    install_requires=['matplotlib==3.7.2',
                        'numpy==1.25.1',
                        'open3d==0.16.1',
                        'pandas==2.0.3',
                        'scikit-learn==1.3.0',
                        'scipy==1.11.1',
                        'shapely==2.0.1',
                        'tqdm==4.65.0'],
    python_requires='==3.10.14',
    entry_points={
    'console_scripts': [
        'pc_cr_gui=pc_cr.gui.gui:main',
    ],
},
)
