#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:35:17 2022

@author: besh2109
"""

import os

CONFIG = {'local_data_dir': 'sunpy/',
          'gong_data_dir': 'https://nispdata.nso.edu/ftp/oQR/zqs/',
          'adapt_data_dir': 'https://gong.nso.edu/adapt/maps/gong/',
          'hmi_data_dir': 'jsoc.stanford.edu/'
          }

if os.environ.get('SUNPY_DATA_DIR'):
    CONFIG['local_data_dir'] = os.environ['SUNPY_DATA_DIR']

if os.environ.get('GONG_DATA_DIR'):
    CONFIG['local_data_dir'] = os.environ['GONG_DATA_DIR']