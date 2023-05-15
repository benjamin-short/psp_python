#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 12:35:17 2022

@author: besh2109
"""

import os

CONFIG = {'local_data_dir': 'gong_data/',
          'public_data_dir': 'https://gong2.nso.edu/oQR/zqs/'}

if os.environ.get('SPEDAS_DATA_DIR'):
    CONFIG['local_data_dir'] = os.sep.join([os.environ['SPEDAS_DATA_DIR'], 'gong'])

if os.environ.get('GONG_DATA_DIR'):
    CONFIG['local_data_dir'] = os.environ['GONG_DATA_DIR']