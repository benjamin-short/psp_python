#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 12:40:08 2021

@author: benshort
"""

import os
import pyspedas as pys

CONFIG = {'local_data_dir': 'psp_data/',
          'public_data_dir': 'https://spdf.gsfc.nasa.gov/pub/data/psp/',
          'secure_data_dir': 'http://research.ssl.berkeley.edu/data/psp/data/sci/'}

"""
if os.environ.get('PSP_FIELDS_ID'):
    CONFIG['remote_data_dir'] = 'http://research.ssl.berkeley.edu/data/psp/data/sci/'
else:
    print("Fields ID not set, using public data")
"""

# override local data directory with environment variables
if os.environ.get('SPEDAS_DATA_DIR'):
    CONFIG['local_data_dir'] = os.sep.join([os.environ['SPEDAS_DATA_DIR'], 'psp'])

if os.environ.get('PSP_DATA_DIR'):
    CONFIG['local_data_dir'] = os.environ['PSP_DATA_DIR']
    
    
enc_flt = [pys.time_float(['2018-08-23 05:51:00','2019-01-20 01:03:00']),
           pys.time_float(['2019-01-20 01:04:00','2019-06-18 20:15:00']),
           pys.time_float(['2019-06-18 20:15:00','2019-11-15 15:26:00']),
           pys.time_float(['2019-11-15 15:27:00','2020-04-03 09:00:00']),
           pys.time_float(['2020-04-03 09:01:00','2020-08-11 07:46:00']),
           pys.time_float(['2020-08-11 07:47:00','2020-12-01 08:39:00']),
           pys.time_float(['2020-12-01 08:40:00','2021-03-23 17:03:00']),
           pys.time_float(['2021-03-23 17:04:00','2021-05-23 17:03:00'])] #encounter list 1-8
    
per_flt = [pys.time_float('2018-11-06 03:27:00'),
           pys.time_float('2019-04-04 22:39:00'),
           pys.time_float('2019-09-01 17:50:00'),
           pys.time_float('2020-01-29 09:37:00'),
           pys.time_float('2020-06-07 08:23:00'),
           pys.time_float('2020-09-27 09:16:00'),
           pys.time_float('2021-01-17 17:40:00'),
           pys.time_float('2021-04-29 08:48:00')] #perihelion dates for encounters 1-8
    
per_dist_lst = [35.6,35.6,35.6,27.8,27.8,20.3,20.3,15.9] #perihelion distances in units of Rs (solar radii)
