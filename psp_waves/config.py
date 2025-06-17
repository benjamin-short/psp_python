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
    
    
enc_flt = [pys.time_float(['2018-08-23 05:51:00','2019-01-20 01:03:00']),# 1
           pys.time_float(['2019-01-20 01:04:00','2019-06-18 20:14:30']),# 2
           pys.time_float(['2019-06-18 20:14:30','2019-11-15 13:43:30']),# 3
           pys.time_float(['2019-11-15 13:43:30','2020-04-03 09:00:00']),# 4
           pys.time_float(['2020-04-03 09:01:00','2020-08-02 08:49:30']),# 5
           pys.time_float(['2020-08-02 08:49:30','2020-11-22 13:28:00']),# 6
           pys.time_float(['2020-11-22 13:28:00','2021-03-09 13:14:00']),# 7
           pys.time_float(['2021-03-09 13:14:00','2021-06-19 13:59:30']),# 8
           pys.time_float(['2021-06-19 13:59:30','2021-09-23 17:03:00']),# 9
           pys.time_float(['2021-09-30 13:47:00','2022-01-08 12:00:30']),#10
           pys.time_float(['2022-01-08 12:00:30','2022-04-14 19:14:30']),#11
           pys.time_float(['2022-04-14 19:14:30','2022-07-20 02:27:30']),#12
           pys.time_float(['2022-07-20 02:27:30','2022-10-24 09:40:00']),#13
           pys.time_float(['2022-10-24 09:40:00','2023-01-28 16:53:00']),#14
           pys.time_float(['2023-01-28 16:53:00','2023-05-05 00:08:00']),#15
           pys.time_float(['2023-05-05 00:08:00','2023-08-10 01:37:00']),#16
           pys.time_float(['2023-08-10 01:37:00','2023-11-13 00:11:00']),#17
           pys.time_float(['2023-11-13 00:11:00','2024-02-13 01:37:00']),#18
           pys.time_float(['2024-02-13 01:37:00','2024-05-15 03:03:00']),#19
           pys.time_float(['2024-05-15 03:03:00','2024-08-15 04:29:30']),#20
           pys.time_float(['2024-08-15 04:29:30','2024-11-11 20:27:00']),#21
           pys.time_float(['2024-11-11 20:27:00','2025-02-06 17:03:00']),#22
           pys.time_float(['2025-02-06 17:03:00','2025-05-06 03:47:00']),#23
           pys.time_float(['2025-05-06 03:47:00','2025-08-02 14:31:00'])] #aphelion dates for nominal mission

    
per_flt = [pys.time_float('2018-11-06 03:27:00'),#1
           pys.time_float('2019-04-04 22:39:00'),#2
           pys.time_float('2019-09-01 17:50:00'),#3
           pys.time_float('2020-01-29 09:37:00'),#4
           pys.time_float('2020-06-07 08:23:00'),#5
           pys.time_float('2020-09-27 09:16:00'),#6
           pys.time_float('2021-01-17 17:40:00'),#7
           pys.time_float('2021-04-29 08:48:00'),#8
           pys.time_float('2021-08-09 19:11:00'),#9
           pys.time_float('2021-11-21 08:23:00'),#10
           pys.time_float('2022-02-25 15:38:00'),#11
           pys.time_float('2022-06-01 22:51:00'),#12
           pys.time_float('2022-09-06 06:04:00'),#13
           pys.time_float('2022-12-11 13:16:00'),#14
           pys.time_float('2023-03-17 20:30:00'),#15
           pys.time_float('2023-06-22 03:46:00'),#16
           pys.time_float('2023-09-27 23:28:00'),#17
           pys.time_float('2023-12-29 00:54:00'),#18
           pys.time_float('2024-03-30 02:20:00'),#19
           pys.time_float('2024-06-30 03:46:00'),#20
           pys.time_float('2024-09-30 05:13:00'),#21
           pys.time_float('2024-12-24 11:41:00'),#22
           pys.time_float('2025-03-22 22:25:00'),#23
           pys.time_float('2025-06-19 09:09:00')] #perihelion dates for nominal mission 
    
per_dist_lst = [35.6,35.6,35.6,27.8,27.8,20.3,20.3,15.9,15.9,13.3,13.3,13.3,13.3,13.3,13.3,13.3,
                11.4,11.4,11.4,11.4,11.4,9.9,9.9,9.9] #perihelion distances in units of Rs (solar radii)

Rs_grps = [[50,45],[45,40],[40,35],[35,30],[30,25],[25,20],[20,15],[15,10],[10,1]]