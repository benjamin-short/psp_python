#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 12:59:41 2021

@author: besh2109
"""

import numpy as np
import matplotlib.pyplot as plt
import pyspedas as pys
import pytplot as pyt
import os
import pandas as pd
from scipy.io import readsav
import datetime
import julian

# from .config import CONFIG
# from .config import enc_flt
# from .config import per_flt
# from .config import per_dist_lst
# from .config import Rs_grps


# variables to be defined within function #

enc_flt = [pys.time_float(['2018-08-23 05:51:00','2019-01-20 01:03:00']),
           pys.time_float(['2019-01-20 01:04:00','2019-06-18 20:15:00']),
           pys.time_float(['2019-06-18 20:15:00','2019-11-15 15:26:00']),
           pys.time_float(['2019-11-15 15:27:00','2020-04-03 09:00:00']),
           pys.time_float(['2020-04-03 09:01:00','2020-08-11 07:46:00']),
           pys.time_float(['2020-08-11 07:47:00','2020-12-01 08:39:00']),
           pys.time_float(['2020-12-01 08:40:00','2021-03-23 17:03:00']),
           pys.time_float(['2021-03-23 17:04:00','2021-05-23 17:03:00'])] #encounter list 1-8


encounter = 2
wavelen = 90
win_len = 1

# --------------------------------------- #


csv_filename = 'harmwave_master_arch.csv'
csv_path='/Users/besh2109/Desktop/PSP_epoch/wave_dates/'

tplot_filenames = ['LFRV1-V2Ne_Tc_Thk_mimo_2018-10-15_2018-11-20.sav',\
                   'LFRV1-V2Ne_Tc_Thk_mimo_2019-03-12_2019-04-18.sav',\
                   'LFRV1-V2Ne_Tc_Thk_mimo_2019-08-15_2019-09-12.sav']
tplot_filename = tplot_filenames[encounter-1]
tplot_path = '/Users/besh2109/Desktop/PSP_electrons/'

isfile = os.path.isfile(csv_path+csv_filename)

if isfile:
    
    df = pd.read_csv(csv_path+csv_filename)
    bf = df.to_numpy()

    af = np.delete(bf,bf[:,1]<wavelen,0)
    
    wave_dates = np.array(pys.time_float(af[:,0]))
    af = np.delete(af,wave_dates<enc_flt[encounter-1][0],0)
    wave_dates = np.array(pys.time_float(af[:,0]))
    af = np.delete(af,wave_dates>enc_flt[encounter-1][1],0)
    
    wave_start = np.array(pys.time_float(af[:,0]))
    wave_end = np.array(wave_start+af[:,1])
    
    window_start = np.array(wave_start - win_len*af[:,1]) #complete epoch analysis for window larger than wave itself
    window_end = np.array(wave_end + win_len*af[:,1]) #want to see if theres a difference between times during and before/after events
    
    dates = list(af[:,0])
    for j in range(len(dates)):
        dates[j] = dates[j][0:10]

    uniq_dates = np.unique(dates)
    date_flt = pys.time_float(uniq_dates)
    uniq_next = pys.time_string(np.array(date_flt)+86400.)
    
    duration = np.array(af[:,1])
    
    
    dens_time = []
    density_data = []
    dens_len_arr = []
    
    alp_time = []
    alpha_data = []
    alp_len_arr = []
    
    ion_med_arr = []
    ion_max_arr = []
    
    elec_time = []
    electron_data = []
    elec_len_arr = []
    elec_med_arr = []
    elec_max_arr = []
        
    for j in range(len(uniq_dates)): #gather data for each day range(2):#
                
        pys.psp.spi(trange=[uniq_dates[j],uniq_next[j]], datatype='spi_sf00', level='L3')
        
        dens_data = pyt.get_data('DENS')
        
        dens_time_arr = dens_data[0]
        dens_data_arr = dens_data[1]
        
        pys.psp.spi(trange=[uniq_dates[j],uniq_next[j]], datatype='spi_sf01', level='L3')
                
        alp_data = pyt.get_data('DENS')
                
        alp_time_arr = alp_data[0]
        alp_data_arr = 2*alp_data[1]
        
        elec_raw = readsav(tplot_path+tplot_filename)
        elec_time_arr = elec_raw.jdtot
        elec_data_arr = elec_raw.denstot
        elec_time_list = []
        elec_time_str = []
        for k in range(len(elec_time)):
            date =  julian.from_jd(elec_time_arr[k])
            date_str = date.strftime('%Y-%m-%d/%H:%M:%S')
            date_float = pys.time_float(date_str)
            elec_time_str.append(date_str)
            elec_time_list.append(date_float)
            
        elec_time_arr = np.array(elec_time_list)
        
        """ date management """
        
        date_strt_flt = pys.time_float(uniq_dates[j])
        date_end_flt = pys.time_float(uniq_next[j])
        
        where = np.where(np.logical_and(wave_start>date_strt_flt,wave_start<date_end_flt))
        
        win_start_tmp = np.array(window_start[where])
        win_end_tmp = np.array(window_end[where])
        wv_start = np.array(wave_start[where])
        wv_end = np.array(wave_end[where])
        wv_dur = np.array(duration[where])
        
        for k in range(len(win_start_tmp)): 
          
            """ electron data """
            elec_where = np.where((elec_time_arr > wv_start[k]) & (elec_time_arr < wv_end[k]))
            elec_where = elec_where[0]
            elec_ti = np.array(elec_time_arr[elec_where])
            elec_data_tmp = np.array(elec_data_arr[elec_where])
            elec_med = np.nanmedian(elec_data_arr[elec_where])
            if len(elec_where) == 0:
                elec_max = np.nan
            else:
                elec_max = np.nanmax(elec_data_arr[elec_where])
            
            elec_time.append(elec_ti)
            electron_data.append(elec_data_tmp)
            elec_len_arr.append(len(elec_ti))
            
            elec_med_arr.append(elec_med)
            elec_max_arr.append(elec_max)
            
            """ ion data """
            
            alp_where = np.where((alp_time_arr > wv_start[k]) & (alp_time_arr < wv_end[k]))
            alp_where = alp_where[0]
            
            dens_where = np.where((dens_time_arr > wv_start[k]) & (dens_time_arr < wv_end[k]))
            dens_where = dens_where[0]
            
            
            alp_ti = np.array(alp_time_arr[alp_where])
            alp_data_tmp = np.array(alp_data_arr[alp_where])
            alp_med = np.nanmedian(alp_data_arr[alp_where])
            
            if len(alp_where) == 0:
                alp_max = np.nan
            else:
                alp_max = np.nanmax(alp_data_arr[alp_where])
            
            alp_time.append(alp_ti)
            alpha_data.append(alp_data_tmp)
            alp_len_arr.append(len(alp_ti))
            
            dens_ti = np.array(dens_time_arr[dens_where])
            dens_data_tmp = np.array(dens_data_arr[dens_where])
            ion_med = np.nanmedian(dens_data_arr[dens_where])
            
            if len(dens_where) == 0:
                ion_max = np.nan
            else:
                ion_max = np.nanmax(dens_data_arr[dens_where])
            
            ion_med_arr.append(alp_med+ion_med)
            ion_max_arr.append(alp_max+ion_max)
            #print(len(drift_ti))
    
    elec_med_arr = np.array(elec_med_arr)
    elec_max_arr = np.array(elec_max_arr)
    
    ion_med_arr = np.array(ion_med_arr)
    ion_max_arr = np.array(ion_max_arr)

    
    print(ion_med_arr)
    print(elec_med_arr)

    print(ion_med_arr/elec_med_arr)
    print(ion_max_arr/elec_max_arr)