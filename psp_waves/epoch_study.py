#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:50:30 2021

@author: besh2109
"""

import numpy as np
import pyspedas as pys
import pytplot as pyt
import os
import pandas as pd

import matplotlib.pyplot as plt

from .config import CONFIG
from .config import enc_flt
from .config import per_flt
from .config import per_dist_lst

def mag_epoch():

    i=0
    while i <= len(per_flt):
        
        if i==0:
            csv_filename = 'harmwave_master_arch.csv'
            csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
            savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/histograms/'
            savename = 'master_mag_epoch.png'
        else:
            csv_filename = 'enc_'+str(i)+'_harmwave_arch.csv'
            csv_path = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/Enc'+str(i)+'/'
            savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/histograms/'
            savename = 'Enc_'+str(i)+'_mag_epoch.png'
        
        isfile = os.path.isfile(csv_path+csv_filename)
        
        if isfile:
            
            df = pd.read_csv(csv_path+csv_filename)
            bf = df.to_numpy()

            af = np.delete(bf,bf[:,1]<90,0)
            
            wave_start = np.array(pys.time_float(af[:,0]))
            wave_end = np.array(wave_start+af[:,1])
            
            window_start = np.array(wave_start - af[:,1]*(1/3)) #complete epoch analysis for window larger than wave itself
            window_end = np.array(wave_end + af[:,1]*(1/3))     #want to see if theres a difference between times during and before/after events
            
            dates = list(af[:,0])
            for j in range(len(dates)):
                dates[j] = dates[j][0:10]

            uniq_dates = np.unique(dates)
            date_flt = pys.time_float(uniq_dates)
            uniq_next = pys.time_string(np.array(date_flt)+86400.)
            
            mag_time = []
            mag_r_data = []
            mag_t_data = []
            mag_n_data = []
            len_arr = []
            for j in range(2):#len(uniq_dates)): #gather data for each day
                pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]], datatype='mag_RTN', level='l2')
                mag_data = pyt.get_data('psp_fld_l2_mag_RTN')
                
                mag_time_arr = mag_data[0]
                mag_data_arr = mag_data[1]
                
                date_strt_flt = pys.time_float(uniq_dates[j])
                date_end_flt = pys.time_float(uniq_next[j])
                
                where = np.where(np.logical_and(wave_start>date_strt_flt,wave_start<date_end_flt))
                
                win_start_tmp = np.array(window_start[where])
                win_end_tmp = np.array(window_end[where])
                
                for k in range(len(win_start_tmp)): 
                    
                    mag_where = np.where((mag_time_arr > win_start_tmp[k]) & (mag_time_arr < win_end_tmp[k]))
                    mag_where = mag_where[0]
                    
                    mag_ti = np.array(mag_time_arr[mag_where])
                    mag_r = np.array(mag_data_arr[mag_where,0])
                    mag_t = np.array(mag_data_arr[mag_where,1])
                    mag_n = np.array(mag_data_arr[mag_where,2])
    
                    mag_time.append(mag_ti)
                    mag_r_data.append(mag_r)
                    mag_t_data.append(mag_t)
                    mag_n_data.append(mag_n)
                    len_arr.append(len(mag_ti))
             
            min_len = min(len_arr) #minimum length of mag data
            n_bins = int(min_len)
            print(n_bins)
            for l in range(len(mag_time)): #this block seeks to normalize all the magnetic field data 
                                           #to the length of the shortest window
                bin_size = len_arr[l]/n_bins
                ind_arr = np.arange(len_arr[l])
                norm_mag_time = np.zeros((n_bins,))
                norm_mag_r_val = np.zeros((n_bins,))
                norm_mag_t_val = np.zeros((n_bins,))
                norm_mag_n_val = np.zeros((n_bins,))
                for m in range(n_bins):
                    win_str = m*bin_size
                    win_end = (m+1)*bin_size
                    wind_where = np.where((ind_arr>=win_str)&(ind_arr<win_end))
                    wind_where = wind_where[0]
                    mag_time_val_tmp = np.median(mag_time[l][wind_where])
                    mag_r_val_tmp = np.median(mag_r_data[l][wind_where])
                    mag_t_val_tmp = np.median(mag_t_data[l][wind_where])
                    mag_n_val_tmp = np.median(mag_n_data[l][wind_where])
                    norm_mag_time[m] = mag_time_val_tmp
                    norm_mag_r_val[m] = mag_r_val_tmp
                    norm_mag_t_val[m] = mag_t_val_tmp
                    norm_mag_n_val[m] = mag_n_val_tmp
                
                mag_time[l] = np.array(norm_mag_time) 
                mag_r_data[l] =  np.array(norm_mag_r_val)-np.median(norm_mag_r_val)
                mag_t_data[l] =  np.array(norm_mag_t_val)-np.median(norm_mag_t_val)
                mag_n_data[l] =  np.array(norm_mag_n_val)-np.median(norm_mag_n_val)

            fis1 = plt.figure(figsize=(15,10))
            axs1 = fis1.add_subplot()
            for n in mag_r_data:
                axs1.plot(n)    
            height = [0,0]
            endpoints = [n_bins/5+30,4*n_bins/5-30]
            axs1.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_r_data)+1)
            plt.title('mag_r first 73 waves, normalized time')
            print('test')
            plt.show()
            
            fis2 = plt.figure(figsize=(15,10))
            axs2 = fis2.add_subplot()
            for n in mag_t_data:
                axs2.plot(n)    
            height = [0,0]
            endpoints = [n_bins/5+30,4*n_bins/5-30]
            axs2.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_t_data)+1)
            plt.title('mag_t first 73 waves, normalized time')
            print('test')
            plt.show()
            
            fis3 = plt.figure(figsize=(15,10))
            axs3 = fis3.add_subplot()
            for n in mag_n_data:
                axs3.plot(n)    
            height = [0,0]
            endpoints = [n_bins/5+30,4*n_bins/5-30]
            axs3.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_n_data)+1)
            plt.title('mag_n first 73 waves, normalized time')
            print('test')
            plt.show()
                
        i+=1