#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  1 15:24:04 2021

@author: besh2109
"""

import numpy as np
import pyspedas as pys
import os
import pandas as pd
import pytplot as pyt
import matplotlib.pyplot as plt

from .config import CONFIG
from .config import enc_flt
from .config import per_flt
from .config import per_dist_lst

def wave_2d_hist(arg=''):
    
    i = 0
    
    while i <= len(per_flt):

        if i == 0:
            csv_filename = 'harmwave_sorted_master.csv'
            csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
            savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/histograms/'
            savename = 'master_histogram.png'
        else:
            if arg=='sorted':
                csv_filename = 'enc_'+str(i)+'_harmwave_sorted.csv'
                csv_path = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/Enc'+str(i)+'/'
                savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/histograms/'
                savename = 'Enc_'+str(i)+'_f_fce_hist_sorted.png'
            else:
                csv_filename = 'enc_'+str(i)+'_harmwave_raw.csv'
                csv_path = '/Users/besh2109/Desktop/psp_islands/wave_data/'
                savepath = csv_path+'histograms/'
                savename = 'Enc_'+str(i)+'_f_fce_hist.png'
        
        isfile = os.path.isfile(csv_path+csv_filename)
        
        if isfile:
            
            df = pd.read_csv(csv_path+csv_filename)
            bf = df.to_numpy()

            f_fce = np.array(bf[:,3])
            
            if i == 0:
                rs_to_peri = np.abs(np.array(bf[:,9]))
            else:     
                rs_to_peri = np.array(bf[:,10])
            
            histo,xedge,yedge = np.histogram2d(rs_to_peri,f_fce,bins=[30,100],range=[[rs_to_peri.min(),rs_to_peri.max()],[0,2]])
            histo = np.transpose(histo)
            histo[histo==0]=np.nan
            
            fig = plt.figure(figsize=(15,15))
            axs1 = fig.add_subplot(111)

            axs1.pcolormesh(xedge,yedge,np.log10(histo), cmap='jet')

            if i ==0:
                axs1.set_xlabel('Distance in Rs',fontsize=16)
            else:
                axs1.set_xlabel('Distance from Perihelion in Rs',fontsize=16)
            
            axs1.set_ylabel('f/fce', fontsize=16)
    
            isdir = os.path.isdir(savepath)
                        
            if isdir == False:
                os.makedirs(savepath)
                        
            plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
            plt.clf()
            plt.cla()
            plt.close('all')
            plt.close(fig)
                
        i+=1
        

def py_archipelago():
    Rs = 6.957e5 #solar radius in km
    i = 0 #start at 0 to operate on the master csv as well as the original csvs
    enc_num = len(per_flt)-1
    while i <= enc_num:
        
        if i == 0:
            csv_filename = 'harmwave_sorted_master.csv'
            csv_path = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
            savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
            savename = 'harmwave_master_arch.csv'
        else:
            csv_filename = 'enc_'+str(i)+'_harmwave_sorted.csv'
            csv_path = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/Enc'+str(i)+'/'
            savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/Enc'+str(i)+'/'
            savename = 'enc_'+str(i)+'_harmwave_arch.csv'
            
        df = pd.read_csv(csv_path+csv_filename)
        bf = df.to_numpy()
        

        
        yes_check = []
        
        for j in range(len(bf[:,0])-1):
            
            wave_start = pys.time_float(bf[j,0])
            wave_end = wave_start + bf[j,1]
            
            next_start = pys.time_float(bf[j+1,0])
            next_end = next_start + bf[j+1,1]
            
            wave_dif = next_start - wave_end
            
            if wave_dif < 120:
                yes_check.append('yes') #could just use boolean objects but thats lame

            else:
                yes_check.append('no')
            
        yes_check.append('no') #to future me: dont delete this, its important to preserve the last wave
      
        wave_starts = []
        wave_ranges = []
        wave_dists = []
        
        j=0
        while j < len(yes_check):
            
            if yes_check[j]=='yes':
                k=0
                tmp = 0
                wave_start = bf[j,0]
                dist = []
                while yes_check[j+k]=='yes':
                    
                    wave_end = pys.time_float(bf[j+k+1,0])+bf[j+k+1,1]
                    wave_range = wave_end - pys.time_float(wave_start)
                    dist.append(bf[j+k+1,9])
                    k+=1
                    tmp+=1
                wave_starts.append(wave_start)
                wave_ranges.append(wave_range)
                wave_dists.append(np.max(dist))
                j+=tmp #skips ahead by the length of the waves
            else:
                wave_start = bf[j,0]
                wave_range = bf[j,1]
                wave_dist = bf[j,9]
                wave_starts.append(wave_start)
                wave_ranges.append(wave_range)
                wave_dists.append(wave_dist)
            
            j+=1
        
        
        dates_arr = np.array(wave_starts)
        range_arr = np.array(wave_ranges)
        
        dates = list(dates_arr)
        for j in range(len(dates)):
            dates[j] = dates[j][0:10]

        uniq_dates = np.unique(dates)
        date_flt = pys.time_float(uniq_dates)
        uniq_next = pys.time_string(np.array(date_flt)+86400.)
        
        wave_start = np.array(pys.time_float(dates_arr))
        wave_end = np.array(wave_start+range_arr)
        
        Rs_data = []
        for j in range(len(uniq_dates)):
            pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]], datatype='ephem_eclipj2000', level='l1') #going to be used to plot parker position
            pos_data_tmp = pyt.get_data('position')

            pos_time_tmp = pos_data_tmp[0]
            pos_data_tmp = pos_data_tmp[1]
            
            date_strt_flt = pys.time_float(uniq_dates[j])
            date_end_flt = pys.time_float(uniq_next[j])
        
            where = np.where(np.logical_and(wave_start>date_strt_flt,wave_start<date_end_flt))
        
            win_start_tmp = np.array(wave_start[where])
            win_end_tmp = np.array(wave_end[where])
            
            for k in range(len(win_start_tmp)): 
                wave_center = (win_end_tmp[k]+win_start_tmp[k])/2
                time_dif = np.array(abs(pos_time_tmp-wave_center))
                pos_where = np.where(time_dif == min(time_dif))
                pos_where = pos_where[0][0]

                r_km = np.sqrt(pos_data_tmp[pos_where,0]**2+pos_data_tmp[pos_where,1]**2+pos_data_tmp[pos_where,2]**2)
                Rs_data.append(np.array(r_km)/Rs)
        

        dist_arr = np.array(wave_dists)
        csv_arr = np.array([dates_arr,range_arr,Rs_data])
        i+=1
        
        csv_arr = np.transpose(csv_arr)
        #print(csv_arr.shape)
        csv_isdir = os.path.isdir(savepath)
                    
        if csv_isdir == False:
            os.makedirs(savepath)
                            
        csv_isfile = os.path.isfile(savepath+savename)
                            
        if csv_isfile == False:
            header = np.array(['wave date','length of wave','dist. to Sun'])
            np.savetxt(savepath+savename,np.array([]),
                       header='wave date,length of wave, dist to Sun',
                       fmt='%s',delimiter=',')
                            
        with open(savepath+savename,"a") as file:
                                
            np.savetxt(file,csv_arr,delimiter=',',fmt='%s')
