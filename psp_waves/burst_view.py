#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 12 09:17:11 2021

@author: benshort
"""

import os
import pyspedas.psp as psp
from pyspedas import time_string
from pyspedas import time_float
import pytplot as pyt
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from statistics import median


def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period. To kill .DS_store files."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def unique(list1):
 
    # intilize a null list
    unique_list = []
     
    # traverse for all elements
    for x in list1:
        # check if exists in unique_list or not
        if x not in unique_list:
            unique_list.append(x)
    # return list
    return unique_list

def burst_view(folder='vlf_whistler'):
    
    fold_path = '/Users/benshort/Desktop/interesting_waves/sorted/'+folder+'/' #folder path
    
    savepath = '/Users/benshort/Desktop/bursts/'
    
    file_list = mylistdir(fold_path)
    file_list.sort()
    
    day_list_tmp = []
    date_list = []
    for x in file_list:
        y0 = x[4:8]
        m0 = x[8:10]
        d0 = x[10:12]
        
        H0 = x[12:14]
        M0 = x[14:16]
        
        day_list_tmp.append(y0+'-'+m0+'-'+d0)
        date_list.append(y0+'-'+m0+'-'+d0+'/'+H0+':'+M0+':00')
        
    
    day_list = unique(day_list_tmp) # days represented inside the folder in question.
    date_float = np.array(time_float(date_list))
    #date_list = np.array(date_list)    
    
    
    for x in day_list:
        
        t0 = x
        tf = time_string(time_float(t0)+86400.) # one day past t0
        
        t0_flt = time_float(t0)
        tf_flt = time_float(t0)+86400.
        
        burst = psp.fields(trange=[t0,tf], datatype='dfb_dbm_dvac', level='l2')
        
        V12_data = pyt.get_data('psp_fld_l2_dfb_dbm_dvac12')
        V34_data = pyt.get_data('psp_fld_l2_dfb_dbm_dvac34')
        V5_data = pyt.get_data('psp_fld_l2_dfb_dbm_dvacz')
        
        V12_time_arr = V12_data[0]
        V12_data_arr = V12_data[1]
        V12_sec_arr = V12_data[2]
        
        V34_time_arr = V34_data[0]
        V34_data_arr = V34_data[1]
        V34_sec_arr = V34_data[2]
        
        V5_time_arr = V5_data[0]
        V5_data_arr = V5_data[1]
        V5_sec_arr = V5_data[2]
        
        rang = np.arange(len(V12_time_arr))
        
        png_where = np.where((date_float>=t0_flt)&(date_float<tf_flt))
        png_where = png_where[0] #finds how many images I have for this day
        
        windowstart = date_float[png_where]
        windowend = (date_float + 1200.)[png_where] #designates the time ranges for those images

        rang = np.arange(len(windowstart))
        
        for y in rang:
            
            ti_flt = windowstart[y]
            tf_flt = windowend[y]
            burst_where = np.where((V12_time_arr>ti_flt)&(V12_time_arr<tf_flt))
            burst_where = burst_where[0]
            
            if len(burst_where) != 0:
                
                b_rang = np.arange(len(burst_where)) 
                
                for z in burst_where:
                    
                    savename = time_string(V12_time_arr[z],fmt='psp_%Y%m%d%H%M%S_burst.png')
                    
                    res = 1./(V12_sec_arr[z,1]-V12_sec_arr[z,0])
                    
                    fig, axs = plt.subplots(3, 1, figsize=(20,10))
                    
                    axs[0].set(title=time_string(V12_time_arr[z]))
                    
                    f12, t12, v12 = signal.spectrogram(V12_data_arr[z,:], res, window='hamming', nperseg=15000) 

                    for f in range(len(v12[:,0])):
                        noise = median(v12[f,:])
                        v12[f,:] = 10*np.log10(v12[f,:]/noise)                    
                    
                    v12_c = axs[0].pcolormesh(t12,f12,v12,cmap="nipy_spectral",shading='nearest')
                    
                    box = axs[0].get_position()
                    axColor = plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])          
                    fig.colorbar(v12_c,cax=axColor, label='dB')                    
                    
                    f34, t34, v34 = signal.spectrogram(V34_data_arr[z,:], res, window='hamming', nperseg=15000)         
                    
                    for f in range(len(v34[:,0])):
                        noise = median(v34[f,:])
                        v34[f,:] = 10*np.log10(v34[f,:]/noise)                      
                    
                    axs[0].set_yscale('log')
                    axs[0].set_ylim(20,20000)
                    
                    v34_c = axs[1].pcolormesh(t34,f34,v34,cmap="nipy_spectral",shading='nearest')
                    
                    
                    box = axs[1].get_position()
                    axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])          
                    fig.colorbar(v34_c,cax=axColor, label='dB')     

                    f5, t5, v5 = signal.spectrogram(V5_data_arr[z,:], res, window='hamming', nperseg=15000)       
                   
                    for f in range(len(v5[:,0])):
                        noise = median(v5[f,:])
                        v5[f,:] = 10*np.log10(v5[f,:]/noise)                        
                    axs[1].set_yscale('log')
                    axs[1].set_ylim(20,20000)
                    
                    v5_c = axs[2].pcolormesh(t5,f5,v5,cmap="nipy_spectral",shading='nearest')                    
                    
                    box = axs[2].get_position()
                    axColor = plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])          
                    fig.colorbar(v5_c,cax=axColor, label='dB')                         
                    axs[2].set_yscale('log')
                    axs[2].set_ylim(20,20000)
                    
                    plt.savefig(savepath+savename)
                    plt.clf()
                    plt.cla()
                    plt.close('all')
                    plt.close(fig)              
                    #plt.show()
                    
            else:
                print('No bursts in this time range.')
        
        #i_min = 0
        #while i_min < 86400.:
            
            #ti_doub = t0_flt + i_min
            #tf_doub = ti_doub + 1200. #20 minute window. 1200 seconds in 20 minutes
            
            #burst_where = np.where((V12_time_arr>ti_doub)&(V12_time_arr<tf_doub))
            #burst_where = burst_where[0]
            
            #print(burst_where)
            
            
            #i_min+=1200.
        
        
            #for y in rang:
            
            
            
             #   res = 1./(V12_sec_arr[y,1]-V12_sec_arr[y,0])
             #   plt.specgram(V12_data_arr[y,:],Fs=res,cmap="nipy_spectral")
             #   plt.title(time_string(V12_time_arr[y]))
             #   plt.show()
        
        #save_tmp = fold_path+'sorted/'