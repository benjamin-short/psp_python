#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:59:49 2021

@author: benshort
"""

from __future__ import print_function  # for Python2
import sys

import time
import gc

import pyspedas.psp as psp
import pyspedas as pys
import pytplot as pyt
import numpy as np
from datetime import date
import os
from statistics import median
from statistics import mean

from scipy.interpolate import interp1d
from scipy.signal import savgol_filter
from scipy.fft import ifft,fft,fftfreq
import pandas as pd

from .py_islands import islands

import matplotlib.pyplot as plt #should only be in here temporarily, to view work
from matplotlib import ticker

from .config import CONFIG

from .config import enc_flt
from .config import per_flt
from .config import per_dist_lst

Rs = 6.957e5 #solar radius in km

def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def harm_wave_id(fd='2018-10-03', mode='csv'): #can choose any date for fd, can choose 'csv', 'picture', or 'both' for mode

    #if mode == 'pictures': mode = 'picture' # i obviously knew nothing about Python when I wrote this line
    if mode not in ['csv','picture','all','none']: mode == 'all'
    
    first_day = fd
    current_day = date.today().strftime("%Y-%m-%d")
    
    i_day = pys.time_float(first_day)
    
    loop_day = pys.time_float(current_day) - 86400.
    
    
    while i_day < loop_day: #pys.time_float(first_day) + 3*86400.:
        
        t0 = pys.time_string(i_day)
        tf = pys.time_string(i_day + 86400.)
        
        
        """
        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),key= lambda x: -x[1])[:10]:
                                 
            print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        """
        
        enci = 0
        for enc in enc_flt:          
            if (i_day>enc[0]) and (i_day<enc[1]): 
                encounter = enci+1 #determine which encounter today is in
                per_date = per_flt[enci]
                per_dist = per_dist_lst[enci]
            enci+=1
        
        #mag_file_path = CONFIG['local_data_dir']+'/fields/l2/mag_RTN/'+t0[0:4]+'/'+t0[5:7]+'/'
        mag_file_path = CONFIG['local_data_dir']+'/fields/l2/mag_RTN_4_Sa_per_Cyc/'+t0[0:4]+'/'+t0[5:7]+'/'
        spec_file_path = CONFIG['local_data_dir']+'/fields/l2/dfb_ac_spec/'+t0[0:4]+'/'+t0[5:7]+'/'
        ephem_file_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/'+t0[0:4]+'/'+t0[5:7]+'/'
        
        ephem_file_name = pys.time_string(i_day,fmt='spp_fld_l1_ephem_eclipj2000_%Y%m%d_v01.cdf')
        
        isdir_mag = os.path.isdir(mag_file_path)
        isdir_spec = os.path.isdir(spec_file_path)
        isdir_eph = os.path.isdir(ephem_file_path)
        
        if isdir_mag == False:
            os.makedirs(mag_file_path)
        if isdir_spec == False:
            os.makedirs(spec_file_path)
        if isdir_eph == False:
            os.makedirs(ephem_file_path)
        
        """ 
        The following could simply be done with psp.fields(arguments), but certain servers pay attention to
        what IPs are hogging data. Particularly the public NASA servers. If I pull too much information for too long
        (like looping through all the Parker Solar Probe data)
        then the servers will seriously slow down how much I can pull after a while. 
        
        Here I'm pulling straight from the Berkeley Servers so that shouldnt
        be an issue but its good practice to check whether you have the data already before unnecessarily making a
        a request of the server.
        """
        

        mag_isfile = []
        mag_infile = []
        mag_res = 6*3600 #seconds: that is, 6 hours. The resolution of the magnetic field files.
        
        spec_tags = ['dV12hg','dV34hg','V5','null'] # 'null' doesnt exist,
                                                    # but including a 4th tag here 
                                                    # allows me to consolidate two 'for' loops into one 'for' loop below.
        spec_isfile = []
        spec_infile = []
        
        for i in [0,1,2,3]:
            #mag_file_name = pys.time_string(i_day+i*mag_res,fmt='psp_fld_l2_mag_RTN_%Y%m%d%H_v01.cdf')
            mag_file_name = pys.time_string(i_day,fmt='psp_fld_l2_mag_RTN_4_Sa_per_Cyc_%Y%m%d_v01.cdf')
            spec_file_name = pys.time_string(i_day,fmt='psp_fld_l2_dfb_ac_spec_'+spec_tags[i]+'_%Y%m%d_v01.cdf')
            
            isfile_mag = os.path.isfile(mag_file_path+mag_file_name)
            isfile_spec = os.path.isfile(spec_file_path+spec_file_name)
            
            mag_isfile.append(isfile_mag)
            mag_infile.append(mag_file_path+mag_file_name)
            
            spec_isfile.append(isfile_spec)
            spec_infile.append(spec_file_path+spec_file_name)
            #print(mag_file_name)
        
        #mag_isfile = mag_isfile[0]
        #mag_infile = mag_infile[0]
        
        spec_isfile = spec_isfile[0:3] #getting rid of the 'null' tag in our list
        spec_infile = spec_infile[0:3]
        
        spec_tplot_names = ['psp_fld_l2_dfb_ac_spec_dV12hg','psp_fld_l2_dfb_ac_spec_dV34hg','psp_fld_l2_dfb_ac_spec_V5hg']
        
        spec_where = [i for i, x in enumerate(spec_isfile) if x]
        if len(spec_where) == 0:
            psp.fields(trange=[t0,tf], datatype='dfb_ac_spec',level='l2')
            
            for i in [0,1,2]: #loop through all tplot names to find the first present data set, if it exists
                tmp = pyt.get_data(spec_tplot_names[i])
                if tmp != None:
                    acspec_data = tmp
                    break #stop for loop at first present data
                else:
                    acspec_data = None
                        
        else:
            spec_ind = spec_where[0]                # chooses the first present spec file, if it exists.
            print(t0[0:10]+" spec file exists, importing data...")
            pyt.cdf_to_tplot(spec_infile[spec_ind])
            spec_name = spec_tplot_names[spec_ind]
            
            acspec_data = pyt.get_data(spec_name)
        
        if acspec_data != None:
            acspec_time_tmp = acspec_data[0]
        
            res_check = []
            for i in range(len(acspec_time_tmp)-1):
                res_check.append(acspec_time_tmp[i+1]-acspec_time_tmp[i])
            
            res_mean = round(mean(res_check),3) #check the mean resolution length, 
                                                #we dont care about times where AC_spec is at 30 second cadence           
        else:
            res_mean = 10 #insert arbitrary number greater than 2.
            
        for i in res_check:
            if i > 2:
                res_mean = 10
        
        #print(res_check)
            
        if res_mean < 2: #if the mean resolution is greater than 2 seconds, dont bother.
            
            
            if False in mag_isfile:
                #psp.fields(trange=[t0,tf], datatype='mag_RTN', level='l2')
                psp.fields(trange=[t0,tf], datatype='mag_RTN_4_Sa_per_Cyc', level='l2')
            else:
                print(t0[0:10]+" mag file exists, importing data...")
                #pyt.cdf_to_tplot(mag_infile)
                pyt.cdf_to_tplot(mag_infile[0])
             
            #mag_data = pyt.get_data('psp_fld_l2_mag_RTN')   #retrieve mag data from tplot variable
            mag_data = pyt.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
            
            ephem_isfile = os.path.isfile(ephem_file_path+ephem_file_name)
            
            if ephem_isfile:
                pyt.cdf_to_tplot(ephem_file_path+ephem_file_name)
            else:
                psp.fields(trange=[t0,tf], datatype='ephem_eclipj2000', level='l1') 
            
            pos_data = pyt.get_data('position') #retrieve PSP position data from tplot variable
            
            data_check = [acspec_data,pos_data] #mag_data,
            
            if None not in data_check: #that is to say, if all data is present. Begin identifying waves.
                
                 #Ive ignored the mag data for now
                mag_time_arr = mag_data[0]
                mag_data_arr = mag_data[1]
                
                acspec_time_arr = np.array(acspec_data[0])
                ac_tmp = np.array(acspec_data[1])
                ac_tmp[ac_tmp==0] = np.nan #Replace all zeroes in data with NaNs so that Log(0)=-Inf doesnt show up.
                acspec_data_arr = np.array(ac_tmp)
                #a = 10*acspec_data_arr#np.log10(np.transpose(acspec_data_arr))
                
                #plt.pcolormesh(a[0:100,:],cmap='nipy_spectral',shading='nearest')
                #plt.show()
                
                for h in range(len(ac_tmp[0,:])):
                    noise = median(ac_tmp[:,h])
                    acspec_data_arr[:,h] = 10*np.log10(ac_tmp[:,h]/noise) #convert to dB on a frequency by frequency basis.
                
                acspec_data_arr = np.transpose(acspec_data_arr)
                acspec_freq_arr = np.transpose(acspec_data[2])
                acspec_freq_lst = np.array(acspec_freq_arr[:,0])
                
                #plt.pcolormesh(acspec_data_arr[0:100,:],cmap='nipy_spectral',shading='nearest')
                #plt.show()
                #print(acspec_freq_lst)
                
                pos_time_arr = pos_data[0]
                pos_data_arr = pos_data[1]
                
                i_min = 0
                
                Bmag_day = np.sqrt(mag_data_arr[:,0]**2+mag_data_arr[:,1]**2+mag_data_arr[:,2]**2) #nT; magnitude of B
                Bmag_day = Bmag_day*10e-10 #T
                q = 1.60218*10e-20 #Coulombs
                me =  9.10938*10e-32 #kg
                mp =  1.67262*10e-28 #kg

                fce_day = q*Bmag_day/(2*np.pi*me) #Hz
                fce_day_mean = np.nanmean(fce_day)
                
                #print(fce_day_mean)
                
                Rs = 6.957e5 #solar radius in km 
                
                while i_min <= 82800.0: #82800 is 86400 seconds minus 60 minutes.
                    
                    ti_doub = pys.time_double(t0) + i_min
                    tf_doub = ti_doub + 3600. #60 minute window. 3600 seconds in 60 minutes
                    
                    ac_where = np.where((acspec_time_arr > ti_doub) & (acspec_time_arr < tf_doub))    
                    ac_where = ac_where[0]
                    
                    mag_where = np.where((mag_time_arr > ti_doub)&(mag_time_arr < tf_doub))
                    mag_where = mag_where[0]
                    
                    pos_where = np.where((pos_time_arr > ti_doub)&(pos_time_arr < tf_doub))
                    pos_where = pos_where[0]
                    
                                        
                    fr_where = np.where((acspec_freq_lst > 0.3*fce_day_mean))
                    fr_where = fr_where[0] 

                    ac_time_tmp = acspec_time_arr[ac_where]
                    
                    ac_data_tmp = acspec_data_arr[fr_where,:] #I also dont know why python forces me to do this in two steps
                    ac_data_tmp = ac_data_tmp[:,ac_where]     #rather than ac_data_tmp = acspec_data_arr[fr_where,ac_where]
                    
                    #ac_data_tmp = acspec_data_arr[:,ac_where]
                    
                    ac_freq_tmp = acspec_freq_arr[fr_where,:]
                    ac_freq_tmp = ac_freq_tmp[:,ac_where]
                    
                    #ac_freq_tmp = acspec_freq_arr[:,ac_where]
                    
                    ac_freq_lst = acspec_freq_arr[fr_where,0]#[fr_where,0] #[:,0]
                    
                    #ac_freq_lst = acspec_freq_arr[:,0]
                    
                    mag_time_tmp = mag_time_arr[mag_where]
                    mag_data_tmp = mag_data_arr[mag_where,:]
                    
                    pos_time_tmp = pos_time_arr[pos_where]
                    pos_data_tmp = pos_data_arr[pos_where,:]
                    
                    #print(len(mag_where))
                    #print(len(fr_where))
                    
                    if len(ac_where)>10: #if len(ac_where) > 1372:
                        res_check2 = []
                        for i in range(len(ac_time_tmp)-1):
                            res_check2.append(ac_time_tmp[i+1]-ac_time_tmp[i])
                
                        res_max = round(max(res_check2),3)
                        if res_max < 1:
                            window_skip = False
                        else:
                            window_skip = True
                        if len(mag_where)<4000:
                            window_skip = True
                        
                    else:
                        window_skip = True
                        
                    #print(len(ac_where))
                    if not window_skip: #skip windows where there is window length greater than 1 second
                        #print('test')
                        Bmag = np.sqrt(mag_data_tmp[:,0]**2+mag_data_tmp[:,1]**2+mag_data_tmp[:,2]**2) #nT; magnitude of B
                        Bmag = Bmag*10e-10 #T
                        
                        fce = q*Bmag/(2*np.pi*me) #Hz
                        
                        vmin0 = 4
                        isl_tmp = islands(ac_data_tmp, vmin0) #three outputs in a list, isl_tmp[0] is island array
                                                          #isl_tmp[1] is the same array but with only the largest island
                                                          #isl_tmp[2] is an array with the island numbers with their 
                                                          #corresponding area
                        
                        isl_arr0 = isl_tmp[0] #array containing original data sorted by connected data.
                        isl_arr0 = isl_arr0.astype(float)
                        isl_arr0[isl_arr0 == 0] = np.nan
                        
                        larg_isl = isl_tmp[1] #an array containing only the largest island
                        larg_isl = larg_isl.astype(float)
                        larg_isl[larg_isl == 0] = np.nan
                        
                        info_arr = np.transpose(isl_tmp[2]) #array containing information about the islands in isl_arr
                        
                        info_arr = info_arr[np.argsort(info_arr[:,7])] #sort by info by longest island in time in ascending order
                        #info_arr = np.flip(info_arr[np.argsort(info_arr[:,7])],axis=0) #sort by info by longest island in time in decending order
 
                        time_res = abs(ac_time_tmp[1]-ac_time_tmp[0]) #time resolution
                        freq_res = abs(ac_freq_lst[1]-ac_freq_lst[0]) #frequency resolution
                    
                        freq_start = round(ac_freq_lst[0],2)
                        time_start = ac_time_tmp[0]

                        freq_info = np.round(info_arr[:,2:5].astype(float)*freq_res,2) #convert array y-axis information into frequency information
                        time_info = np.round(info_arr[:,5:8].astype(float)*time_res,2) #convert array x-axis information into time information
                        
                        freq_info[:,0] = np.round(freq_info[:,0]+freq_start,2)
                        freq_info[:,1] = np.round(freq_info[:,1]+freq_start,2)
                        
                        
                        time_info[:,0] = np.round(time_info[:,0]+time_start,2)
                        time_info[:,1] = np.round(time_info[:,1]+time_start,2)+time_info[:,2]
                        
                        #print(info_arr[0:10,:])
                        mod_isl_arr = np.array(isl_arr0) #create copy of isl_arr0 to modify
                        
                        loopend = len(info_arr[:,0])
                        forget = []
                        
                        for i in range(loopend): #calculate the fce local to the wave event in question
                            
                            wv_chk_start = time_info[i,0]
                            wv_chk_end = time_info[i,1]
                            #print(wv_chk_start,wv_chk_end)
                            for j in range(i+1,loopend):
                                
                                wv_cross_start = time_info[j,0]
                                wv_cross_end = time_info[j,1]
                                #print(wv_cross_start,wv_cross_end)
                                #break
                                if wv_cross_start <= wv_chk_start and wv_chk_end <= wv_cross_end:
                                    forget_wave_name = float(info_arr[i,0])
                                    merge_wave_name = float(info_arr[j,0])
                                    
                                    forget.append(forget_wave_name)
                                    
                                    mod_isl_arr[mod_isl_arr==forget_wave_name] = merge_wave_name #set small wave name to larger wave name
                        
                        
                        for i in forget:
                            time_info = np.delete(time_info,info_arr[:,0]==i,0)
                            freq_info = np.delete(freq_info,info_arr[:,0]==i,0)
                            info_arr = np.delete(info_arr,info_arr[:,0]==i,0)
                            
                        isl_arr0 = np.array(mod_isl_arr) #set isl_arr0 back to the modified island array once operations are complete
                        
                        fce_list = []
                        rs_list = []
                        
                        #print(pos_data_tmp[:,0])
                        for i in range(len(info_arr[:,0])): #calculate the fce local to the wave event in question
                            fce_time_dif = np.abs(mag_time_tmp-time_info[i,0])
                            fce_td_where = np.where(fce_time_dif == fce_time_dif.min())
                            fce_td_where = fce_td_where[0][0]
                            #print(fce_td_where)
                            #fce_tmp = fce[fce_where]
                            fce_list.append(fce[fce_td_where])
                            
                            pos_time_dif = np.abs(pos_time_tmp-time_info[i,0])
                            pos_td_where = np.where(pos_time_dif == pos_time_dif.min())
                            pos_td_where = pos_td_where[0][0]
                            
                            rs_list.append(np.sqrt(pos_data_tmp[pos_td_where,0]**2+pos_data_tmp[pos_td_where,1]**2+pos_data_tmp[pos_td_where,2]**2)/Rs)
                        
                        fce_arr = np.array(fce_list)
                        rs_arr = np.array(rs_list)
                        rs_peri_arr = np.array(rs_arr-per_dist)
                        
                        for i in range(len(rs_arr)):
                            if time_info[i,0] < per_date:
                                rs_peri_arr[i] = -rs_peri_arr[i]
                                rs_arr[i] = -rs_arr[i] #set times before perihelion to negative distances
                        
                        tmin = 7 #ignore events less than tmin seconds long.
                        info_arr = np.delete(info_arr,time_info[:,2]<tmin,0) 
                        freq_info = np.delete(freq_info,time_info[:,2]<tmin,0)
                        fce_arr = np.delete(fce_arr,time_info[:,2]<tmin)
                        rs_arr = np.delete(rs_arr,time_info[:,2]<tmin)
                        rs_peri_arr = np.delete(rs_peri_arr,time_info[:,2]<tmin)
                        time_info = np.delete(time_info,time_info[:,2]<tmin,0)

                        
                        isl_num_arr = np.array(info_arr[:,0]) #island numbers in a list
                        isl_area_arr = np.array(info_arr[:,1])
                        #print(isl_list)
                        
                        uniq1 = np.unique(isl_tmp[0])
                        uniq1 = np.delete(uniq1,uniq1==0)
                        
                        #print(uniq)
                        ignore1 = np.array(uniq1)
                        for i in uniq1: #create a list of all islands to remove from isl_arr
                            if i in isl_num_arr:
                                ignore1 = np.delete(ignore1,ignore1==i)
                                
                        #print(ignore)
                        db_isl_arr = np.array(ac_data_tmp)
                        db_isl_arr = np.where(db_isl_arr <= vmin0,np.nan,db_isl_arr)
                        
                        isl_arr1 = np.array(isl_arr0) #did you know that if you want to clone a variable V, L = V doesnt 
                                                      #create a clone? You have to do L = np.array(V) or L = list(V)
                                                      # L = V only links L and V as one object, so anything you do to one
                                                      # you do to the other, kind of weird if you ask me...
                        
                        for i in ignore1:
                            
                            db_isl_arr[isl_arr1 == i] = np.nan
                            isl_arr1[isl_arr1 == i] = np.nan
                        
                        
                        """
                        First filter complete, cut out all events shorter than 5 seconds.
                        
                        The next bit is filled with literally any quantity I can think of that might help me distinguish
                        what I'm looking for.
                        """
                        
                        median_freq_list = []
                        mean_freq_list = []
                        max_freq_list = []
                        med_f_fce_list = []
                        mean_f_fce_list = []
                        max_f_fce_list = []
                        wght_mean_freq_list = []
                        std_dev_list = []
                        median_pow_list = []
                        mean_pow_list = []
                        max_pow_list = []

                        for i in isl_num_arr:
                            
                            arr_tmp = np.array(db_isl_arr)
                            arr_tmp[isl_arr1 != i] = np.nan
                            
                            n = np.where(isl_num_arr == i)
                            n = n[0][0]
                            
                            median_freq = np.median(ac_freq_tmp[isl_arr1 == i])
                            mean_freq = np.mean(ac_freq_tmp[isl_arr1 == i])
                            
                            
                            med_f_fce = median_freq/fce_arr[n]
                            mean_f_fce = mean_freq/fce_arr[n]
                            
                            #perform a weighted average, weighted by dB strength of signal
                            wght_freq_sum = np.sum(ac_data_tmp[isl_arr1 == i]*ac_freq_tmp[isl_arr1 == i])
                            wght_mean_freq = wght_freq_sum/np.sum(ac_data_tmp[isl_arr1 == i])
                                                        
                            #standard deviation of signal frequency, just another measure by which to tell shape.
                            std_dev = np.sqrt(np.sum((ac_freq_tmp[isl_arr1 == i]-wght_mean_freq)**2)/isl_area_arr[n])
                            
                            median_pow = np.median(ac_data_tmp[isl_arr1 == i])
                            mean_pow = np.mean(ac_data_tmp[isl_arr1 == i])
                            max_pow = np.max(ac_data_tmp[isl_arr1 == i])
                            
                            #print(np.nanmax(arr_tmp))
                            max_pow_where = np.where(arr_tmp==np.nanmax(arr_tmp))
                            max_pow_freq = ac_freq_tmp[max_pow_where]
                            max_pow_freq = max_pow_freq[0]
                            #print(max_pow_freq[0]/fce_arr[n])

                            median_freq_list.append(median_freq)
                            mean_freq_list.append(mean_freq)
                            max_freq_list.append(max_pow_freq)
                            med_f_fce_list.append(med_f_fce)
                            mean_f_fce_list.append(mean_f_fce)
                            max_f_fce_list.append(max_pow_freq/fce_arr[n])
                            wght_mean_freq_list.append(wght_mean_freq)                            
                            std_dev_list.append(std_dev)
                            median_pow_list.append(median_pow)
                            mean_pow_list.append(mean_pow)
                            max_pow_list.append(max_pow)
                         
                        median_freq_arr = np.array(median_freq_list)
                        mean_freq_arr = np.array(mean_freq_list)
                        max_freq_arr = np.array(max_freq_list)
                        med_f_fce_arr = np.array(med_f_fce_list)
                        mean_f_fce_arr = np.array(mean_f_fce_list)
                        max_f_fce_arr = np.array(max_f_fce_list)
                        wght_mean_freq_arr = np.array(wght_mean_freq_list)
                        std_dev_arr = np.array(std_dev_list)
                        median_pow_arr = np.array(median_pow_list)
                        mean_pow_arr = np.array(mean_pow_list)
                        max_pow_arr = np.array(max_pow_list)
                        
                        """
                        print('')
                        print('Isl, median, mean, weighted mean, std dev, median pow, mean pow, max pow')
                        for i in range(len(isl_num_arr)):
                            print(isl_num_arr[i],median_freq_arr[i],mean_freq_arr[i],wght_mean_freq_arr[i],
                                  std_dev_arr[i],median_pow_arr[i],mean_pow_arr[i],max_pow_arr[i])
                        """

                        f_min = 0.5*fce_day_mean
                        
                        info_arr = np.delete(info_arr,median_freq_arr<f_min,0) #ignore events with mean less than 0.5 fce.
                        freq_info = np.delete(freq_info,median_freq_arr<f_min,0)
                        time_info = np.delete(time_info,median_freq_arr<f_min,0)
                        isl_num_arr = np.delete(isl_num_arr,median_freq_arr<f_min)
                        isl_area_arr = np.delete(isl_area_arr,median_freq_arr<f_min)
                        
                        mean_freq_arr = np.delete(mean_freq_arr,median_freq_arr<f_min)
                        
                        max_freq_arr = np.delete(max_freq_arr,median_freq_arr<f_min)
                        mean_f_fce_arr = np.delete(mean_f_fce_arr,median_freq_arr<f_min)
                        max_f_fce_arr = np.delete(max_f_fce_arr,median_freq_arr<f_min)
                        wght_mean_freq_arr = np.delete(wght_mean_freq_arr,median_freq_arr<f_min)
                        std_dev_arr = np.delete(std_dev_arr,median_freq_arr<f_min)
                        median_pow_arr = np.delete(median_pow_arr,median_freq_arr<f_min)
                        mean_pow_arr = np.delete(mean_pow_arr,median_freq_arr<f_min)
                        max_pow_arr = np.delete(max_pow_arr,median_freq_arr<f_min)
                        
                        fce_arr = np.delete(fce_arr,median_freq_arr<f_min)
                        rs_arr = np.delete(rs_arr,median_freq_arr<f_min)
                        rs_peri_arr = np.delete(rs_peri_arr,median_freq_arr<f_min)
                        med_f_fce_arr = np.delete(med_f_fce_arr,median_freq_arr<f_min)
                        
                        median_freq_arr = np.delete(median_freq_arr,median_freq_arr<f_min)
                        
                        """
                        f_max = 1.2*fce_day_mean
                        
                        info_arr = np.delete(info_arr,median_freq_arr>f_max,1) #ignore events with frequency greater than 1.2 fce.
                        freq_info = np.delete(freq_info,median_freq_arr>f_max,1)
                        time_info = np.delete(time_info,median_freq_arr>f_max,1)
                        isl_num_arr = np.delete(isl_num_arr,median_freq_arr>f_max)
                        isl_area_arr = np.delete(isl_area_arr,median_freq_arr>f_max)
                        
                        mean_freq_arr = np.delete(mean_freq_arr,median_freq_arr>f_max)                       
                        max_freq_arr = np.delete(max_freq_arr,median_freq_arr>f_max)                                              
                        mean_f_fce_arr = np.delete(mean_f_fce_arr,median_freq_arr>f_max)

                        wght_mean_freq_arr = np.delete(wght_mean_freq_arr,median_freq_arr>f_max)
                        std_dev_arr = np.delete(std_dev_arr,median_freq_arr>f_max)
                        median_pow_arr = np.delete(median_pow_arr,median_freq_arr>f_max)
                        mean_pow_arr = np.delete(mean_pow_arr,median_freq_arr>f_max)
                        max_pow_arr = np.delete(max_pow_arr,median_freq_arr>f_max)
                        
                        fce_arr = np.delete(fce_arr,median_freq_arr>f_max)
                        rs_arr = np.delete(rs_arr,median_freq_arr>f_max)
                        rs_peri_arr = np.delete(rs_peri_arr,median_freq_arr>f_max)                        
                        max_f_fce_arr = np.delete(max_f_fce_arr,median_freq_arr>f_max)
                        med_f_fce_arr = np.delete(med_f_fce_arr,median_freq_arr>f_max)                      
                        median_freq_arr = np.delete(median_freq_arr,median_freq_arr>f_max)
                        """
                        
                        

                        uniq2 = np.unique(isl_tmp[0])
                        uniq2 = np.delete(uniq2,uniq2==0)
                        
                        ignore2 = np.array(uniq2)
                        for i in uniq2: #create a list of all islands to remove from isl_arr1
                            if i in isl_num_arr:
                                ignore2 = np.delete(ignore2,ignore2==i)
                        
                        isl_arr2 = np.array(isl_arr1)
                        
                        for i in ignore2:
                            
                            db_isl_arr[isl_arr2 == i] = np.nan
                            isl_arr2[isl_arr2 == i] = np.nan
        
                        if mode == 'csv' or mode == 'all':
                            
                            time_info = time_info[np.argsort(time_info[:,0])]
                            max_freq_arr = max_freq_arr[np.argsort(time_info[:,0])]
                            max_f_fce_arr = max_f_fce_arr[np.argsort(time_info[:,0])]
                            median_freq_arr = median_freq_arr[np.argsort(time_info[:,0])]
                            med_f_fce_arr = med_f_fce_arr[np.argsort(time_info[:,0])]
                            mean_freq_arr = mean_freq_arr[np.argsort(time_info[:,0])]
                            mean_f_fce_arr = mean_f_fce_arr[np.argsort(time_info[:,0])]
                            fce_arr = fce_arr[np.argsort(time_info[:,0])]
                            rs_arr = rs_arr[np.argsort(time_info[:,0])]
                            rs_peri_arr = rs_peri_arr[np.argsort(time_info[:,0])]
                            
                            
                            wave_date = pys.time_string(time_info[:,0],fmt='%Y-%m-%d/%H:%M:%S')
                            #                 wave date,length of wave,freq max pow,f/fce max pow, median freq,   median f/fce, mean freq,
                            csv_arr = np.array([wave_date,time_info[:,2],max_freq_arr,max_f_fce_arr,median_freq_arr,med_f_fce_arr,mean_freq_arr,
                                                mean_f_fce_arr,fce_arr,rs_arr,rs_peri_arr])
                            #                   mean f/fce    ,fce    ,dist in Rs, distance from perihelion in Rs.
                            
                            
                            csv_arr = np.transpose(csv_arr)
                            #print(csv_arr.shape)
                            
                            csv_savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/'
                            csv_savename = 'enc_'+str(encounter)+'_harmwave_raw.csv'
                            csv_isdir = os.path.isdir(csv_savepath)
                    
                            if csv_isdir == False:
                                os.makedirs(csv_savepath)
                            
                            csv_isfile = os.path.isfile(csv_savepath+csv_savename)
                            
                            if csv_isfile == False:
                                header = np.array(['wave date','length of wave','freq@max pow','f/fce@max pow','median freq','median f/fce', 'mean freq',
                                                   'mean f/fce','fce', 'dist in Rs','dist to peri'])
                                header = np.transpose(header)
                                np.savetxt(csv_savepath+csv_savename,np.array([]),
                                           header='wave date,length of wave,freq@max pow,f/fce@max pow,median freq,median f/fce, mean freq,'+
                                           'mean f/fce,fce, dist in Rs,dist to peri',
                                           fmt='%s',delimiter=',')
                            
                            with open(csv_savepath+csv_savename,"a") as file:
                                
                                np.savetxt(file,csv_arr,delimiter=',',fmt='%s')
                                #f.write("\n") 

                        
                        if mode=='picture' or mode == 'all':
                            
                            f = ticker.ScalarFormatter(useOffset=False, useMathText=True)
                            g = lambda xx,pos : "${}$".format(f._formatSciNotation('%1.10e' % xx))
                            #fig = plt.figure()
                            

                            plt.rcParams['font.size']='24'
                            fig = plt.figure(figsize=(25,20))
                            
                            title = ['Strahl Width','Core Drift','Electron Temperature','Temperature Anisotropy']
                            colorlabs = ['dB','Island Number','Island Number','Island Number']
                            ylabs = ['Spectrogram (Hz)','Islands Routine','Time Filter','Frequency Filter']
                            datas = [ac_data_tmp,isl_arr0,isl_arr1,isl_arr2]
                            
                            for ii in range(4):
                            
                                axs = fig.add_subplot(4,1,ii+1)

                                if ii==0:
                                    axs.set(title=pys.time_string(ac_time_tmp[0],fmt='%Y-%m-%d/%H:%M:%S') + ' Encounter '+str(encounter))
                                    acspec = axs.pcolormesh(ac_time_tmp,ac_freq_lst,datas[ii],cmap='nipy_spectral',shading='nearest')
                                
                                else:
                                    vmax = np.nanmax(isl_arr0)
                                    acspec = axs.pcolormesh(ac_time_tmp,ac_freq_lst,datas[ii],cmap='nipy_spectral',shading='nearest',vmin=0,vmax=vmax)
                                
                                axs.set_yscale("log")

                                axs.yaxis.set_major_formatter(ticker.FuncFormatter(g))

                                box = axs.get_position()
                                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                                
                                fig.colorbar(acspec,cax=axColor, label=colorlabs[ii])
                                axs.set_ylabel(ylabs[ii])
                                if ii!=3:
                                    axs.get_xaxis().set_ticks([])
                                else:
                                    axs.set_xlabel('Seconds since 01-01-1970')
                                    
                            plt.subplots_adjust(wspace=0, hspace=0.05)
                            
                            
                            savepath = "/Users/besh2109/Desktop/psp_islands/"+pys.time_string(ti_doub,fmt='%Y/%m/%d/')
                            savename = "psp_"+pys.time_string(ti_doub,fmt='%Y%m%d%H%M')+"_isl.png"
                    
                            isdir = os.path.isdir(savepath)
                            
                            if isdir == False:
                                os.makedirs(savepath)
                    
                            plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                            plt.clf()
                            plt.cla()

                            plt.close(fig)
                            plt.close('all')
                            
                            
                            
                        if mode == 'waveforms' or mode == 'all': #and pys.time_string(time_info[0,0],fmt='%Y%m%d_%H') == '20200607_07'
                            for i in range(len(isl_num_arr)):
                                
                                t_start = time_info[i,0]
                                t_end = time_info[i,1]
                                t_range = time_info[i,2]
                                
                                time_where = np.where((acspec_time_arr >= t_start-25) & (acspec_time_arr < t_start+t_range+25))
                                time_where = time_where[0]
                                """
                                if len(time_where) < 30:
                                    time_where_tmp = np.array(time_where)
                                    
                                    time_where = np.where((acspec_time_arr >= t_start-15) & (acspec_time_arr < t_start+t_range+15))
                                    time_where = time_where[0]
                                    
                                    #print(t_start)
                                    #print(t_start+t_range)
                                    #print(acspec_time_arr[0])
                                    #print(time_where)
                                    #endpoints = np.array([acspec_time_arr[time_where_tmp[0]],acspec_time_arr[time_where_tmp[len(time_where_tmp)-1]]])
                                """   
                                
                                tmp_time_arr = acspec_time_arr[time_where]  
                                tmp_data_arr = acspec_data_arr[:,time_where]
                                tmp_freq_lst = acspec_freq_arr[:,time_where]
                                
                                tmp_savefold = str(isl_num_arr[i])
                                tmpdate = pys.time_string(time_info[i,0],fmt='%Y%m%d_%H')
                                tmpyear = pys.time_string(time_info[i,0],fmt='%Y')
                                tmpmonth = pys.time_string(time_info[i,0],fmt='%m')
                                tmpday = pys.time_string(time_info[i,0],fmt='%d')
                                tmp_isdir = os.path.isdir('/Users/besh2109/Desktop/psp_islands/waveforms/'+tmpyear+'/'+tmpmonth+'/'+tmpday+'/')
                                if not tmp_isdir:
                                    os.makedirs('/Users/besh2109/Desktop/psp_islands/waveforms/'+tmpyear+'/'+tmpmonth+'/'+tmpday+'/')
                                                          
                                
                                #print(isl_num_arr[i])
                                #tmp_data_arr[tmp_data_arr <= vmin0] = np.nan
                                
                                #endpoints = np.array([acspec_time_arr[time_where[0]],acspec_time_arr[time_where[len(time_where)-1]]])
                                
                                date_tmp = pys.time_string(t_start,fmt='%Y%m%d_%H%M%S')
      
                                #endpoints = np.array([t_start+0.5,t_start+t_range-0.5])
                                endpoints = np.array([t_start+t_range/100,t_start+t_range-t_range/100])
                                
                                height = np.array([7500,7500])
                                fis1 = plt.figure()
                                axs = fis1.add_subplot()
                                axs.pcolormesh(tmp_time_arr,tmp_freq_lst,tmp_data_arr,cmap='nipy_spectral',shading='nearest')
                                #axs.pcolormesh(tmp_data_arr,cmap='nipy_spectral',shading='flat')
                                axs.scatter(endpoints,height, marker='|',s=20000, color='lime')
                                plt.title('wave '+date_tmp)
                                axs.set_yscale("log")
                                
                                plt.savefig('/Users/besh2109/Desktop/psp_islands/waveforms/'+tmpyear+'/'+tmpmonth+'/'+tmpday+'/wave_'+date_tmp+'.png', bbox_inches = 'tight',pad_inches = 0.2)
                                plt.clf()
                                plt.cla()
                                plt.close('all')
                                plt.close(fis1)
                                #plt.show()
                            
                            """
                            del fig
                            del axs1
                            del axs2
                            del axs3
                            del axs4
                            del axColor
                            del acspec1
                            del acspec2
                            del acspec3
                            del acspec4
                            del box
                            del isdir
                            del f
                            del g
                            """
                            
                            #gc.collect()
                            
                            
                    else:
                        pass
                    
                    i_min += 3600.
            else: #do nothing, skip this day
                pass
        
        i_day+=86400.        
        
def quiescent_id(fd='2018-10-03', mode='csv'): #can choose any date for fd, can choose 'csv', 'picture', or 'both' for mode
    
    first_day = fd
    current_day = date.today().strftime("%Y-%m-%d")
    
    i_day = pys.time_float(first_day)
    
    loop_day = pys.time_float(current_day) - 86400.
    
    csv_filename = 'harmwave_master_arch.csv'
    csv_path = '/Users/besh2109/Desktop/PSP_epoch/wave_dates/'
    
    df = pd.read_csv(csv_path+csv_filename)
    bf = df.to_numpy()
    
    wave_starts = np.array(pys.time_float(bf[:,0]))
    
    qual_check = []
    times = []
    
    while i_day < loop_day: #pys.time_float(first_day) + 3*86400.:
        
        t0 = pys.time_string(i_day-21600)
        tf = pys.time_string(i_day + 86400.+21600)
        
        wave_where = np.where((wave_starts>i_day) & (wave_starts<i_day+86400))
        wave_where = wave_where[0]
        
        wave_location = wave_starts[wave_where]
        wave_zeros = np.zeros(len(wave_where))
        
        """
        for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()),key= lambda x: -x[1])[:10]:
                                 
            print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
        """
        
        enci = 0
        for enc in enc_flt:          
            if (i_day>enc[0]) and (i_day<enc[1]): 
                encounter = enci+1 #determine which encounter today is in
                per_date = per_flt[enci]
                per_dist = per_dist_lst[enci]
            enci+=1

        #mag_file_path = CONFIG['local_data_dir']+'/fields/l2/mag_RTN_4_Sa_per_Cyc/'+t0[0:4]+'/'+t0[5:7]+'/'
        mag_file_path = CONFIG['local_data_dir']+'/data/sci/fields/l2/mag_RTN/'+t0[0:4]+'/'+t0[5:7]+'/'
        ephem_file_path = CONFIG['local_data_dir']+'/data/sci/fields/l1/ephem_eclipj2000/'+t0[0:4]+'/'+t0[5:7]+'/'
        
        
        ephem_file_name = pys.time_string(i_day,fmt='spp_fld_l1_ephem_eclipj2000_%Y%m%d_v01.cdf')
        
        isdir_mag = os.path.isdir(mag_file_path)
        isdir_eph = os.path.isdir(ephem_file_path)
        
        if isdir_mag == False:
            os.makedirs(mag_file_path)
        if isdir_eph == False:
            os.makedirs(ephem_file_path)
        
        """ 
        The following could simply be done with psp.fields(arguments), but certain servers pay attention to
        what IPs are hogging data. Particularly the public NASA servers. If I pull too much information for too long
        (like looping through all the Parker Solar Probe data)
        then the servers will seriously slow down how much I can pull after a while. 
        
        Here I'm pulling straight from the Berkeley Servers so that shouldnt
        be an issue but its good practice to check whether you have the data already before unnecessarily making a
        a request of the server.
        """
        

        mag_res = 6*3600 #seconds: that is, 6 hours. The resolution of the magnetic field files.
        
        #mag_file_name = pys.time_string(i_day,fmt='psp_fld_l2_mag_RTN_4_Sa_per_Cyc_%Y%m%d_v01.cdf')
        mag_file_name = []
        mag_isfile = []
        mag_infile = []
        for j in range(-1,5):
            
            mag_file = pys.time_string(i_day+j*21600,fmt='psp_fld_l2_mag_RTN_%Y%m%d%H_v02.cdf')
            mag_file_name.append(mag_file)
            mag_isfile.append(os.path.isfile(mag_file_path+mag_file))
            mag_infile.append(mag_file_path+mag_file)
        
        #print(mag_isfile)
        
        if False in mag_isfile:
            #psp.fields(trange=[t0,tf], datatype='mag_RTN', level='l2')
            #psp.fields(trange=[t0,tf], datatype='mag_RTN_4_Sa_per_Cyc', level='l2',last_version=True)
            psp.fields(trange=[t0,tf], datatype='mag_RTN', level='l2',last_version=True)
        else:
            #print(t0[0:10]+" mag file exists, importing data...")
            #pyt.cdf_to_tplot(mag_infile)
            pyt.cdf_to_tplot(mag_infile)
             
        #mag_data = pyt.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
        mag_data = pyt.get_data('psp_fld_l2_mag_RTN')   #retrieve mag data from tplot variable
        
        if mag_data != None: # if data exists
        
            mag_time_arr = mag_data[0]
            mag_data_arr = mag_data[1]
            
            ephem_isfile = os.path.isfile(ephem_file_path+ephem_file_name)
            
            if ephem_isfile:
                pyt.cdf_to_tplot(ephem_file_path+ephem_file_name)
            else:
                psp.fields(trange=[t0,tf], datatype='ephem_eclipj2000', level='l1') 
            
            pos_data = pyt.get_data('position') #retrieve PSP position data from tplot variable
    
            pos_time_arr = pos_data[0]
            pos_data_arr = pos_data[1]
            
            
            # print(pys.time_string(i_day))
            # print(np.mean(np.sqrt(pos_data_arr[0]**2+pos_data_arr[1]**2+pos_data_arr[2]**2)/Rs),' Rs')

            i=0
            while i < 2880:
                #print(i)
                fftwhere = np.where((mag_time_arr>i_day+30*i-30) & (mag_time_arr<i_day+30*i+30))
                fftwhere = fftwhere[0]
                
                time_pos_flt = i_day+30*i
                
                
                
                mag_time_tmp = mag_time_arr[fftwhere]
                mag_r_tmp = mag_data_arr[fftwhere,0]
                mag_t_tmp = mag_data_arr[fftwhere,1]
                mag_n_tmp = mag_data_arr[fftwhere,2]
                
                #print(len(fftwhere))
                
                if len(fftwhere) < 360:
                    qual_check.append(np.nan)
                    # print('yeeyee')
                else:
                    j=0
                    res = []
                    while j<60:
                        res.append(np.round(mag_time_tmp[j+1]-mag_time_tmp[j],3))
                        j+=1
                    res = np.array(res)
                    av_res = round(np.mean(res),3)
                    # print(av_res)
                    # print(res)
                    if False in res==av_res: #if the resolution deviates (significantly) across a window.
                        qual_check.append(np.nan)
                    else:
                        
                        Nr = len(mag_r_tmp)
                        Nt = len(mag_t_tmp)
                        Nn = len(mag_n_tmp)
                        
                        r_fft = fft(mag_r_tmp)[:Nr//2]/Nr
                        t_fft = fft(mag_t_tmp)[:Nt//2]/Nt
                        n_fft = fft(mag_n_tmp)[:Nn//2]/Nn
                        
                        r_fft = np.abs(r_fft)**2
                        t_fft = np.abs(t_fft)**2
                        n_fft = np.abs(n_fft)**2
                        
                        r_fft_freq = fftfreq(Nr,av_res)[:Nr//2]
                        t_fft_freq = fftfreq(Nt,av_res)[:Nt//2]
                        n_fft_freq = fftfreq(Nn,av_res)[:Nn//2]
                        
                        r_fft_where = np.where((r_fft_freq>=0.05) & (r_fft_freq<=1))
                        r_fft_where=r_fft_where[0]
                        
                        r_fft_wind = r_fft[r_fft_where]
                        
                        t_fft_where = np.where((t_fft_freq>=0.05) & (t_fft_freq<=1))
                        t_fft_where=t_fft_where[0]
                        
                        t_fft_wind = t_fft[t_fft_where]
                        
                        n_fft_where = np.where((n_fft_freq>=0.05) & (n_fft_freq<=1))
                        n_fft_where=n_fft_where[0]
                        
                        n_fft_wind = n_fft[n_fft_where]
                        
                        freq_wind = r_fft_freq[r_fft_where]
                        
                        # plt.plot(r_fft_freq, (np.abs(r_fft+t_fft+n_fft))/r_fft_freq**-(5/3))
                        # plt.xlim((0.05,1.2))
                        # plt.ylim((0,3000))
                        # plt.show()
                        quality = np.mean((r_fft_wind+t_fft_wind+n_fft_wind)/freq_wind**-(5/3))
                        quality = quality/r_fft_freq[1]
                        
                        qual_check.append(quality)
                        
                times.append(time_pos_flt)
                i+=1
            
            
            # print(r_fft_freq[0:4], 'Hz')
            
            # qual_check = np.array(qual_check)
            # times = np.array(times)
            
            # #breakpoint()
            
            # fig = plt.figure(figsize=(15,10))
            
            # ax = fig.add_subplot(111)
            
            # ax.plot(times,np.log10(qual_check))
            # ax.scatter(wave_location,wave_zeros,marker='|',color='green', s=10000)
            # plt.show()
            

        i_day+=86400.
    
    savepath = '/Users/besh2109/Desktop/'
    tplotfile = 'turbulence_quality_check.tplot'
    npfile = 'turbulence_quality_check.npy'
    
    qual_check = np.array(qual_check)
    times = np.array(times)
    
    save_arr = np.array([times,qual_check])
    
    np.save(savepath+npfile,save_arr)
    
    pyt.store_data('turbulence_quality_check',data={'x':times,'y':qual_check})
    pyt.tplot_save('turbulence_quality_check',filename=savepath+tplotfile)
    
    