#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 14 08:15:52 2022

@author: besh2109
"""

import os
import pyspedas.psp as psp
import pyspedas as pys
import matplotlib.pyplot as plt
import numpy as np
import pytplot as pyt
from scipy.interpolate import interp1d
# from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
# from datetime import date
# from matplotlib import ticker
# from statistics import median
# from mpl_toolkits.axes_grid1 import make_axes_locatable
# import gc
# import math
import random
# import cartopy.crs as ccrs
# import pyspedas.stereo as ste
# from matplotlib import gridspec
import time
import pickle as pkl

from .config import CONFIG
from .config import enc_flt
from .config import per_flt
from .config import per_dist_lst

from findpeaks import findpeaks
from scipy.interpolate import UnivariateSpline

import pandas as pd

import matplotlib.colors
import dateutil.parser
# import matplotlib.pyplot as plt


fields_id = os.environ['PSP_FIELDS_ID']
fields_pass = os.environ['PSP_FIELDS_PW']

sweap_id = os.environ['PSP_SWEAP_ID']
sweap_pass = os.environ['PSP_SWEAP_PW']

jsoc_email = os.environ['JSOC_EMAIL']

Rs = 6.957e5 #solar radius in km
Rs_in_m = Rs*10**3 #solar radius in m
w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec

def find_nan_groups_indices(arr):
    nan_indices = np.where(np.isnan(arr))[0]
    nan_groups_indices = np.split(nan_indices, np.where(np.diff(nan_indices) != 1)[0]+1)
    nan_groups_indices = [group for group in nan_groups_indices if len(group) > 0]
    return nan_groups_indices

def nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.

    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]

def sliding_median(array, window_size):
    # Pad the array to handle edge cases
    padded_array = np.pad(array, (window_size//2, window_size//2), mode='edge')

    # Create a 2D array of shape (len(array), window_size)
    rolling_window = np.lib.stride_tricks.sliding_window_view(padded_array, window_shape=(window_size,))

    # Compute the median along the window axis
    medians = np.nanmedian(rolling_window, axis=1)
    
    # Determine if the number of NaNs exceeds half the window size
    num_nans = np.sum(np.isnan(rolling_window), axis=1)
    too_many_nans = num_nans > 3*window_size / 4
    
    # Set medians to np.nan where there are too many NaNs
    medians[too_many_nans] = np.nan
    
    return medians


def split_data_evenly_in_time(time_array, data_array, num_segments):
    # Calculate the total time span
    total_time_span = time_array[-1] - time_array[0]

    # Calculate the time interval for each segment
    segment_interval = total_time_span / num_segments

    # Compute the indices corresponding to the boundaries of each segment
    segment_indices = [np.searchsorted(time_array, time_array[0] + i * segment_interval) for i in range(1, num_segments)]

    # Split the data into segments based on the computed indices
    time_segments = np.split(time_array, segment_indices)
    data_segments = np.split(data_array, segment_indices)

    return time_segments, data_segments

def group_zeros(array):
    
    # Example array of 1s and 0s
    # array = np.array([1, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0])
    
    # Initialize variables to track group boundaries
    group_started = False
    start_index = None
    end_index = None
    
    # List to store the start and end indices of groups of 0s
    groups = []
    
    # Iterate through the array
    for i, value in enumerate(array):
        if value == 0:
            if not group_started:
                # Start of a new group
                start_index = i
                group_started = True
            end_index = i
        elif group_started:
            # End of the current group
            groups.append((start_index, end_index))
            group_started = False
    
    # If the last group continued to the end of the array, add it
    if group_started:
        groups.append((start_index, end_index))
    
    # Print the start and end indices of groups of 0s
    # for start, end in groups:
    #     print(f"Group of 0s: Start Index {start}, End Index {end}")
    
    return groups

def assign_closest_index(arr_one, arr_two):
    # Sort arr_one for binary search
    sorted_arr_two = np.sort(arr_two)

    # Initialize an empty array to store assigned values
    assigned_index = np.empty_like(arr_one,dtype=int)

    # Iterate over each value in arr_one
    for i, val in enumerate(arr_one):
        # Find the index of the closest value in arr_two using binary search
        closest_index = np.abs(sorted_arr_two - val).argmin()
        
        if i%55000==0:
            print(round(i*100/len(arr_one)),'%')
        # Assign the closest index from arr_one to arr_two
        assigned_index[i] = closest_index

    return assigned_index

def assign_closest_index_vec(arr_one, arr_two):
    # Calculate the absolute differences between each element of arr1 and arr2
    abs_diff = np.abs(arr_one[:, None] - arr_two)

    # Find the index of the minimum absolute difference for each element of arr1
    closest_indices = np.argmin(abs_diff, axis=1)

    # # Use the indices to extract the closest values from arr2
    # closest_values = arr2[closest_indices]

    return closest_indices

def quiescent_id(analysis='all', mode='csv',runs=20,enc_start=1,enc_end=13,thresh=0.95,bincount=499,full_run=True): # analysis chooses type of analysis you want to do, mode is broken, 
                                                                  # runs means number of runs to do the random analysis, 
                                                                  # enc_start specifies the encounter to start on.
        
    if runs<1:
        print()
        print("Invalid run count. Number entered ("+str(runs)+") is < 1. Setting to 1 run.")
        print()
        time.sleep(4)
        runs = 1
    if runs>5000:
        print()
        print("You do not need "+str(runs)+" runs. Setting it to 5000.")
        print()
        time.sleep(4)
        runs = 5000
        
    enc_num = enc_start
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    enc_ind = (enc_num-1)

    for enc in enc_flt[enc_ind:enc_end]:
        enc_str = pys.time_string(enc[0])
        enc_end = pys.time_string(enc[1])
        
        hpos_where = np.where((hpos_time_arr>enc[0])&(hpos_time_arr<enc[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<65)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        # breakpoint()
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        title = 'Encounter '+str(enc_num)+': '+t0p[0:20]+' to '+tfp[0:20]
        
        z_save_path = '/Users/besh2109/Desktop/z_save/tplot/'
        z_save_name = 'z_tplot_enc'+str(enc_num)+'.cdf'
        
        filecheck = os.path.isfile(z_save_path+z_save_name)
        
        # breakpoint()
        
        if filecheck:
            pyt.tplot_restore(z_save_path+z_save_name)
            
            z_data = pyt.get_data('z')
            z_time = z_data[0]
            z = z_data[1]
            
            quiet_z_data = pyt.get_data('quiet_z')
            quiet_z = quiet_z_data[1]
            
            b_data = pyt.get_data('Br')
            b_time = b_data[0]
            Br = b_data[1]
            
            quiet_B_data = pyt.get_data('quiet_B')
            quiet_B = quiet_B_data[0]
            B_mag = quiet_B_data[1]
            
            carr_data = pyt.get_data('carr_lon')
            carr_time = carr_data[0]
            carr_long = carr_data[1]
            
            carr_data = pyt.get_data('carr_lat')
            carr_lat = carr_data[1]
            
            carr_data = pyt.get_data('carr_dif')
            carr_dif = carr_data[1]
            
        else:
            varis = quiescent_calc(t0=t0,tf=tf)
    
            z_time = varis['z_time']
            z = varis['z']
            quiet_z = varis['quiet_z']
            b_time = varis['b_time']
            Br = varis['Br']
            quiet_B = varis['quiet_B']
            B_mag = varis['B_mag']
            
            carr_time = varis['carr_time']
            carr_long = varis['carr_long']
            carr_lat = varis['carr_lat']
            carr_dif = varis['carr_dif']
            
        # breakpoint()
        
        analysis_type = np.array(['long','time','rand'])
        
        if analysis == 'all':
            analyses=analysis_type
        elif analysis in ['no rand','no long','no time']:
            no_type = analysis[3:]
            no_check = np.array(analysis_type)
            no_where = np.where(no_check != no_type)
            no_where = no_where[0]
            analyses = no_check[no_where]
        elif analysis in analysis_type:
            analyses = np.array([analysis])
        else:
            analyses = np.array(['time'])
            print('')
            print("Analysis specified not in type list, using time. Acceptable keywords are 'long' (longitude) ,'time' (time), and 'rand' (random).")
            print('')
        
        

        
        z_i = np.array(z)
        
        nans, x = nan_helper(z_i)
        nan_group_indices = find_nan_groups_indices(z_i)
        
        long_arr = np.array([],dtype=int)
        for bbeg in nan_group_indices:
            if len(bbeg)>200:
                long_arr = np.append(long_arr,bbeg)
                
        nans[long_arr] = False
        
        #interpolate over the nans that are very short
        z_i[nans] = np.interp(x(nans), x(~nans), z_i[~nans])
        
        nanwhere = np.isnan(z_i)
        
        nanlst=[]
        
        for i in range(len(nanwhere)-1):
            if nanwhere[i]==nanwhere[i+1]: 
                nanlst.append(False)
            else:
                nanlst.append(True)
        
        if np.isnan(z_i[-1]):
            nanlst.append(True)
        else:
            nanlst.append(False)
        
        nanlst_arr = np.array(nanlst)
        
        truewhere = np.where(nanlst_arr)
        
        bindices = truewhere[0] #bin indices, or BINdicess
        
        bin_lst = []
        bin_time_lst = []
        for i in range(int(len(bindices)/2)):
            bin_lst.append([bindices[2*i],bindices[2*i+1]])
            bin_time_lst.append([z_time[bindices[2*i]],z_time[bindices[2*i+1]]])

        bindices = np.array(bin_lst)

        bin_edges = np.array(bin_time_lst)
        # bin_edges = np.sort(bin_edges)
        
        # breakpoint()
        
        for j in range(len(analyses)):
            
            atype = analyses[j]
        #---------------- longitude analysis, set bars evenly in longitude --------------------#
            if atype == 'long':
                i_max = len(carr_time) #maximum index, i 
                wndw_bars = np.array(carr_time[0])
                
                csv_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/longitude_regions/'
                csv_savename = 'enc_'+str(enc_num)+'_quiescent_longitude.csv'
                
                carr_int = 0
                for i in range(i_max):
                    
                    carr_int+=carr_dif[i]
                    
                    if np.abs(carr_int) > 0.82: #0.082 degrees for granulation scale, 0.82 is supergranulation scale
                        wndw_bars = np.append(wndw_bars,carr_time[i])
                        carr_int = 0
                    if i==i_max-1:
                        wndw_bars = np.append(wndw_bars,z_time[-1])
                
                for jjj in bin_edges:
                    bar_where = np.where((wndw_bars>=jjj[0])&(wndw_bars<=jjj[1]))
                    bar_where = bar_where[0]
                    
                    wndw_bars = np.delete(wndw_bars,bar_where)
                    wndw_bars = np.append(wndw_bars,jjj)
                
                wndw_bars = np.sort(wndw_bars)
                wndw_bars_shape = wndw_bars.shape
                
                bin_store_l = wndw_bars
                
                # fis1 = plt.figure(figsize=(15,5))
                # axs = fis1.add_subplot(1,1,1)
                # axs.plot(b_time,Br)
                # axs.vlines(wndw_bars,color='green',ymax=50,ymin=-50)
                # plt.show()
                
                # breakpoint()
            
        #------------------ time analysis, set bars evenly in time ---------------------------#
        
            if atype == 'time':
                i_max = len(z_time)
                # wndw_bars = np.array(z_time[0])
                
                csv_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/time_regions/'
                csv_savename = 'enc_'+str(enc_num)+'_quiescent_time.csv'
                
                bin_size = i_max/500
                
                bins = np.arange(501)*bin_size
                bins[-1] = np.floor(bins[-1])-1 #makes sure the last index is correctly in indexing range.
                bins = np.array(np.round(bins),dtype=np.int32)
                
                
                for jjj in bindices:
                    bar_where = np.where((bins>=jjj[0])&(bins<=jjj[1]))
                    bar_where = bar_where[0]
                    
                    bins = np.delete(bins,bar_where)
                    bins = np.append(bins,jjj)
                
                bins = np.sort(bins) #sorts the indices in increasing order.
                
                wndw_bars = z_time[bins]
                wndw_bars_shape = wndw_bars.shape
                
                bin_store_t = z_time[bins]
                
                # fis1 = plt.figure(figsize=(15,5))
                # axs = fis1.add_subplot(1,1,1)
                # axs.plot(b_time,Br)
                # axs.vlines(wndw_bars,color='green',ymax=50,ymin=-50)
                # plt.show()
                
                # breakpoint()
                
                
        #----------------------- random analysis, set bars randomly --------------------------#

            if atype == 'rand':
                # wndw_bars = np.array([])
                wndw_list = []
                # breakpoint()
                for jj in range(runs): #runs = number of random runs
                    i_max = len(z_time)
                    # wndw_bars_i = np.array(z_time[0])
                    
                    # csv_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/random_regions/'
                    # csv_savename = 'enc_'+str(enc_num)+'_quiescent_random.csv'
                    
                    bins = np.array([0])
                    bins = np.append(bins,random.sample(range(i_max),bincount))
                    bins = np.append(bins,np.array(i_max-1))
                    
                    for jjj in bindices:
                        bar_where = np.where((bins>=jjj[0])&(bins<=jjj[1]))
                        bar_where = bar_where[0]
                        
                        bins = np.delete(bins,bar_where)
                        bins = np.append(bins,jjj)
                    
                    bins = np.sort(bins) #sorts the indices in increasing order.
                    
                    wndw_list.append(list(z_time[bins]))
                    # wndw_bars_tmp = np.array(wndw_list)
                    
                    # wndw_bars = np.append(wndw_bars,z_time[bins], axis=0)
                    bin_store_r = z_time[bins]
                    
                    # fis1 = plt.figure(figsize=(15,5))
                    # axs = fis1.add_subplot(1,1,1)
                    # axs.plot(b_time,Br)
                    # axs.vlines(wndw_bars,color='green',ymax=50,ymin=-50)
                    # plt.show()
                    
                    
                    # if jj==4:
                    #     breakpoint()
                    
            
                        
                # wndw_bars_shape = wndw_bars.shape
            
            # print(wndw_bars_shape)
            
           
            
            # if atype=='rand':
            #     breakpoint()
            
            if atype == 'rand' and runs > 1: #this should only be used for random bars.
                
                # pos_len = 50
                # progress = np.linspace(0,pos_len,21)
                range_check = np.arange(10)/10*runs
                # i=0
                q_z_lst = []
                q_z_add = np.zeros(len(z_time))
                # breakpoint()
                for kk in range(len(wndw_list)):
                    # print(kk)
                    wnd_strs = wndw_list[kk][0:-1]
                    wnd_ends = wndw_list[kk][1:]
                    
                    # csv_arr = np.array([])
                    q_z_arr = np.array([])
                    
                    
                    
                    if kk in range_check:
                        print(str(100*kk/runs)+'% Complete')
                    
                    # breakpoint()
                    
                    for iii in range(len(wnd_strs)):
                        
                        if iii == len(wnd_strs)-1:
                            z_where = np.where((z_time>=wnd_strs[iii]) & (z_time<=wnd_ends[iii])) #the last bin includes the last data point
                        else:
                            z_where = np.where((z_time>=wnd_strs[iii]) & (z_time<wnd_ends[iii]))
                        
                        
                        
                        
                        start_date = pys.time_string(wnd_strs[iii])
                        end_date = pys.time_string(wnd_ends[iii])
                        
                        z_time_in_bin = z_time[z_where]
                        z_in_bin = z[z_where]
                        
                        q_z_where = np.where((z_in_bin<0.05) | (z_in_bin>0.95))
                        
                        q_z_pnts = len(q_z_where[0])
                        z_pnts = len(z_in_bin)
                        
                        vari = np.nanvar(z_in_bin)
                        std = np.nanstd(z_in_bin)
                        med_z = np.nanmedian(z_in_bin)
                        
                        med_z = med_z - 1/2
                        med_z = np.sqrt(med_z**2)
                        
                        
                        if z_pnts==0:
                            q_z_frac=np.nan
                        
                        else:
                            q_z_frac = q_z_pnts/z_pnts
                        
                        if q_z_frac < thresh:
                            # q_z_frac = q_z_frac*0.3 #suppress values with low q/z
                            q_z_frac = 0
                        else:
                            # q_z_frac = q_z_frac*2 #inflate values with high q/z
                            q_z_frac = 1
                        # q_z_tmp_arr = (np.ones(len(z_in_bin))*q_z_frac)/((vari+1)*(med_z+1))
                        # q_z_tmp_arr = np.ones(len(z_in_bin))*(q_z_frac)/((std+1)*(med_z+1))
                        
                        q_z_tmp_arr = np.ones(len(z_in_bin))*(q_z_frac)
                        
                        q_z_arr = np.append(q_z_arr,q_z_tmp_arr) #create array that gives a q/z fraction for every data point.
                        
                        q_z_arr = q_z_arr #this line ensures that q_z_arr maintains the correct shape through the np.append. Seems redundant, but it is not.
                        
                        
                        
                    # q_z_lst.append(q_z_arr)
                    q_z_add = q_z_add + q_z_arr
                    if kk == 0:
                        qual_r_one = q_z_arr
                        
                    if kk == 4:
                        qual_r_five = q_z_add/5
                # q_z_tot_arr = np.array(q_z_lst)
                # q_z_med_arr = np.nanmedian(q_z_tot_arr,axis=0)
                # q_z_mean_arr = np.nanmean(q_z_tot_arr,axis=0)
                
                qual_r_arr = q_z_add/runs #*(1/z)
                # qual_r_arr = qual_r_arr*(1/z)
                
                # qual_r_arr = q_z_med_arr
                
                # plt.plot(q_z_mean_arr,linewidth=0.4)
                # plt.show()
                # plt.plot(z,linewidth=0.05)
                # plt.show()
                # breakpoint()
                # print('yeet')
                
            #----------------------- all other analysis get run through this --------------------------#
                    
            else:
                wnd_strs = wndw_bars[0:-1]
                wnd_ends = wndw_bars[1:]
                
                # breakpoint()
                
                csv_arr = np.array([])
                q_z_arr = np.array([])
                
                for iii in range(len(wnd_strs)):
                    # z_where = np.where((z_time>=wnd_strs[iii]) & (z_time<wnd_ends[iii]))
                    
                    if iii == len(wnd_strs)-1:
                        z_where = np.where((z_time>=wnd_strs[iii]) & (z_time<=wnd_ends[iii])) #the last bin includes the last data point
                    else:
                        # print(iii, atype)
                        z_where = np.where((z_time>=wnd_strs[iii]) & (z_time<wnd_ends[iii]))
                    
                    start_date = pys.time_string(wnd_strs[iii])
                    end_date = pys.time_string(wnd_ends[iii])
                    
                    z_time_in_bin = z_time[z_where]
                    z_in_bin = z[z_where]
                    
                    q_z_where = np.where((z_in_bin<0.05)|(z_in_bin>0.95))
                    
                    q_z_pnts = len(q_z_where[0])
                    z_pnts = len(z_in_bin)
                    
                    if z_pnts==0:
                        q_z_frac=np.nan
                        # breakpoint()
                    
                    else:
                        q_z_frac = q_z_pnts/z_pnts
                    
                    if q_z_frac < thresh:
                        q_z_frac = q_z_frac*0.2 #suppress values with low q/z 
                    
                    q_z_tmp_arr = np.ones(len(z_in_bin))*q_z_frac
                    
                    q_z_arr = np.append(q_z_arr,q_z_tmp_arr) #create array that gives a q/z fraction for every data point.
                    
                    # csv_arr_tmp = np.array([start_date,end_date,q_z_frac])
                    # csv_arr_tmp = np.transpose(csv_arr_tmp)
                    
                    # csv_arr = np.append(csv_arr,csv_arr_tmp,axis=0)
            
                
                if atype == 'time':
                    qual_t_arr = q_z_arr
                if atype == 'long':
                    qual_l_arr = q_z_arr
                if atype == 'rand':
                    qual_r_arr = q_z_arr
        
        # plt.plot(z_time,qual_l_arr)
        # plt.show()
        # plt.plot(z_time,z)
        # plt.show()
        
        tplot_savename = 'Enc_'+str(enc_num)+'_quiescent_flags.cdf'
        tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/'

        tplot_time = z_time
        tplot_z = z
        tplot_qual_l = qual_l_arr
        tplot_qual_t = qual_t_arr
        tplot_qual_r = qual_r_arr
        
        tplot_qual_r_one = qual_r_one
        tplot_qual_r_five = qual_r_five
        
        tplot_carr_time = carr_time
        tplot_carr = np.array([carr_long,carr_lat])
        
        run_count = np.array([runs])
        # breakpoint()
        
        pyt.store_data("z", data={'x':tplot_time, 'y':tplot_z})
        
        pyt.store_data("B_mag",data={'x':b_time,'y':B_mag})
        
        pyt.store_data("Br", data={'x':b_time, 'y':Br})
        # print(1)
        pyt.store_data("qual_l", data={'x':tplot_time, 'y':tplot_qual_l})
        # print(2)
        pyt.store_data("qual_t", data={'x':tplot_time, 'y':tplot_qual_t})
        # print(3)
        pyt.store_data("qual_r", data={'x':tplot_time, 'y':tplot_qual_r})
        
        pyt.store_data("qual_r_one", data={'x':tplot_time, 'y':tplot_qual_r_one})
        
        pyt.store_data("qual_r_five", data={'x':tplot_time, 'y':tplot_qual_r_five})
        
        # print(4)
        
        pyt.store_data("carr_coords", data={'x':tplot_carr_time, 'y':np.transpose(tplot_carr)})
        # print(5)
        
        pyt.store_data("bins_l",data={'x':bin_store_l, 'y':bin_store_l})
        # print(6)
        pyt.store_data("bins_t",data={'x':bin_store_t, 'y':bin_store_t})
        # print(7)
        pyt.store_data("bins_r_last",data={'x':bin_store_r, 'y':bin_store_r}) #stores last random bin set
        # print(8)
        pyt.store_data("run_count",data={'x':run_count, 'y':run_count})
        
        cdf_var_list = ["z","B_mag","Br","qual_l","qual_t","qual_r","qual_r_one","qual_r_five","carr_coords","bins_l","bins_t","bins_r_last","run_count"]
        
        pyt.tplot_save(cdf_var_list,tplot_savepath+tplot_savename) #saves the quality flags to a .cdf file
        
        
        if full_run: #if you specify full run, this routine will go ahead and write results to csv.
            quiescent_id_thresh(enc=enc_num,mode='csv')
        
        
        enc_num+=1
        
def quiescent_id_thresh(enc='all',kind='rand',thresh=0.5,split_num=6,mode='soloplot'):
    
    if enc == 'all':
        enc = list(range(1,14))
        
    if type(enc) is int:
        enc = [enc]
    
    for i in enc:
        
        tplot_filepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/'
        tplot_filename = 'Enc_'+str(i)+'_quiescent_flags.cdf'
        
        # pyt.tplot_names()
        
        pyt.tplot_restore(tplot_filepath+tplot_filename)
        
        run_data = pyt.get_data('run_count')
        runs = run_data[0]
        
        z_data = pyt.get_data('z')
        
        z_time = z_data[0]
        z = z_data[1]
        
        b_mag_data = pyt.get_data('B_mag')
        
        b_mag_time = b_mag_data[0]
        b_mag = b_mag_data[1]
        
        b_data = pyt.get_data('Br')
        
        b_time = b_data[0]
        Br = b_data[1]
        
        qual_r_data = pyt.get_data('qual_r')
        
        qual_r_time = qual_r_data[0]
        qual_r = qual_r_data[1]
        
        qual_r_where = np.where((qual_r>thresh))#&((z<0.05)|(z>0.95)))

        qual_l_data = pyt.get_data('qual_l')
        
        qual_l_time = qual_l_data[0]
        qual_l = qual_l_data[1]
        
        qual_l_where = np.where(qual_l>thresh)
        
        qual_t_data = pyt.get_data('qual_t')
        
        qual_t_time = qual_t_data[0]
        qual_t = qual_t_data[1]
        
        qual_t_where = np.where(qual_t>thresh)
        
        
        
        qual_r_one_data = pyt.get_data('qual_r_one')
        qual_r_one_time = qual_r_one_data[0]
        qual_r_one = qual_r_one_data[1]
        
        qual_r_five_data = pyt.get_data('qual_r_five')
        qual_r_five_time = qual_r_five_data[0]
        qual_r_five = qual_r_five_data[1]
        
        
        
        if kind == 'rand':
            quiescent_where = qual_r_where[0]
        elif kind == 'long':
            quiescent_where = qual_l_where[0]
        elif kind == 'time':
            quiescent_where = qual_t_where[0]
        else:
            quiescent_where = qual_r_where[0]


        z_where = np.where((z>0.95)|(z<0.05))   
        # z_where = np.where(z<0.05)   
        z_where = z_where[0]
        
        q_z_time = b_time
        q_z = np.empty(Br.shape)
        q_z[:] = np.nan
        q_z[z_where] = Br[z_where]

        quiet_time = b_time
        
        quiet_b = np.empty(Br.shape)
        quiet_b[:] = np.nan
        
        quiet_b[quiescent_where]=Br[quiescent_where]
        
        # breakpoint()
        
        if mode in ['zplot','both']:
        
            fis1 = plt.figure(figsize=(15,9))
            
            # times = [b_time,qual_r_time,qual_l_time,qual_t_time]
            # datas = [Br,qual_r,qual_l,qual_t]
            # labels = ['Br (nT)','random qual','long qual','time qual']
            
            times = [b_time,qual_r_time,qual_r_time,qual_r_time]
            # times = [z_time,qual_r_time,qual_r_time,qual_r_time]
            datas = [Br,qual_r_one,qual_r_five,qual_r]
            # datas = [z,qual_r_one,qual_r_five,qual_r]
            labels = ['nT','quality flag, 1 run','quality flag, 5 runs','quality flag, '+str(int(runs[0]))+' runs']
            names = ['Br','1 iteration','5 iterations',str(int(runs[0]))+' iterations']
            
            
            split_num = split_num
            
            tick_num = split_num+1
            
            min_rat = 2.69/split_num
            min_ind = round(min_rat*len(b_time))
            
            max_rat = 3.795/split_num
            max_ind = round(max_rat*len(b_time))
            
            all_inds = np.round(np.linspace(min_ind,max_ind,tick_num)).astype(int)

            # breakpoint()
            x_label_list = []
            x_axes = []
            for iii in range(tick_num):
                if iii == 0:
                    x_label_list.append(pys.time_string(b_time[all_inds[iii]],fmt='%Y-%m-%d/%H:%M:%S')) #first label
                    x_axes.append(b_time[all_inds[iii]])
                elif iii == tick_num-1:
                    x_label_list.append(pys.time_string(b_time[all_inds[iii]-1],fmt='%m-%d/%H:%M:%S')) #last label
                    x_axes.append(b_time[all_inds[iii]-1])
                else:
                    x_label_list.append(pys.time_string(b_time[all_inds[iii]],fmt='%m-%d/%H:%M:%S')) #middle labels
                    x_axes.append(b_time[all_inds[iii]])
            
            

            # x_label_list = [pys.time_string(b_time[0],fmt='%Y-%m-%d/%H:%M:%S'),\
            #                 pys.time_string(b_time[round(len(b_time)/6)],fmt='%m-%d/%H:%M:%S'), \
            #                 pys.time_string(b_time[round(2*len(b_time)/6)],fmt='%m-%d/%H:%M:%S'), \
            #                 pys.time_string(b_time[round(3*len(b_time)/6)],fmt='%m-%d/%H:%M:%S'), \
            #                 pys.time_string(b_time[round(4*len(b_time)/6)],fmt='%m-%d/%H:%M:%S'), \
            #                 pys.time_string(b_time[round(5*len(b_time)/6)],fmt='%m-%d/%H:%M:%S'), \
            #                 pys.time_string(b_time[len(b_time)-1],fmt='%m-%d/%H:%M:%S')]
                

            
            for j in range(4):
                
                if j!=3:    
                    axs = fis1.add_subplot(4,1,j+1)#,sharex=axs)
                else:
                    axs = fis1.add_subplot(4,1,j+1,sharex=axs)


                    
                if j == 0:
                    axs.plot(times[j],datas[j],linewidth=0.1,label=names[j])
                    axs.set_ylabel(labels[j])
                    
                    axs.set_title('Encounter '+str(i)+', Run count:'+str(int(runs[0])))
                    a = axs.plot(quiet_time,quiet_b,linewidth=0.1,label='Quiescent Br')
                    # a = axs.plot(z_time,z,linewidth=0.1,label='Quiescent Br')
                    axs.tick_params(axis='x',labelbottom=False,direction='in')
                    lgnd = axs.legend(loc='upper left')
                    axs.axhline(y = 0, color = 'grey', linestyle = '--')
                    
                    # for legobj in lgnd.legendHandles:
                    #     legobj.set_linewidth(2.0)
                    
                elif j!=0 and j!= 3:
                    axs.plot(times[j],datas[j],linewidth=0.75,label=names[j])
                    axs.set_ylabel(labels[j])
                    axs.tick_params(axis='x',labelbottom=False,direction='in')
                    lgnd = axs.legend(loc='upper right')
                else:
                    axs.plot(times[j],datas[j],linewidth=0.75,label=names[j])
                    axs.set_ylabel(labels[j])
                    lgnd = axs.legend(loc='upper right')
                
                if j != 0:
                    
                    axs.axhline(y = 0.5, color = 'grey', linestyle = '--')
                
                
                for legobj in lgnd.legendHandles:
                    legobj.set_linewidth(2.0)
                        
                # axs.set_xticks([b_time[0],b_time[round(len(b_time)/6)],b_time[round(2*len(b_time)/6)],\
                #                    b_time[round(3*len(b_time)/6)],b_time[round(4*len(b_time)/6)],\
                #                        b_time[round(5*len(b_time)/6)],b_time[len(b_time)-1]])
                # axs.set_xticklabels(x_label_list,rotation = 0)
                # axs.set_xlim([b_time[round(1*len(b_time)/5)],b_time[round(2.5*len(b_time)/5)]])
                axs.set_xticks(x_axes)
                axs.set_xticklabels(x_label_list,rotation = 0)
                axs.set_xlim([b_time[min_ind],b_time[max_ind-1]])
                # axs.xticks(rotation = 45)

            plt.subplots_adjust(wspace=0, hspace=0.012)
            plt.show()
            
        if mode in ['deWit']:
            
            fis1 = plt.figure(figsize=(12,7.5))
            
            # times = [b_time,qual_r_time,qual_l_time,qual_t_time]
            # datas = [Br,qual_r,qual_l,qual_t]
            # labels = ['Br (nT)','random qual','long qual','time qual']
            
            times = [b_time,b_time]
            datas = [Br,Br]
            labels = ['nT','nT']
            names = ['Br','Br']
            label1 = r"Q Regions"\
                     "\n"  \
                     r"(de Wit)"
            label2 = r"Q Regions"\
                     "\n"  \
                     r"(this study)"
                     

            x_label_list = [pys.time_string(b_time[0],fmt='%Y-%m-%d/%H:%M:%S'),\
                            pys.time_string(b_time[round(len(b_time)/6)],fmt='%m-%d/%H:%M:%S'), \
                            pys.time_string(b_time[round(2*len(b_time)/6)],fmt='%Y-%m-%d/%H:%M:%S'), \
                            pys.time_string(b_time[round(3*len(b_time)/6)],fmt='%Y-%m-%d/%H:%M:%S'), \
                            pys.time_string(b_time[round(4*len(b_time)/6)],fmt='%m-%d/%H:%M:%S'), \
                            pys.time_string(b_time[round(5*len(b_time)/6)],fmt='%m-%d/%H:%M:%S'), \
                            pys.time_string(b_time[len(b_time)-1],fmt='%m-%d/%H:%M:%S')]
                
            
            
            for j in range(2):
                
                if j ==0:
                    axs = fis1.add_subplot(2,1,j+1)
                else:
                    axs = fis1.add_subplot(2,1,j+1,sharex=axs)

                axs.plot(times[j],datas[j],linewidth=0.1,label=names[j])
                axs.set_ylabel(labels[j])
                    
                if j == 0:
                    axs.set_title('Encounter '+str(i)+', Run count:'+str(runs[0]))
                    a = axs.plot(q_z_time,q_z,linewidth=0.1,label=label1)
                    axs.tick_params(axis='x',labelbottom=False,direction='in')
                    lgnd = axs.legend()
                    
                    for legobj in lgnd.legendHandles:
                        legobj.set_linewidth(2.0)
                    
                else:
                    a = axs.plot(quiet_time,quiet_b,linewidth=0.1,label=label2)
                    # axs.tick_params(direction='inout')
                    lgnd = axs.legend()
                    
                    for legobj in lgnd.legendHandles:
                        legobj.set_linewidth(2.0)
                        
                axs.set_xticks([b_time[0],b_time[round(len(b_time)/6)],b_time[round(2*len(b_time)/6)],\
                                   b_time[round(3*len(b_time)/6)],b_time[round(4*len(b_time)/6)],\
                                       b_time[round(5*len(b_time)/6)],b_time[len(b_time)-1]])
                axs.set_xticklabels(x_label_list,rotation = 0)
                axs.set_xlim([b_time[round(2.5*len(b_time)/6)],b_time[round(4.5*len(b_time)/6)]])
                # axs.xticks(rotation = 45)

            plt.subplots_adjust(wspace=0, hspace=0.012)
            plt.show()
            
        if mode in ['soloplot']:
            
            fis1 = plt.figure(figsize=(12,5))
            
            # times = [b_time,qual_r_time,qual_l_time,qual_t_time]
            # datas = [Br,qual_r,qual_l,qual_t]
            # labels = ['Br (nT)','random qual','long qual','time qual']
            
            times = [b_time,b_time]
            datas = [Br,Br]
            labels = ['nT','nT']
            names = ['Br','Br']
            label1 = r"Q Regions"\
                     "\n"  \
                     r"(de Wit)"
            label2 = "Quiescent Regions"
                     
            
            split_num = split_num
            
            tick_num = split_num+1
            
            min_rat = 1/split_num
            min_ind = round(min_rat*len(b_time))
            
            max_rat = 2.5/split_num
            max_ind = round(max_rat*len(b_time))
            
            all_inds = np.round(np.linspace(min_ind,max_ind,tick_num)).astype(int)

            # breakpoint()
            x_label_list = []
            x_axes = []
            for iii in range(tick_num):
                if iii == 0:
                    x_label_list.append(pys.time_string(b_time[all_inds[iii]],fmt='%Y-%m-%d/%H:%M:%S')) #first label
                    x_axes.append(b_time[all_inds[iii]])
                elif iii == tick_num-1:
                    x_label_list.append(pys.time_string(b_time[all_inds[iii]-1],fmt='%m-%d/%H:%M:%S')) #last label
                    x_axes.append(b_time[all_inds[iii]-1])
                else:
                    x_label_list.append(pys.time_string(b_time[all_inds[iii]],fmt='%m-%d/%H:%M:%S')) #middle labels
                    x_axes.append(b_time[all_inds[iii]])
                    
                # x_axes.append(b_time[all_inds[iii]])
                # print(iii)

            axs = fis1.add_subplot(1,1,1)


            axs.plot(b_time,Br,linewidth=0.1,label='Br')
            axs.set_ylabel('Br (nT)')
                
            axs.set_title('Encounter '+str(i))
            a = axs.plot(quiet_time,quiet_b,linewidth=0.1,label=label2)
            # axs.tick_params(axis='x',labelbottom=False,direction='in')
            lgnd = axs.legend()
            
            for legobj in lgnd.legendHandles:
                legobj.set_linewidth(2.0)
                
                    
            axs.set_xticks(x_axes)
            axs.set_xticklabels(x_label_list,rotation = 0)
            axs.set_xlim([b_time[min_ind],b_time[max_ind-1]])
            
            axs.axhline(y = 0, color = 'grey', linestyle = '--')
            
            # axs.xticks(rotation = 45)

            plt.subplots_adjust(wspace=0, hspace=0.012)
            plt.show()
        
        if mode in ['csv','both']:
        
            r_reg_check = (qual_r>thresh)
            t_reg_check = (qual_t>thresh)
            l_reg_check = (qual_l>thresh)
            
            r_regions = np.zeros(r_reg_check.shape)*np.nan
            i_ind = 1
            for k in range(len(r_regions)):
                
                if r_reg_check[k]==True:
                    r_regions[k] = i_ind
                
                if k!=0:
                    if r_reg_check[k]==False and r_reg_check[k-1]==True:
                        i_ind+=1
                        
            # breakpoint()
            start_dates = []
            end_dates = []
            durations = []
            for ii in range(int(np.nanmax(r_regions))):
                isl_where = np.where(r_regions==(ii+1))
                isl_where = isl_where[0]
                
                start_ind = isl_where[0]
                end_ind = isl_where[-1]
                
                start_time_flt = qual_r_time[start_ind]
                end_time_flt = qual_r_time[end_ind]
                
                duration = end_time_flt-start_time_flt
                
                start_date = pys.time_string(start_time_flt) 
                end_date = pys.time_string(end_time_flt)
                
                if duration>=30: #cut out short events, probably flukes

                    start_dates.append(start_date)
                    end_dates.append(end_date)
                    durations.append(duration)
                
            
            csv_arr = np.array([start_dates,end_dates,durations])
            
            csv_arr = np.transpose(csv_arr)
            #print(csv_arr.shape)
            
            # breakpoint()
            
            csv_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
            csv_savename = 'enc_'+str(i)+'_regions_raw.csv'
            csv_isdir = os.path.isdir(csv_savepath)
    
            if csv_isdir == False:
                os.makedirs(csv_savepath)
            
            csv_isfile = os.path.isfile(csv_savepath+csv_savename)
            
            # if csv_isfile == False:
            #     csv_header = np.array(['start dates','end dates', 'duration'])
            #     csv_header = np.transpose(csv_header)
            #     np.savetxt(csv_savepath+csv_savename,np.array([]),
            #                header='start dates, end dates, duration',
            #                fmt='%s',delimiter=',',comments='')
            
            with open(csv_savepath+csv_savename,"w") as file:
                # breakpoint()
                np.savetxt(file,csv_arr,header='start dates,end dates,duration',delimiter=',',fmt='%s',comments='')
        
        pyt.del_data()    
        
def quiescent_prop_enc(enc='all', plot=True):
    # if enc == 'all':
    #     enc_arr = enc_flt[0:16]
        
        
    if enc == 'all':
        enc = list(range(1,19))
        enc_arr = enc_flt[0:18]
        
        # enc = list(range(1,17))
        # enc_arr = enc_flt[0:16]
        
        title_mod = 'for All Encounters'
        
    elif enc == 'no 13':
        enc = list(range(2,13))
        enc.extend(range(14,17))
        # breakpoint()
        enc_arr = [enc_flt[i-1] for i in enc]
        # enc_arr = enc_flt[enc]
        title_mod = 'for All Encounters, no 1 or 13'    
    
    elif type(enc)==int:
        enci = enc
        encs = [enci]
        enc_arr = [enc_flt[enci]]
    else:
        enc_arr = [enc_flt[enc]]
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    # enc_ind = (enc-1)
    
    radial_bins = [[11.4,15],[15,25],[25,35],[35,45],[45,55],[55,65]]
    
    numpy_arr = []
    
    for enc in enc_arr:
        enc_str = pys.time_string(enc[0])
        enc_end = pys.time_string(enc[1])
        
        hpos_where = np.where((hpos_time_arr>enc[0])&(hpos_time_arr<enc[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<65)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        dfn = quiescent_prop(t0=t0,tf=tf)
        
        numpy_arr.append(dfn)
        
    durations = []
    event_num = 1
    for i in numpy_arr:
        for j in i:
            durations.append(j[2])
            event_num+=1
            
    data_time = np.nansum(durations)/3600 # units in hours
    
    print("Total Event Number: "+str(event_num))
    print("Total Event Duration: "+str(data_time))
    
    
    if plot:
        
        fig = plt.figure(figsize=(20,10))
        data = np.array(durations)/3600 #units in hours
        
        q_mean = np.nanmean(data)
        q_std = np.nanstd(data)
        q_median = np.nanmedian(data)
        q_quart = np.nanquantile(data,0.25)
        
        axs = fig.add_subplot(111)
        
        bins = np.linspace(0,4, 20)
        hist, _ = np.histogram(data, bins=bins)
        
        error1 = np.sqrt(hist)

        axs.bar(bins[:-1], hist, width=np.diff(bins), align='center', alpha=0.5, label='Quiescent Region Durations',edgecolor='black',color='tab:orange')
        # axs.bar(bins[:-1], hist2_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
        axs.errorbar(bins[:-1], hist, yerr=error1, fmt='none', color='k', capsize=3)
        # axs.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
        axs.text(0.4,0.65,"Number of Quiescent Regions: "+str(event_num),fontsize=32,transform=axs.transAxes)
        axs.text(0.4,0.55,"Quiescent Mean: "+str(round(q_mean,2)),fontsize=32,transform=axs.transAxes)
        axs.text(0.4,0.45,"Quiescent Median: "+str(round(q_median,2)),fontsize=32,transform=axs.transAxes)
        
        axs.vlines(q_mean,ymin=0,ymax=100,color='red',linestyle='dashed',label='Mean',linewidth=3)
        axs.vlines(q_median,ymin=0,ymax=100,color='blue',linestyle='dashed',label='Median',linewidth=3)
                
        axs.set_ylabel("Quiescent Region Counts",fontsize=30)
        axs.set_xlabel("Durations in Hours",fontsize=30)
        axs.set_title("Quiescent Region Duration Histogram",fontsize=30)
        # axs.set_ylim((0,100))
        axs.tick_params(axis='both', which='major', labelsize=24)
        
        plt.legend(fontsize=20)
        
        plt.show()
    
    breakpoint()
    
def quiescent_prop(t0='2018-11-05',tf='2018-11-06'): #quiescent region properties

    i_day = pys.time_float(t0)
    enci = 0
    for enc in enc_flt:          
        if (i_day>enc[0]) and (i_day<enc[1]): 
            encounter = enci+1 #determine which encounter today is in
            per_date = per_flt[enci]
            per_dist = per_dist_lst[enci]
        enci+=1
    
    csv_filename = 'enc_'+str(encounter)+'_regions_raw.csv'
    csv_path='/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'

    # psp.fields(trange=[t0,tf], datatype='mag_RTN_4_Sa_per_Cyc',username=fields_id,password=fields_pass, level='l2',last_version=True)
    # mag_data = pyt.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
    # breakpoint()
    # if encounter in [1,12]:

    #     psp.spc(trange=[t0,tf],level='L3',username=sweap_id,password=sweap_pass,last_version=True)
    #     vel_data = pyt.get_data('psp_spc_vp_fit_RTN')
    
    # else:
    #     psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)
    #     vel_data = pyt.get_data('psp_spi_VEL_RTN_SUN')
    
    # psp.fields(trange=[t0,tf], datatype='ephem_spp_hg', level='l1',username=fields_id,password=fields_pass,last_version=True)
    # pos_data = pyt.get_data('position') #retrieve PSP position data from tplot variable
    
    df = pd.read_csv(csv_path+csv_filename)
    dfn = df.to_numpy()
    
    return dfn
    
def quiescent_calc(t0='2018-11-05',tf='2018-11-06',pickle=False): #calculates z, as well as other parameters.

    Rs = 6.957e5 #solar radius in km
    Rs_in_m = Rs*10**3
    w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec
    
    #mag_file_path = CONFIG['local_data_dir']+'/fields/l2/mag_RTN/'+t0[0:4]+'/'+t0[5:7]+'/'
    mag_file_path = CONFIG['local_data_dir']+'/fields/l2/mag_RTN_4_Sa_per_Cyc/'+t0[0:4]+'/'+t0[5:7]+'/'
    spc_file_path = CONFIG['local_data_dir']+'/sweap/spc/L3/'+t0[0:4]+'/'+t0[5:7]+'/'
    spi_file_path = CONFIG['local_data_dir']+'/sweap/spi/L3/spi_sf00/'+t0[0:4]+'/'+t0[5:7]+'/'
    ephem_file_path = CONFIG['local_data_dir']+'/fields/l1/ephem_spp_hg/'+t0[0:4]+'/'+t0[5:7]+'/'
    
    i_day = pys.time_float(t0)
    enci = 0
    for enc in enc_flt:          
        if (i_day>enc[0]) and (i_day<enc[1]): 
            encounter = enci+1 #determine which encounter today is in
            per_date = per_flt[enci]
            per_dist = per_dist_lst[enci]
        enci+=1
    
    ephem_file_name = pys.time_string(i_day,fmt='spp_fld_l1_ephem_spp_hg_%Y%m%d_v01.cdf')
    
    isdir_mag = os.path.isdir(mag_file_path)
    isdir_vel_spc = os.path.isdir(spc_file_path)
    isdir_vel_spi = os.path.isdir(spi_file_path)
    isdir_eph = os.path.isdir(ephem_file_path)
    
    if isdir_mag == False:
        os.makedirs(mag_file_path)
    if isdir_vel_spc == False:
        os.makedirs(spc_file_path)
    if isdir_vel_spi == False:
        os.makedirs(spi_file_path)
    if isdir_eph == False:
        os.makedirs(ephem_file_path)
    
    """ 
    The following could simply be done with psp.fields(arguments), but certain servers pay attention to
    what IPs are hogging data. Namely, the public NASA servers. If I pull too much information for too long
    (like looping through all the Parker Solar Probe data)
    then the servers will seriously slow down how quickly I can pull data after a while. 
    
    Here I'm pulling straight from the secure servers so that shouldnt
    be an issue but its good practice to check whether you have the data already before unnecessarily making a
    a request of the server.
    
    Incidentally, this will make it so that you can run the code without using the internet (assuming you already have the data onhand).
    
    01-31-24
    
    I think I broke the code refered to in this above comment. I'll fix it eventually.
    
    """

    
    mag_file_name = pys.time_string(i_day,fmt='psp_fld_l2_mag_RTN_4_Sa_per_Cyc_%Y%m%d_v02.cdf')
    # mag_file_name = pys.time_string(i_day,fmt='psp_fld_l2_mag_RTN_%Y%m%d_v02.cdf')
    # mag_file_name = pys.time_string(i_day,fmt='psp_fld_l2_mag_RTN_1min_%Y%m%d_v02.cdf')
    mag_isfile = os.path.isfile(mag_file_path+mag_file_name)     
    mag_infile = mag_file_path+mag_file_name
    

    psp.fields(trange=[t0,tf], datatype='mag_RTN_4_Sa_per_Cyc', level='l2',last_version=True,username=fields_id,password=fields_pass)
    # psp.fields(trange=[t0,tf], datatype='mag_RTN', level='l2',last_version=True)
    
    mag_data = pyt.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
    # mag_data = pyt.get_data('psp_fld_l2_mag_RTN')
    
    mag_time_arr = mag_data[0]
    mag_data_arr = mag_data[1]
    
    b_mag = np.sqrt(mag_data_arr[:,0]**2+mag_data_arr[:,1]**2+mag_data_arr[:,2]**2)
    
    br_meas = -mag_data_arr[:,0]/b_mag  #define the magnetic field to be opposite what it is in RTN to be consistent with DeWit 2020
    bt_meas = -mag_data_arr[:,1]/b_mag
    bn_meas = -mag_data_arr[:,2]/b_mag

    mag_len = len(b_mag)
    
    br_where = np.where(br_meas < 0)
    
    ephem_isfile = os.path.isfile(ephem_file_path+ephem_file_name)
    
    spc_file_name = pys.time_string(i_day,fmt='psp_swp_spc_l3i_%Y%m%d_v01.cdf')
    spc_isfile = os.path.isfile(spc_file_path+spc_file_name)  
    spc_infile = spc_file_path+spc_file_name

    
    if encounter in [1]:

        psp.spc(trange=[t0,tf], level='L3',username=sweap_id,password=sweap_pass,last_version=True)
        vel_data = pyt.get_data('psp_spc_vp_fit_RTN')
        
        vel_time_arr = vel_data[0]
        vel_data_arr = vel_data[1]
        
        v_r_data = vel_data_arr[:,0]
        
    else:
        
        psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)
        vel_data_spi = pyt.get_data('psp_spi_VEL_RTN_SUN')
        vel_time_arr_spi = vel_data_spi[0]
        vel_data_arr_spi = vel_data_spi[1]
        
        spi_vr = vel_data_arr_spi[:,0]
        
        
        
        psp.spc(trange=[t0,tf], level='L3',username=sweap_id,password=sweap_pass,last_version=True)
        
        # breakpoint()
        vel_data_spc = pyt.get_data('psp_spc_vp_fit_RTN')
        
        if vel_data_spc == None:
            vel_data_spc = pyt.get_data('spp_spc_vp_fit_RTN')
        
        vel_time_arr_spc = vel_data_spc[0]
        vel_data_arr_spc = vel_data_spc[1]
        
        spc_vr = vel_data_arr_spc[:,0]
        
        #------------------ downsample and clean SPC data ------------------#
        
        # spc_vr_tmp = []
        # for i in range(len(vel_time_arr_spi)):
        #     spi_window_start = vel_time_arr_spi[i]-30
        #     spi_window_end = vel_time_arr_spi[i]+30
        #     spc_where = np.where((vel_time_arr_spc>spi_window_start) & (vel_time_arr_spc<spi_window_end))
        #     spc_where = spc_where[0]
        #     nan_total = sum(np.isnan(spc_vr[spc_where]))
        #     if nan_total > len(spc_where)/2:
        #         spc_vr_tmp.append(np.nan)
        #     else:
        #         spc_vr_tmp.append(np.nanmedian(spc_vr[spc_where]))
        #     if i%2500 == 0:
        #         print(round(i/len(vel_time_arr_spi)*100,2),"%")
                
        # spc_time_down = vel_time_arr_spi
        # spc_vr_down = np.array(spc_vr_tmp)
        
        # spi_time_tmp = np.emptylike(spi_time)
        # spi_strs = spi_time-30
        # spi_ends = spi_time+30
        
        spc_vr_clean = sliding_median(spc_vr,275) #about one minute long windows at max cadence

        interpolating_func = interp1d(vel_time_arr_spc, spc_vr_clean, kind='linear', fill_value='extrapolate')
    
        # Create a new array of time values with fixed cadence
        spc_time_down = np.arange(vel_time_arr_spc[0], vel_time_arr_spc[-1], 6)
    
        # Interpolate arr_two to the new time values
        spc_vr_down = interpolating_func(spc_time_down)
    
        # return new_arr_one, new_arr_two

        #-----------attempt to make the above for loop more efficient--------#
        
        # Define the window size in time (delta_t), 3 minutes or 180 seconds
        # delta_t = 180  # Adjust this as needed
        
        # # breakpoint()
        # # print('ey1')
        
        # spc_time_down = np.array([])
        # spc_vr_down = np.array([])
        
        # n_arrays = 100
        # # break data into 100 chunks so that can be looped through. 
        # # This saves time and RAM.
        
        # spc_time_segs, spc_vr_segs = split_data_evenly_in_time(vel_time_arr_spc, spc_vr, n_arrays)
        # spi_time_segs, spi_vr_segs = split_data_evenly_in_time(vel_time_arr_spi, spi_vr, n_arrays)
        
        # for i in range(n_arrays):
            
        #     # split_arrays = np.array_split(array1, n_arrays)
        #     spc_split_time_arr = spc_time_segs[i]
        #     spc_split_data_arr = spc_vr_segs[i]
        #     spi_split_time_arr = spi_time_segs[i]
                          
        #     time_diff = spc_split_time_arr[:, None] - spi_split_time_arr

        #     # Use boolean indexing to create a 2D mask
        #     window_mask = (time_diff >= -delta_t/2) & (time_diff <= delta_t/2)
            
        #     window_values = np.where(window_mask.T, spc_split_data_arr, np.nan)
        #     spc_medians = np.nanmedian(window_values, axis=1)
            
        #     # Compute the median along the second axis (axis=1) ignoring NaN values
        #     spc_time_down = np.append(spc_time_down,spi_split_time_arr)
        #     spc_vr_down = np.append(spc_vr_down,spc_medians)
        #     # print(i)
        
        #--------------------------------------------------------------------#
        
        span_check_savename = 'SPAN_SPC_QTN_flags'+'_enc_'+str(encounter)+'.cdf'
        span_check_savepath = '/Users/besh2109/Desktop/SPAN Checks/'
        
        pyt.tplot_restore(span_check_savepath+span_check_savename)
    
        SPAN_QTN_flag = pyt.get_data('SPAN_qual_flag')
        SPAN_flag_time = SPAN_QTN_flag[0]
        SPAN_flag = SPAN_QTN_flag[1]
        
        SPC_QTN_flag = pyt.get_data('SPC_qual_flag')
        SPC_flag_time = SPC_QTN_flag[0]
        SPC_flag = SPC_QTN_flag[1]
        
        flag_zero_group = group_zeros(SPAN_flag)
        
        bad_span_times = []
        vel_time_construct = np.array([],dtype=float)
        vel_r_data_construct = np.array([],dtype=float)
        
        # breakpoint()
        bins = np.array([])
        for i in range(len(flag_zero_group)):
            
            t0s = SPAN_flag_time[flag_zero_group[i][0]]
            tfs = SPAN_flag_time[flag_zero_group[i][1]]
            # edge_times.append((t0,tf))
            bins = np.append(bins,np.array([t0s,tfs]))
            # bad_span_times.append((t0s,tfs))
        good_bad = 0 #loop through times when quality flag is good and when its bad. start with bad.
        # bad = 0
        for i in range(1,len(bins)):
            

            if i==1:
                t0i = 10
                tfi = bins[i]
            elif i==len(bins)-1:
                # tf0 = bins[i]
                tfi = np.inf  
            else:
                t0i = bins[i-1]
                tfi = bins[i]
                
            if good_bad == 0:
                
                spc_where = np.where((spc_time_down>t0i)&(spc_time_down<tfi))
                spc_where = spc_where[0]
                
                vel_time_construct = np.append(vel_time_construct,spc_time_down[spc_where])
                vel_r_data_construct = np.append(vel_r_data_construct,spc_vr_down[spc_where])
                
                good_bad = 1 #alternate between good and bad times. Should start with bad.
                # breakpoint()
            elif good_bad ==1:
                
                spi_where = np.where((vel_time_arr_spi>t0i)&(vel_time_arr_spi<tfi))
                spi_where = spi_where[0]
                
                vel_time_construct = np.append(vel_time_construct,vel_time_arr_spi[spi_where])
                vel_r_data_construct = np.append(vel_r_data_construct,spi_vr[spi_where])
                
                good_bad = 0
            
        # fig = plt.figure(figsize=(30,18))

        # axs = fig.add_subplot(111)
        # axs.plot(vel_time_construct,vel_r_data_construct+400, color='tab:orange',linewidth=1)
        # axs.plot(vel_time_arr_spi,spi_vr+200, color='tab:blue',linewidth=1)
        # axs.plot(spc_time_down,spc_vr_down, color='green',linewidth=1)

        # axs.tick_params(axis='both', which='major', labelsize=22)

        # plt.show()
        # breakpoint()
        
        #lengthen the SPAN flag to the length of the SPAN data used here. Assume flag is zero everywhere outside the original quality flag.
        
        # SPAN_flag_new = np.zeros_like(spi_vr,dtype=float)
        # start_where = np.where(vel_time_arr_spi==SPAN_flag_time[0])
        # start_where = start_where[0][0]
        # SPAN_flag_new[start_where:len(SPAN_flag_time)+start_where]=SPAN_flag
        
        #start with SPAN-I as the default velocity

        # vel_r_data_construct = spi_vr
        
        #find all the places where SPAN-I has low quality flag and replace with SPC
        # span_0_where = np.where(SPAN_flag_new==0)s
        # span_0_where = span_0_where[0]
        # vel_r_data_construct[span_0_where] = spc_vr[span_0_where]
        
        #find all the places where SPC has missing data and full back in with SPAN-I because we have no choice
        nanwhere = np.where(np.isnan(vel_r_data_construct))
        nanwhere = nanwhere[0]
        nan_bool = 1-np.isnan(vel_r_data_construct)*1
        
        nan_groups = group_zeros(nan_bool)
        nan_times = vel_time_construct[nan_groups]
        
        for i in range(len(nan_times)):
            nan_t0 = nan_times[i,0]
            nan_tf = nan_times[i,1]
            
            vel_where = np.where((vel_time_construct>nan_t0)&(vel_time_construct<nan_tf))
            vel_where = vel_where[0]

            spi_nan_where = np.where((vel_time_arr_spi>nan_t0)&(vel_time_arr_spi<nan_tf))
            spi_nan_where = spi_nan_where[0]
            
            spc_nan_where = np.where((spc_time_down>nan_t0)&(spc_time_down<nan_tf))
            spc_nan_where = spc_nan_where[0]
            
            vel_time_construct = np.delete(vel_time_construct,vel_where)
            vel_r_data_construct = np.delete(vel_r_data_construct,vel_where)
            
            vel_time_construct = np.append(vel_time_construct,vel_time_arr_spi[spi_nan_where])
            vel_r_data_construct = np.append(vel_r_data_construct,spi_vr[spi_nan_where])
        
        sorted_indices = np.argsort(vel_time_construct)
        
        vel_time_construct = vel_time_construct[sorted_indices]
        vel_r_data_construct = vel_r_data_construct[sorted_indices]
        
        # breakpoint()
        # vel_r_data_construct[nanwhere] = spi_vr[nanwhere]
        
        vel_time_arr = vel_time_construct
        v_r_data = vel_r_data_construct
        
    psp.fields(trange=[t0,tf], datatype='ephem_spp_hg', level='l1',last_version=True,username=fields_id,password=fields_pass)
    pos_data = pyt.get_data('position') #retrieve PSP position data from tplot variable

    pos_time_arr = pos_data[0]
    pos_data_arr = pos_data[1]
    
    pos_time_arr = pos_time_arr[0:len(pos_time_arr)]
    
    x = pos_data_arr[:,0]
    y = pos_data_arr[:,1]
    z = pos_data_arr[:,2]
    
    R = np.sqrt(pos_data_arr[:,0]**2+pos_data_arr[:,1]**2+pos_data_arr[:,2]**2)-Rs #parker spiral source height
    R_true = np.sqrt(pos_data_arr[:,0]**2+pos_data_arr[:,1]**2+pos_data_arr[:,2]**2) #height from center of sun
    
    # carr_long_in = np.arctan(y/x)*(180/np.pi)
    carr_long_in = 2*np.arctan(y/(np.sqrt(x**2+y**2)+x))*(180/np.pi)
    carr_lat_in = np.arcsin(z/R_true)*(180/np.pi)
    
    # """
    # Useful to fill in NaNs in SPC data in order to make parker calculation.
    # Slice day into 102 slices to take the median of to make a median array.
    # Fill in nans with values from this median array.
    # This ought to be okay because we're only looking for a rough estimate.
    # """
    # v_mean_arr = np.ones(len(v_r_data))
    # v_len = len(v_r_data)
    # v_bin_size = int(v_len/102)
    # ii=0
    # while ii<103:
    #     v_nan_check = v_r_data[ii*v_bin_size:(ii+1)*v_bin_size]
    #     slice_mean = np.nanmedian(v_nan_check)
    #     v_mean_arr[ii*v_bin_size:(ii+1)*v_bin_size]=slice_mean
    #     ii+=1
    
    # if v_mean_arr[-1]==1:
    #     v_mean_arr[-1] = v_mean_arr[-2]
        
    # v_r_data[np.isnan(v_r_data)]=v_mean_arr[np.isnan(v_r_data)]
    
    
    # breakpoint()
    
    vr_indices = assign_closest_index(mag_time_arr,vel_time_arr)
    # breakpoint()
    
    vr_values_check = v_r_data[vr_indices]
    
    
    # spc_time_down = np.array([])
    # vr_values_check = np.array([])
    
    # n_arrays = 40
    # # break data into 100 chunks so that can be looped through. 
    # # This saves time and RAM.
    
    # mag_time_segs, b_mag_seg = split_data_evenly_in_time(mag_time_arr, b_mag, n_arrays)
    # vel_time_segs, v_r_segs = split_data_evenly_in_time(vel_time_arr, v_r_data, n_arrays)
    
    # # mag_time_split = np.array_split(mag_time_arr, n_arrays)
    # # v_time_split = np.array_split(vel_time_arr, n_arrays)
    
    
    # for i in range(n_arrays):
        
    #     # split_arrays = np.array_split(array1, n_arrays)
    #     mag_split_time_arr = mag_time_segs[i]
    #     v_split_time_arr = vel_time_segs[i]
                      
    #     index_vector = assign_closest_index_vec(mag_split_time_arr,v_split_time_arr)
        
    #     print(i)
    #     # breakpoint()
        
    #     # Use boolean indexing to create a 2D mask
    #     # window_mask = (time_diff >= -delta_t/2) & (time_diff <= delta_t/2)
        
    #     # window_values = np.where(window_mask.T, spc_split_data_arr, np.nan)
    #     # spc_medians = np.nanmedian(window_values, axis=1)
        
    #     # Compute the median along the second axis (axis=1) ignoring NaN values
    #     # spc_time_down = np.append(spc_time_down,spi_split_time_arr)
    #     vr_values_check = np.append(vr_values_check,v_r_data[index_vector])
    
    
    # v_r_xp = np.arange(len(v_r_data))
    # v_r_x = np.arange(mag_len)/mag_len*(len(v_r_data)-1)
    # v_r_i = np.interp(v_r_x,v_r_xp,v_r_data)
    
    # v_r=v_r_i
    
    v_r = vr_values_check
    
    sin_theta = pos_data_arr[:,2]/R
    sin_theta_xp = np.arange(len(sin_theta))
    sin_theta_x = np.arange(mag_len)/mag_len*(len(sin_theta)-1)
    sin_theta_i = np.interp(sin_theta_x,sin_theta_xp,sin_theta)
    
    R_xp = np.arange(len(R))
    R_x = np.arange(mag_len)/mag_len*(len(R)-1)
    R_i = np.interp(R_x,R_xp,R)
    
    carr_xp = np.arange(len(carr_long_in))
    carr_x = np.arange(mag_len)/mag_len*(len(carr_long_in)-1)
    carr_long_i = np.interp(carr_x,carr_xp,carr_long_in)
    carr_lat_i = np.interp(carr_x,carr_xp,carr_lat_in)
    carr_time_i = np.interp(carr_x,carr_xp,pos_time_arr)
    
    carr_dif = np.diff(carr_long_i)
    time_dif = np.diff(carr_time_i)
    
    carr_dif_xp = np.arange(len(carr_dif))
    carr_dif_x = np.arange(mag_len)/mag_len*(len(carr_dif)-1)
    carr_dif_i = np.interp(carr_dif_x,carr_dif_xp,carr_dif)
    time_dif_i = np.interp(carr_dif_x,carr_dif_xp,time_dif)
    
    # breakpoint()
    carr_time = carr_time_i
    carr_lat = carr_lat_i
    carr_long = carr_long_i
    
    parker_x = R_i*w/v_r*sin_theta_i
    
    gamma = np.arctan(parker_x)
    
    br_park = np.cos(gamma)
    bt_park = np.sin(gamma)
    bn_park = np.zeros(len(mag_time_arr))
    
    # br_park[br_where] = -br_park[br_where]
    
    # cos_alpha = np.abs(br_meas*br_park+bt_meas*bt_park+bn_meas*bn_park)
    cos_alpha = br_meas*br_park+bt_meas*bt_park+bn_meas*bn_park
    
    z = (1/2)*(1-cos_alpha)
    # z = (1-cos_alpha)
    z_where = np.where((z>0.05)&(z<0.95))
    # z_where = np.where(z>0.025)
    quiet_z = np.array(z)
    quiet_z[z_where]=np.nan
    quiet_b = np.array(-br_meas*b_mag)
    quiet_b[z_where]=np.nan
    
    z_time_arr = mag_time_arr
    # breakpoint()
    if pickle:
        
        pickle_path = '/Users/besh2109/Desktop/z_pickle/'
        pickle_name = ''
        
    
    return {'z_time':z_time_arr,'z':z,'quiet_z':quiet_z,\
            'b_time':mag_time_arr,'Br':-br_meas*b_mag,'Bt':-bt_meas*b_mag,'Bn':-bn_meas*b_mag,'quiet_B':quiet_b,'B_mag':b_mag, \
            'carr_time':carr_time,'carr_long':carr_long,'carr_lat':carr_lat,'carr_dif':carr_dif_i,'time_dif':time_dif_i}
        
def quiescent_calc_enc(enc_num,enc_radius=60,save=False):
    # enc_num = enc_start
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    #print(hpos_path)
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    #print(pyt.tplot_names())
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    enc_ind = (enc_num-1)
    
    enc = enc_flt[enc_ind]
    
    enc_str = pys.time_string(enc[0])
    enc_end = pys.time_string(enc[1])
    
    hpos_where = np.where((hpos_time_arr>enc[0])&(hpos_time_arr<enc[1]))
    hpos_where = hpos_where[0]
    
    hpos_time = hpos_time_arr[hpos_where]
    
    hposx_data = hpos_data_arr[hpos_where,0]
    hposy_data = hpos_data_arr[hpos_where,1]
    hposz_data = hpos_data_arr[hpos_where,2]
    
    R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
    
    R_where = np.where(R<enc_radius)
    R_where = R_where[0]
    
    time_select = hpos_time[R_where]
    t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
    tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
    
    t0 = pys.time_string(time_select[0])
    tf = pys.time_string(time_select[-1])
    
    title = 'Encounter '+str(enc_num)+': '+t0p[0:20]+' to '+tfp[0:20]
    
    varis = quiescent_calc(t0=t0,tf=tf)

    if save:
        
        z_time = varis['z_time']
        z = varis['z']
        quiet_z = varis['quiet_z']
        b_time = varis['b_time']
        Br = varis['Br']
        quiet_B = varis['quiet_B']
        B_mag = varis['B_mag']
        
        carr_time = varis['carr_time']
        carr_long = varis['carr_long']
        carr_lat = varis['carr_lat']
        carr_dif = varis['carr_dif']
        
        save_path = '/Users/besh2109/Desktop/z_save/'
        
        pickle_tag = 'pickle/'
        numpy_tag = 'numpy/'
        tplot_tag = 'tplot/'
        
        numpy_save_arr = np.array([z_time,z])
        
        pickle_name = 'z_pickle_enc_'+str(enc_num)+'.pkl'
        
        # with open(save_path+pickle_tag+pickle_name,'wb') as f:
        #     pkl.dump(numpy_save_arr,f)
        
        
        numpy_name = 'z_numpy_enc_'+str(enc_num)
        
        
        np.save(save_path+numpy_tag+numpy_name,numpy_save_arr)
        
        
        
        tplot_name = 'z_tplot_enc'+str(enc_num)+'.cdf'
        

        
        pyt.store_data("z", data={'x':z_time, 'y':z})
        pyt.store_data("quiet_z", data={'x':z_time, 'y':quiet_z})
        pyt.store_data("Br", data={'x':b_time, 'y':Br})
        pyt.store_data("B_mag", data={'x':b_time, 'y':B_mag})
        pyt.store_data("quiet_B", data={'x':b_time, 'y':quiet_B})
        
        pyt.store_data("carr_lon", data={'x':carr_time, 'y':carr_long})
        pyt.store_data("carr_lat", data={'x':carr_time, 'y':carr_lat})
        pyt.store_data("carr_dif",data={'x':carr_time, 'y':carr_dif})
        
        pyt_save_list = ['z',"quiet_z","Br","B_mag","quiet_B","carr_lon","carr_lat","carr_dif"]
        
        pyt.tplot_save(pyt_save_list,save_path+tplot_tag+tplot_name)
        
    return varis
       
def quiescent_plot(t0='2018-11-04',tf='2018-11-05',enc=None,enc_radius=75,title=None,save=False,other_comp=False,enc_save=False): #plots z and other parameters.

    if enc != None:
        
        enc_num = enc
        
        hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
        #print(hpos_path)
        pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
        #print(pyt.tplot_names())
        hpos = pyt.get_data('position')
        
        hpos_time_arr = hpos[0]
        hpos_data_arr = hpos[1]
        
        enc_ind = (enc_num-1)
        
        enc = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc[0])
        enc_end = pys.time_string(enc[1])
        
        hpos_where = np.where((hpos_time_arr>enc[0])&(hpos_time_arr<enc[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<enc_radius)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        t0_flt = pys.time_float(t0)
        tf_flt = pys.time_float(tf)

    else:
        t0_flt = pys.time_float(t0)
        tf_flt = pys.time_float(tf)
        
        enci = 0
        for enc_check in enc_flt:          
            if (t0_flt>enc_check[0]) and (t0_flt<enc_check[1]): 
                enc_num = enci+1 #determine which encounter today is in
                # per_date = per_flt[enci]
                # per_dist = per_dist_lst[enci]
            enci+=1

    z_save_path = '/Users/besh2109/Desktop/z_save/z_v03/tplot/'
    z_save_name = 'z_tplot_enc'+str(enc_num)+'.cdf'
    
    filecheck = os.path.isfile(z_save_path+z_save_name)
    
    # breakpoint()
    
    if filecheck:
        pyt.tplot_restore(z_save_path+z_save_name)
        
        z_data = pyt.get_data('z')
        z_time = z_data[0]
        z = z_data[1]
        
        quiet_z_data = pyt.get_data('quiet_z')
        quiet_z = quiet_z_data[1]
        
        b_data = pyt.get_data('Br')
        b_time = b_data[0]
        Br = b_data[1]
        
        quiet_B_data = pyt.get_data('quiet_B')
        quiet_B = quiet_B_data[0]
        
        B_mag_data = pyt.get_data('B_mag')
        B_mag_time = B_mag_data[0]
        B_mag = B_mag_data[1]
        
        carr_data = pyt.get_data('carr_lon')
        carr_time = carr_data[0]
        carr_long = carr_data[1]
        
        carr_data = pyt.get_data('carr_lat')
        carr_lat = carr_data[1]
        
        carr_data = pyt.get_data('carr_dif')
        carr_dif = carr_data[1]
        
    else:
        varis = quiescent_calc(t0=t0,tf=tf)

        z_time = varis['z_time']
        z = varis['z']
        # quiet_z = varis['quiet_z']
        b_time = varis['b_time']
        Br = varis['Br']
        quiet_B = varis['quiet_B']
        B_mag = varis['B_mag']
        
        carr_time = varis['carr_time']
        carr_long = varis['carr_long']
        carr_lat = varis['carr_lat']
        carr_dif = varis['carr_dif']
    
    zlen = len(z_time)

    z_where = np.where((z_time>=t0_flt)&(z_time<tf_flt))

    z_time = z_time[z_where]
    z = z[z_where]  
    # quiet_z = quiet_z[z_where]
    
    b_where = np.where((b_time>=t0_flt)&(b_time<tf_flt))

    b_time = b_time[b_where]
    Br = Br[b_where]
    # Bt = Bt[b_where]
    # Bn = Bn[b_where]
    quiet_B = quiet_B[b_where]
    
    if filecheck:
        b_where = np.where((B_mag_time>=t0_flt)&(B_mag_time<tf_flt))
        B_mag_time = B_mag_time[b_where]
    else:
        B_mag_time = b_time
    # B_mag_time = B_mag_time[b_where]
    B_mag = B_mag[b_where]
    
    carr_where = np.where((carr_time>=t0_flt)&(carr_time<tf_flt))
    carr_time = carr_time[carr_where]
    carr_long = carr_long[carr_where]
     
    
    z_t_string = pys.time_string(z_time)
    
    z_time_dates = pd.to_datetime(z_t_string)
    z_time_dates = z_time_dates.to_numpy()
    
    b_t_string = pys.time_string(B_mag_time)
    B_time_dates = pd.to_datetime(b_t_string)
    B_time_dates = B_time_dates.to_numpy()
    
    # breakpoint()
    
    # c_t_string = pys.time_string(carr_time)
    
    # carr_time_dates = pd.to_datetime(c_t_string)
    # carr_time_dates = carr_time_dates.to_numpy()
    
    # # carr_time_dates = [dateutil.parser.parse(t) for t in pys.time_string(carr_time)]    
    
    # i_max = len(carr_time)
    # wndw_bars = np.array(carr_time[0])
    # carr_int = 0
    # for i in range(i_max):
        
    #     carr_int+=carr_dif[i]
        
    #     if np.abs(carr_int) > 0.82: #0.082 for granulation scale, 0.82 is supergranulation scale
    #         wndw_bars = np.append(wndw_bars,carr_time[i])
    #         carr_int = 0
    #     if i==i_max:
    #         wndw_bars = np.append(wndw_bars,carr_time[i])

    #--------------#
    # i_max = len(z_time)
    # wndw_bars = np.array(z_time[0])
    
    # bins = np.array([0])
    # bins = np.append(bins,random.sample(range(i_max),499))
    # bins = np.append(bins,np.array(i_max-1))
    
    # # breakpoint()
    
    # wndw_bars = z_time[bins]
    
    #--------------#
    
    if tf==None:
        tf = pys.time_string(pys.time_float(t0)+86400)

    # if enc != None:    
    qregion_savename = 'enc_'+str(enc_num)+'_regions_raw.csv'
    # else:
        # qregion_savename = t0+'_'+tf+'_regions_raw.csv'
        
        # t0float = pys.time_float(t0)
        # enci = 1
        # for i in enc_flt:
        #     if t0float>i[0] and t0float<i[1]:
        #         enc_num = enci
        #     enci+=1
        # if tf==None:
        #     tffloat = pys.time_float(t0)+86400
        # else:
        #     tffloat=pys.time_float(tf)

    qregion_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
    
    # breakpoint()

    #---------------------------- Read in quiescent regions --------------------------------#

    qregion_df = pd.read_csv(qregion_savepath+qregion_savename)
    qregion_array = qregion_df.to_numpy()
    
    if enc != None:    
        regions_arr = qregion_array
        
    else:
        
        pre_flt = qregion_array[:,:2]
        
        qregion_lst = []
        for j in pre_flt:
            qregion_lst.append(pys.time_float(j))
            # print(j)
        qregion_flt = np.array(qregion_lst) 
        reg_where = np.where((qregion_flt[:,0]>t0_flt)&(qregion_flt[:,1]<tf_flt))
        
        regions_arr = qregion_array[reg_where,:]
        regions_arr = regions_arr[0]

    #---------------- find footpoints associated with quiescent regions --------------------#
    
    # breakpoint()
    
    region_time = np.array([])
    region_Br = np.array([])
    region_Bt = np.array([])
    region_Bn = np.array([])
    
    region_z = np.array([])
    
    for k in regions_arr:
        # breakpoint()
        pre_flt = k[:2]
        reg_flt = pys.time_float(pre_flt)
        point_where = np.where((b_time>reg_flt[0])&(b_time<reg_flt[1]))
        point_where = point_where[0]
        
        region_time_tmp = b_time[point_where]
        region_Br_tmp =  Br[point_where]
        # region_Bt_tmp = Bt[point_where]
        # region_Bn_tmp = Bn[point_where]
        region_z_tmp = z[point_where]

        region_time = np.append(region_time,region_time_tmp)
        region_Br = np.append(region_Br,region_Br_tmp)
        # region_Bt = np.append(region_Bt,region_Bt_tmp)
        # region_Bn = np.append(region_Bn,region_Bn_tmp)
        region_z = np.append(region_z,region_z_tmp)
        
        # breakpoint()
        
        # if k[2] > 1800.0:
            
        #     long_region_time_tmp = b_time[point_where]
        #     long_region_foot_lon_tmp =  sol_lon[point_where]
        #     long_region_foot_lat_tmp = sol_lat[point_where]
            
        #     long_region_time = np.append(long_region_time,long_region_time_tmp)
        #     long_region_foot_lon = np.append(long_region_foot_lon,long_region_foot_lon_tmp)
        #     long_region_foot_lat = np.append(long_region_foot_lat,long_region_foot_lat_tmp)
        
    r_t_string = pys.time_string(region_time)
    
    region_time_dates = pd.to_datetime(r_t_string)
    region_time_dates = region_time_dates.to_numpy()
    
    breakpoint()
    
    #--------------#

    # fis1 = plt.figure(figsize=(30,18))
    # fis1 = plt.figure(figsize=(25,10))
    fis1 = plt.figure(figsize=(25,7.5))
    
    time1 = [z_time_dates,z_time_dates]
    time2 = [region_time_dates,region_time_dates]
    
    # time1 = [z_time,z_time]
    # time2 = [region_time,region_time]
    
    data1 = [Br,z]
    # data2 = [quiet_B, quiet_z]
    data2 = [region_Br, region_z]
    
    label1 = ['Br','z']
    label2 = ['Br (Quiescent)','z (Quiescent)']
    # label2 = ['Br (z<0.05)','z<0.05']
    
    y_labs = ['Magnetic Field (nT)','z-parameter']
   
    # data1.reverse()
    # data2.reverse()
    # label1.reverse()
    # label2.reverse()
    # y_labs.reverse()
   
    # for ii in range(2):
    for ii in range(1):
        
        # axs = fis1.add_subplot(2,1,ii+1)
        axs = fis1.add_subplot(1,1,ii+1)
        axs.plot(time1[ii],data1[ii], color='tab:blue',linewidth=0.22,label=label1[ii],zorder=1)
        # axs.plot(time1[ii],data1[ii], color='tab:blue',linewidth=2,label=label1[ii],zorder=1)
        # axs.plot(time2[ii],data2[ii], color='tab:orange',linewidth=0.18,label=label2[ii],zorder=3)
        axs.tick_params(axis='y', which='major', labelsize=22)
        axs.tick_params(axis='x', which='major', labelsize=24)
        axs.xaxis.get_offset_text().set_size(24)
        if ii==0:
            axs.plot(B_time_dates,B_mag,color='black',linewidth=0.22,label='|B|',zorder=2)
            
            # axs.tick_params('x', labelbottom=False)
            # axs.plot(carr_time,carr_long,color='purple',linewidth=1,label='Carrington Long',zorder=5)
            
            # if other_comp:
            #     axs.plot(z_time_dates,Bt,color='red',linewidth=0.12,label='Bt',zorder=3)
            #     axs.plot(z_time_dates,Bn,color='green',linewidth=0.12,label='Bn',zorder=3)
            
            
            # breakpoint()
            if title==None:
                if enc != None:
                    axs.set_title('Encounter '+str(enc_num)+' Quiescent Regions',fontsize=30)
                else:
                    # axs.set_title(pys.time_string(z_time[0])[0:19]+' to '+pys.time_string(z_time[-1]+1)[0:19],fontsize=30)
                    axs.set_title('Encounter '+str(enc_num)+' Quiescent Regions',fontsize=30)
            else:
                axs.set_title(title,fontsize=36)
                
            # axs.vlines(wndw_bars,-120,120, color='green',zorder=4,linewidth=0.5)
            # axs.set_ylim([16,17])
        
        # axs.set_xticks([z_time[0],z_time[round(len(z_time)/3)],\
        #                z_time[round(2*len(z_time)/3)],z_time[(len(z_time)-1)]])
        
        # x_label_list = [pys.time_string(z_time[0],fmt='%Y-%m-%d/%H:%M:%S'),\
        #                 pys.time_string(z_time[round(len(z_time)/3)],fmt='%Y-%m-%d/%H:%M:%S'), \
        #                 pys.time_string(z_time[round(2*len(z_time)/3)],fmt='%Y-%m-%d/%H:%M:%S'), \
        #                 pys.time_string(z_time[len(z_time)-1],fmt='%Y-%m-%d/%H:%M:%S')]
        
        # x_label_list = [pys.time_string(z_time[0],fmt='%Y-%m-%d/%H:%M:%S'),\
        #             pys.time_string(z_time[round(len(z_time)/3)],fmt='%Y-%m-%d/%H:%M:%S'), \
        #             pys.time_string(z_time[round(2*len(z_time)/3)],fmt='%Y-%m-%d/%H:%M:%S'), \
        #             pys.time_string(z_time[len(z_time)-1]+1,fmt='%Y-%m-%d')]
            
            
        # axs.set_xticklabels(x_label_list)
        axs.set_ylabel(y_labs[ii],fontsize=28)
        
        # if ii==1:
        # if ii==0:
            # axs.set_ylim([0,1])
            # hline = axs.axhline(0.5,color='grey',alpha=0.5)
            # vlines = axs.axvline(time1[ii][round(197754/2)],color='grey',alpha=0.5)
            # vlines = axs.axvline(time1[ii][round(197754/4)],color='grey',alpha=0.5)
            # vlines = axs.axvline(time1[ii][round(3*197754/4)],color='grey',alpha=0.5)
            
        axs.set_xlim([z_time_dates[0],z_time_dates[(len(z_time_dates)-1)]])
        
        leg = axs.legend(loc='upper left', fontsize=20)
        leg.legend_handles[0].set_linewidth(3)
        # leg.legend_handles[1].set_linewidth(3)

        # if ii==0:
        #     leg.legend_handles[2].set_linewidth(3) 
            # leg.legend_handles[3].set_linewidth(1.5) 
    
    savepath_no_enc = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/z_plots/'
    
    savepath_enc = savepath_no_enc+'Encs/'#+str(enc_num)+'/'
    
    
    if enc_save:
        savename = 'Enc'+str(enc_num)
        savepath = savepath_enc   
    elif title != None:
        savename = title.replace(" ", "")
        savepath = savepath_no_enc   
    else:
        savename = t0+'to'+tf
        savepath = savepath_no_enc
    
    savename = savename+'_quiescent_regions'
    
    isdir = os.path.isdir(savepath)
    
    if isdir == False:
        os.makedirs(savepath)
    
    plt.subplots_adjust(wspace=0, hspace=0)
    
    # breakpoint()
    
    if save:    
        plt.savefig(savepath+savename,bbox_inches='tight')
    else:
        plt.show()
        
    plt.clf()
    plt.cla()

    plt.close(fis1)
    plt.close('all')
        
def quiescent_enc_plot(save=True,enc_save=True,other_comp=False):
    enc_num = 1
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    #print(hpos_path)
    
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    #print(pyt.tplot_names())
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    for enc in enc_flt[1:18]:
        enc_str = pys.time_string(enc[0])
        enc_end = pys.time_string(enc[1])
        
        hpos_where = np.where((hpos_time_arr>enc[0])&(hpos_time_arr<enc[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<65)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        title = 'Encounter '+str(enc_num+1)+': '+t0p[0:20]+' to '+tfp[0:20]
        
        if other_comp:
            title = title+' All Components'
        
        quiescent_plot(t0=t0,tf=tf,title=title,save=save,enc_save=enc_save,other_comp=other_comp)
        
        enc_num+=1
        
def quiescent_vel(t0='2018-11-04',tf='2018-11-05'):
    
    psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00')
    spi_data = pyt.get_data('VEL_RTN_SUN')
    
    psp.spc(trange=[t0,tf],level='L3')
    spc_data = pyt.get_data('vp_fit_RTN')
    
    spi_time = spi_data[0]
    spi_dat = spi_data[1]
    spc_time = spc_data[0]
    spc_dat = spc_data[1]
    
    plt.plot(spi_time,spi_dat[:,0])
    plt.plot(spc_time,spc_dat[:,0])
    
    plt.show()
    
def SPAN_FOV(enc=1,enc_radius=45,plot=False, store=False):
    # print('memes')
    
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc_arr = list(range(1,17))
        
        title_mod = 'for All Encounters'
        
    elif enc == 'no 13':
        enc_arr = list(range(2,13))
        enc_arr.append(14)
        enc_arr.append(15)
        enc_arr.append(16)
        title_mod = 'for All Encounters, no 1 or 13'
        
    elif enc == 'no 1':
        enc_arr = list(range(2,15)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        title_mod = 'for All Encounters, no 1'
        
    elif type(enc) is int:
        enc_arr = [enc]
        
    else:
        enc_arr = enc
        
    # breakpoint()
        
    for encs in enc_arr:
        
        enc_num = encs
        
        hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
        #print(hpos_path)
        pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
        #print(pyt.tplot_names())
        hpos = pyt.get_data('position')
        
        hpos_time_arr = hpos[0]
        hpos_data_arr = hpos[1]
        
        enc_ind = (enc_num-1)
        
        enc = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc[0])
        enc_end = pys.time_string(enc[1])
        
        hpos_where = np.where((hpos_time_arr>enc[0])&(hpos_time_arr<enc[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<enc_radius)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        t0_flt = pys.time_float(t0)
        tf_flt = pys.time_float(tf)

        #specify time range in the form ['yyyy-mm-dd/hh:mm:ss','yyyy-mm-dd/hh:mm:ss']
        trange=[t0,tf]
        
        #specify data type to plot
        datatype='spi_sf00' #protons
        spi_vars = psp.spi(trange=trange, datatype=datatype, level='L3', time_clip=True,username=sweap_id,password=sweap_pass,last_version=True)
        
        prefix='psp_spi_'
        # pyt.tplot([prefix+'SUN_DIST',prefix+'EFLUX_VS_ENERGY',prefix+'EFLUX_VS_THETA',prefix+'EFLUX_VS_PHI'])
    
    
        #define variables
        eflux_phi_data=pyt.get_data(prefix+'EFLUX_VS_PHI')
        times_unix=eflux_phi_data.times
        eflux = eflux_phi_data.y
        phi = eflux_phi_data.v
        
        rad_data = pyt.get_data('psp_spi_SUN_DIST')
        rad = rad_data.y
        rad = np.array(rad-Rs)/Rs
        
        # breakpoint()
        
        #determine phi angle with max eflux
        max_phi_ind = np.argmax(eflux, axis=1)
        max_phi = phi[0, max_phi_ind]
        # print(max_phi)
        
        t_string = pys.time_string(times_unix)
        
        times = pd.to_datetime(t_string)
        times = times.to_numpy()
        
        # times = [dateutil.parser.parse(t) for t in pys.time_string(times_unix)]     
        
        # breakpoint()
        
        # if min(rad) < 35:
        #     spline_r_35 = UnivariateSpline(times_unix, rad-35., s=0)
        #     r1x_35, r2x_35 = spline_r_35.roots()
        
        # root_times = [dateutil.parser.parse(t) for t in pys.time_string([r1x_35,r2x_35])]     
        
        #define fov array
        tlen = times_unix.shape[0]
        phi_fov=np.ones(tlen)
        phi_av_fov = np.ones(tlen)
        
        #set threshold 163.125 degrees
        phi_thresh = phi[0,1]
        
        i = 0
        # Initialize an empty list to store moving averages
        moving_averages = []
        point_ratios = []

        window_size = 515 #roughly 30 minute windows

        # Loop through the array to consider
        # every window of size 3
        while i < len(max_phi):

            # Store elements from i to i+window_size
            # in list to get the current window
            
            if i < int(window_size/2):
                window = max_phi[i : i + int(window_size/2)]
                win_len = len(window)
            elif (i+int(window_size/2)) > len(max_phi):
                window = max_phi[i - int(window_size/2) : i]
                win_len = len(window)
            else:
               window = max_phi[i - int(window_size/2) : i + int(window_size/2)]
               win_len = window_size

            # Calculate the average of current window
            window_average = round(sum(window) / win_len, 2)

            # Store the average of current
            # window in moving average list
            moving_averages.append(window_average)
            
            # Check ratio of good points to bad points in window
            
            good = np.where(window<phi_thresh)
            good = good[0]
            
            bad = np.where(window>=phi_thresh)
            bad = bad[0]
            
            if len(bad) != 0:
                
                if len(good)/len(bad)>300.:
                    good_bad_ratio = 300.
                
                else:    
                    good_bad_ratio = len(good)/len(bad)
            
            else:
                good_bad_ratio = 300.
                
            point_ratios.append(good_bad_ratio)

            # Shift window to right by one position
            i += 1
        
        moving_avs = np.array(moving_averages)
        
        point_ratios = np.array(point_ratios)
        
        exact_where = np.where(max_phi<phi_thresh)
        exact_where = exact_where[0]
        
        phi_fov[exact_where] = 0
        
        av_where = np.where(moving_avs<155)
        av_where = av_where[0]
        
        phi_av_fov[av_where] = 0

        if plot:
                
            fig, ax = plt.subplots(figsize=(12, 5))
        
            #print(times.shape)
            # start_tind = 66500
            # stop_tind = 67000
            if np.isnan(phi[0,0]):
                phi[0,0] = 174.375
            # breakpoint()
            p = ax.pcolormesh(times, phi[0,:], eflux.T, norm=matplotlib.colors.LogNorm())
            plt.colorbar(p, ax=ax, label=f'$(cm^2 \\ s \\ sr \\ eV)^{-1}$')
            # ax.plot(times, max_phi, 'k')
            ax.plot(times, moving_avs, 'k')
            
            # ax.plot(times,phi_fov*160,'r+')
            
            ax.plot(times,phi_av_fov*120,'b+')
            
            ax.plot(times,(point_ratios*0.2)+100,'y')
            
            # ax.axvspan(root_times[0], root_times[1], facecolor='g', alpha=0.5)
            
            ax.set_title("Encounter "+str(enc_num)+" SPAN-Ion Phi-direction.")
                    
            ax.set(ylim=(100, 185), xlabel='Time', ylabel=f"$\\phi$ [deg]")
            
            plt.show()
        
        #------------------store quality flags-------------------#
        
        if store:
        
            tplot_savename = 'SPAN_ion_fov_flags'+'_enc_'+str(enc_num)+'.cdf'
            tplot_savepath = '/Users/besh2109/Desktop/SPAN Checks/FOV flags/'
    
            tplot_time = times_unix
            
            tplot_phi_fov = phi_fov
            
            tplot_phi_fov_av = phi_av_fov
            
            tplot_phi_ratio = point_ratios
            
            tplot_r_Rs = rad
            
            # breakpoint()
            
            pyt.store_data("phi_fov", data={'x':tplot_time, 'y':tplot_phi_fov})
            pyt.store_data("phi_fov_average", data={'x':tplot_time, 'y':tplot_phi_fov_av})
            pyt.store_data("phi_fov_ratio", data={'x':tplot_time, 'y':tplot_phi_ratio})
            pyt.store_data("psp_radial_dist_Rs",data={'x':tplot_time,'y':tplot_r_Rs})
    
            cdf_var_list = ["phi_fov","phi_fov_average","phi_fov_ratio","psp_radial_dist_Rs"]
            
            pyt.tplot_save(cdf_var_list,tplot_savepath+tplot_savename) #saves the quality flags to a .cdf file
            
            
            pyt.del_data()
            
def SPAN_SPC_QTN(enc=6,enc_radius=60,plot=False, store=False):
    
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc_arr = list(range(1,17))
        
        title_mod = 'for All Encounters'
        
    elif enc == 'no 13':
        enc_arr = list(range(2,13))
        enc_arr.append(14)
        enc_arr.append(15)
        enc_arr.append(16)
        title_mod = 'for All Encounters, no 1 or 13'
        
    elif enc == 'no 1':
        enc_arr = list(range(2,17)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        title_mod = 'for All Encounters, no 1'
        
    elif type(enc) is int:
        enc_arr = [enc]
        
    else:
        enc_arr = enc
        
    # breakpoint()
        
    for encs in enc_arr:
        
        enc_num = encs
        
        hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
        #print(hpos_path)
        pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
        #print(pyt.tplot_names())
        hpos = pyt.get_data('position')
        
        hpos_time_arr = hpos[0]
        hpos_data_arr = hpos[1]
        
        enc_ind = (enc_num-1)
        
        enc = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc[0])
        enc_end = pys.time_string(enc[1])
        
        hpos_where = np.where((hpos_time_arr>enc[0])&(hpos_time_arr<enc[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<enc_radius)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        t0_flt = pys.time_float(t0)
        tf_flt = pys.time_float(tf)

        #specify time range in the form ['yyyy-mm-dd/hh:mm:ss','yyyy-mm-dd/hh:mm:ss']
        trange=[t0,tf]
        
        psp.spi(trange=trange,level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)
        spi_data = pyt.get_data('psp_spi_DENS')
        spi_time = spi_data[0]
        spi_dens = spi_data[1]
        
        psp.spc(trange=trange, level='L3',username=sweap_id,password=sweap_pass,last_version=True)
        spc_data = pyt.get_data('psp_spc_np_fit')
        
        if spc_data==None:
            spc_data = pyt.get_data('spp_spc_np_fit')
        
        spc_time = spc_data[0]
        spc_dens = spc_data[1]
        
        psp.fields(trange=trange, datatype='sqtn_rfs_V1V2', level='L3',last_version=True,username=fields_id,password=fields_pass)
        qtn_data = pyt.get_data('electron_density')
        
        if qtn_data is None:
            qtn_data_exists = False
        else:
            qtn_data_exists = True
        
            qtn_time = qtn_data[0]
            qtn_dens = qtn_data[1]
            
            qtn_errs = pyt.get_data('electron_density_delta')
            qtn_err_time = qtn_errs[0]
            qtn_errs = qtn_errs[1]

        # breakpoint()
        #------------------ downsample and clean SPC data ------------------#
        
        # spc_tmp = []
        # for i in range(len(spi_time)):
        #     spi_window_start = spi_time[i]-30
        #     spi_window_end = spi_time[i]+30
        #     spc_where = np.where((spc_time>spi_window_start) & (spc_time<spi_window_end))
        #     spc_where = spc_where[0]
        #     nan_total = sum(np.isnan(spc_dens[spc_where]))
        #     if nan_total > len(spc_where)/2:
        #         spc_tmp.append(np.nan)
        #     else:
        #         spc_tmp.append(np.nanmedian(spc_dens[spc_where]))
        #     # if i%2500 == 0:
        #     #     print(round(i/len(spi_time)*100,2),"%")
                
        # # plt.plot(spc_tmp)
        
        # spc_time_down = spi_time         
        # spc_dens_down = np.array(spc_tmp) #downsampled SPC data
        

        
        # Define the window size in time (delta_t), 1 minute or 60 seconds
        delta_t = 180  # Adjust this as needed
        
        # breakpoint()
        # print('ey1')
        
        spc_time_down = np.array([])
        spc_dens_down = np.array([])
        
        n_arrays = 100
        # break data into 100 chunks so that can be looped through. 
        # This saves time and RAM.
        
        spc_time_segs, spc_dens_segs = split_data_evenly_in_time(spc_time, spc_dens, n_arrays)
        spi_time_segs, spi_dens_segs = split_data_evenly_in_time(spi_time, spi_dens, n_arrays)
        
        
        for i in range(n_arrays):
            
            # split_arrays = np.array_split(array1, n_arrays)
            spc_split_time_arr = spc_time_segs[i]
            spc_split_data_arr = spc_dens_segs[i]
            spi_split_time_arr = spi_time_segs[i]
                          
            time_diff = spc_split_time_arr[:, None] - spi_split_time_arr

            # Use boolean indexing to create a 2D mask
            window_mask = (time_diff >= -delta_t/2) & (time_diff <= delta_t/2)
            
            window_values = np.where(window_mask.T, spc_split_data_arr, np.nan)
            spc_medians = np.nanmedian(window_values, axis=1)
            
            # Compute the median along the second axis (axis=1) ignoring NaN values
            spc_time_down = np.append(spc_time_down,spi_split_time_arr)
            spc_dens_down = np.append(spc_dens_down,spc_medians)
            # print(i)
        
        # print('ey2')

        # breakpoint()
        #------------------construct SPAN quality flag---------------------#
        
        span_check_savename = 'SPAN_ion_fov_flags_enc_'+str(enc_num)+'.cdf'
        span_check_savepath = '/Users/besh2109/Desktop/SPAN Checks/FOV flags/'
        
        pyt.tplot_restore(span_check_savepath+span_check_savename)
        
        fov_flag_average = pyt.get_data('phi_fov_average')
        fov_av_time = fov_flag_average[0]
        fov_flag_av = fov_flag_average[1]
        
        fov_where = np.where(np.isin(fov_av_time,spi_time))
        fov_where = fov_where[0]
        
        fov_flag_inv = np.array(fov_flag_av[fov_where],dtype=int)
        fov_flag = (1-fov_flag_inv)*2
        
        #--------------------compare both SPAN and SPC to QTN density-------#
        
        if qtn_data_exists:
        
            qtn_indices = assign_closest_index(spi_time,qtn_time)
    
            qtn_values_check = qtn_dens[qtn_indices]
            
            qtn_spc_diff = abs(qtn_values_check - spc_dens_down)
            qtn_spi_diff = abs(qtn_values_check - spi_dens)
            
            spc_per_diff = qtn_spc_diff/qtn_values_check*100 #percent difference of both SPC and SPI
            spi_per_diff = qtn_spi_diff/qtn_values_check*100
            
            # breakpoint()
            
            dens_cor_where = np.where(np.isin(spi_time,fov_av_time))
            dens_cor_where = dens_cor_where[0]
            
            spi_per_diff_cor = spi_per_diff[dens_cor_where]
            
            
            qtn_err_pers = qtn_errs[qtn_indices,:]/qtn_values_check[:,np.newaxis]*100
            
            qtn_err_pers_corr = qtn_err_pers[dens_cor_where,:]
            
            one_sig_max = np.max(qtn_err_pers,axis=1)
            two_sig_max = 2*np.max(qtn_err_pers,axis=1)
            three_sig_max = 3*np.max(qtn_err_pers,axis=1)
            
            one_sig_max_corr = np.max(qtn_err_pers_corr,axis=1)
            two_sig_max_corr = 2*np.max(qtn_err_pers_corr,axis=1)
            three_sig_max_corr = 3*np.max(qtn_err_pers_corr,axis=1)
            
            # spi_one_sig_where = np.where(spi_per_diff<one_sig_max)
            # spi_one_sig_where = spi_one_sig_where[0]
            # spi_two_sig_where = np.where(spi_per_diff<two_sig_max)
            # spi_two_sig_where = spi_two_sig_where[0]
            # spi_three_sig_where = np.where(spi_per_diff<three_sig_max)
            # spi_three_sig_where = spi_three_sig_where[0]
            
            spi_one_sig_where = np.where(spi_per_diff_cor<one_sig_max_corr)
            spi_one_sig_where = spi_one_sig_where[0]
            spi_two_sig_where = np.where(spi_per_diff_cor<two_sig_max_corr)
            spi_two_sig_where = spi_two_sig_where[0]
            spi_three_sig_where = np.where(spi_per_diff_cor<three_sig_max_corr)
            spi_three_sig_where = spi_three_sig_where[0]
            
            spc_one_sig_where = np.where(spc_per_diff<one_sig_max)
            spc_one_sig_where = spc_one_sig_where[0]
            spc_two_sig_where = np.where(spc_per_diff<two_sig_max)
            spc_two_sig_where = spc_two_sig_where[0]
            spc_three_sig_where = np.where(spc_per_diff<three_sig_max)
            spc_three_sig_where = spc_three_sig_where[0]
        
        dens_cor_where = np.where(np.isin(spi_time,fov_av_time))
        dens_cor_where = dens_cor_where[0]
        
        spi_dens_time_cor = spi_time[dens_cor_where]
        spi_dens_corrected = spi_dens[dens_cor_where]
        
        # breakpoint()
        
        if qtn_data_exists:
        
            # span_quality_flag = np.empty_like(spi_time,dtype=float)
            span_quality_flag = np.array(fov_flag,dtype=float)
            nanwhere = np.where(np.isnan(spi_dens_corrected))
            nanwhere = nanwhere[0]
            # breakpoint()
            span_quality_flag[nanwhere] = np.nan
            span_quality_flag[spi_three_sig_where] += 1
            span_quality_flag[spi_two_sig_where] += 1
        
        else:
            # span_quality_flag = np.empty_like(spi_time)
            span_quality_flag = np.array(fov_flag,dtype=float)
            nanwhere = np.where(np.isnan(spi_dens_corrected))
            nanwhere = nanwhere[0]
            span_quality_flag[nanwhere] = np.nan
        
        #-------------------construct SPC quality flag---------------------#
        
        if qtn_data_exists:
        
            spc_quality_flag = np.zeros(spi_time.shape,dtype=float)
            nanwhere = np.where(np.isnan(spc_dens_down))
            nanwhere = nanwhere[0]
            spc_quality_flag[nanwhere] = np.nan
            # spc_quality_flag = fov_flag
            spc_quality_flag[spc_three_sig_where] += 1
            spc_quality_flag[spc_two_sig_where] += 1
        
        else:
            
            spc_quality_flag = np.zeros(spi_time.shape,dtype=float)
            nanwhere = np.where(np.isnan(spc_dens_down))
            nanwhere = nanwhere[0]
            spc_quality_flag[nanwhere] = np.nan
            
            
            #------------------store quality flags-------------------#
            
        if store:
            # breakpoint()
            tplot_savename = 'SPAN_SPC_QTN_flags'+'_enc_'+str(enc_num)+'.cdf'
            tplot_savepath = '/Users/besh2109/Desktop/SPAN Checks/'
    
            tplot_time_spi = spi_dens_time_cor
            tplot_time_spc = spi_time
            
            tplot_span_quality_flag = span_quality_flag
            tplot_spc_quality_flag = spc_quality_flag
            
            pyt.store_data("SPAN_qual_flag", data={'x':tplot_time_spi, 'y':tplot_span_quality_flag})
            pyt.store_data("SPC_qual_flag", data={'x':tplot_time_spc, 'y':tplot_spc_quality_flag})
    
            cdf_var_list = ["SPAN_qual_flag","SPC_qual_flag"]
            
            pyt.tplot_save(cdf_var_list,tplot_savepath+tplot_savename) #saves the quality flags to a .cdf file
        
        #------------------check agreement with figure---------------------#
        
        if plot:
            
            # breakpoint()
            
            fig = plt.figure(figsize=(15,10))
            axs = fig.add_subplot(111)
            axs.set_title('QTN Density Comparison Encounter '+str(enc_num))
            axs.plot(spc_time_down,spc_dens_down,linewidth=0.5,label='SPC Proton Density',color='tab:orange')
            # axs.plot(spc_time,spc_dens,linewidth=0.1,label='SPC Proton Density',color='tab:orange')
            # axs.plot(spi_time,spi_dens,linewidth=0.1,label='SPAN-I Proton Density',color='tab:blue')
            # axs.plot(qtn_time,qtn_dens,linewidth=0.1,label='QTN Electron Density', color='black')
            
            axs.set_ylim(-3,5000)
            
            leg = plt.legend(loc='upper right',fontsize=18)
            
            for legobj in leg.legend_handles:
                legobj.set_linewidth(2.0)
            
            
            # leg.legend_handles[0].set_sizes = [30]
            # leg.legend_handles[1].set_sizes = [30]
            
            plt.show()
        
        pyt.del_data() #clear all tplot variables
        
        # breakpoint()


def total_time_check(enc_radius=65):

    enc_num = 1
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    #print(hpos_path)
    
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    #print(pyt.tplot_names())
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    check = []
    
    for enc in enc_flt[0:18]:
        enc_str = pys.time_string(enc[0])
        enc_end = pys.time_string(enc[1])
        
        hpos_where = np.where((hpos_time_arr>enc[0])&(hpos_time_arr<enc[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<enc_radius)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        t0f = time_select[0]
        tff = time_select[-1]
        
        # print((tff-t0f)/3600.)
        
        check.append((tff-t0f)/3600.)
    
    print(len(check))
    print(np.sum(check))
    
def convergence_test(t0,tf,bincount=499,runs=1000,z_thresh=0.95,q_thresh = 0.5):
    
    varis = quiescent_calc(t0=t0,tf=tf)

    z_time = varis['z_time']
    z = varis['z']
    
    qual_r_time = np.array(z_time)
    # quiet_z = varis['quiet_z']
    # b_time = varis['b_time']
    # Br = varis['Br']
    # quiet_B = varis['quiet_B']
    # B_mag = varis['B_mag']
    
    # carr_time = varis['carr_time']
    # carr_long = varis['carr_long']
    # carr_lat = varis['carr_lat']
    # carr_dif = varis['carr_dif']
    
    #--------------------interpolate over nans in z------------------------------#
    
    z_i = np.array(z)
    
    nans, x = nan_helper(z_i)
    nan_group_indices = find_nan_groups_indices(z_i)
    
    long_arr = np.array([],dtype=int)
    for bbeg in nan_group_indices:
        if len(bbeg)>200:
            long_arr = np.append(long_arr,bbeg)
            
    nans[long_arr] = False
    
    #interpolate over the nans that are very short
    z_i[nans] = np.interp(x(nans), x(~nans), z_i[~nans])
    
    nanwhere = np.isnan(z_i)
    
    nanlst=[]
    
    for i in range(len(nanwhere)-1):
        if nanwhere[i]==nanwhere[i+1]: 
            nanlst.append(False)
        else:
            nanlst.append(True)
    
    if np.isnan(z_i[-1]):
        nanlst.append(True)
    else:
        nanlst.append(False)
    
    nanlst_arr = np.array(nanlst)
    
    truewhere = np.where(nanlst_arr)
    
    bindices = truewhere[0] #bin indices, or BINdicess
    
    bin_lst = []
    bin_time_lst = []
    for i in range(int(len(bindices)/2)):
        bin_lst.append([bindices[2*i],bindices[2*i+1]])
        bin_time_lst.append([z_time[bindices[2*i]],z_time[bindices[2*i+1]]])

    bindices = np.array(bin_lst)
    
    #----------------------- random analysis, set bars randomly --------------------------#



                # wndw_bars = np.array([])
    wndw_list = []
    # breakpoint()
    for jj in range(runs): #runs = number of random runs
        i_max = len(z_time)
        # wndw_bars_i = np.array(z_time[0])
        
        # csv_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/random_regions/'
        # csv_savename = 'enc_'+str(enc_num)+'_quiescent_random.csv'
        
        bins = np.array([0])
        bins = np.append(bins,random.sample(range(i_max),bincount))
        bins = np.append(bins,np.array(i_max-1))
        
        for jjj in bindices:
            bar_where = np.where((bins>=jjj[0])&(bins<=jjj[1]))
            bar_where = bar_where[0]
            
            bins = np.delete(bins,bar_where)
            bins = np.append(bins,jjj)
        
        bins = np.sort(bins) #sorts the indices in increasing order.
        
        wndw_list.append(list(z_time[bins]))
        # wndw_bars_tmp = np.array(wndw_list)
        
        # wndw_bars = np.append(wndw_bars,z_time[bins], axis=0)
        bin_store_r = z_time[bins]

    # if atype == 'rand' and runs > 1: #this should only be used for random bars.
        
    # pos_len = 50
    # progress = np.linspace(0,pos_len,21)
    range_check = np.arange(10)/10*runs
    # i=0
    q_z_lst = []
    q_z_add = np.zeros(len(z_time))
    
    run_count_list = []
    region_count_list = []
    region_len_list = []
    
    # breakpoint()
    for kk in range(len(wndw_list)):
        # print(kk)
        wnd_strs = wndw_list[kk][0:-1]
        wnd_ends = wndw_list[kk][1:]
        
        # csv_arr = np.array([])
        q_z_arr = np.array([])
        
        
        
        if kk in range_check:
            print(str(100*kk/runs)+'% Complete')
        
        # breakpoint()
        
        for iii in range(len(wnd_strs)):
            
            if iii == len(wnd_strs)-1:
                z_where = np.where((z_time>=wnd_strs[iii]) & (z_time<=wnd_ends[iii])) #the last bin includes the last data point
            else:
                z_where = np.where((z_time>=wnd_strs[iii]) & (z_time<wnd_ends[iii]))
            
            
            
            
            start_date = pys.time_string(wnd_strs[iii])
            end_date = pys.time_string(wnd_ends[iii])
            
            z_time_in_bin = z_time[z_where]
            z_in_bin = z[z_where]
            
            q_z_where = np.where((z_in_bin<0.05) | (z_in_bin>0.95))
            
            q_z_pnts = len(q_z_where[0])
            z_pnts = len(z_in_bin)
            
            vari = np.nanvar(z_in_bin)
            std = np.nanstd(z_in_bin)
            med_z = np.nanmedian(z_in_bin)
            
            med_z = med_z - 1/2
            med_z = np.sqrt(med_z**2)
            
            
            if z_pnts==0:
                q_z_frac=np.nan
            
            else:
                q_z_frac = q_z_pnts/z_pnts
            
            if q_z_frac < z_thresh:
                # q_z_frac = q_z_frac*0.3 #suppress values with low q/z
                q_z_frac = 0
            else:
                # q_z_frac = q_z_frac*2 #inflate values with high q/z
                q_z_frac = 1
            # q_z_tmp_arr = (np.ones(len(z_in_bin))*q_z_frac)/((vari+1)*(med_z+1))
            # q_z_tmp_arr = np.ones(len(z_in_bin))*(q_z_frac)/((std+1)*(med_z+1))
            
            q_z_tmp_arr = np.ones(len(z_in_bin))*(q_z_frac)
            
            q_z_arr = np.append(q_z_arr,q_z_tmp_arr) #create array that gives a q/z fraction for every data point.
            
            q_z_arr = q_z_arr #this line ensures that q_z_arr maintains the correct shape through the np.append. Seems redundant, but it is not.
            
    
        run_arr = np.linspace(1,runs,num=250,dtype=int)
            
        # q_z_lst.append(q_z_arr)
        q_z_add = q_z_add + q_z_arr
        # if kk == 0:
        #     qual_r_one = q_z_arr
            
        # if kk == 4:
        #     qual_r_five = q_z_add/5

        qual_r_arr = q_z_add/(kk+1)
        
        
    
    #---------------- check quiescent region counts over threshold = 0.5 --------------------------#
    
        if kk+1 in run_arr:
        
            r_reg_check = (qual_r_arr>q_thresh)
            
            r_regions = np.zeros(r_reg_check.shape)*np.nan
            i_ind = 1
            for k in range(len(r_regions)):
                
                if r_reg_check[k]==True:
                    r_regions[k] = i_ind
                
                if k!=0:
                    if r_reg_check[k]==False and r_reg_check[k-1]==True:
                        i_ind+=1
                        
            reg_where = np.where(r_reg_check)
            reg_where=reg_where[0]
            reg_len = len(reg_where)
                        
            run_count_list.append(kk+1)
            region_count_list.append(i_ind)
            region_len_list.append(reg_len)
            
            # breakpoint()
            # start_dates = []
            # end_dates = []
            # durations = []
            # for ii in range(int(np.nanmax(r_regions))):
            #     isl_where = np.where(r_regions==(ii+1))
            #     isl_where = isl_where[0]
                
            #     start_ind = isl_where[0]
            #     end_ind = isl_where[-1]
                
            #     start_time_flt = qual_r_time[start_ind]
            #     end_time_flt = qual_r_time[end_ind]
                
            #     duration = end_time_flt-start_time_flt
                
            #     start_date = pys.time_string(start_time_flt) 
            #     end_date = pys.time_string(end_time_flt)
                
            #     if duration>=30: #cut out short events, probably flukes
        
            #         start_dates.append(start_date)
            #         end_dates.append(end_date)
            #         durations.append(duration)
            
            
    #----------------------------- generate plots ---------------------------------#
    
    median_counts = np.median(region_len_list)
    
    fig = plt.figure(figsize=(10,5))
    
    axs = fig.add_subplot(111)
    axs.set_title('Quiescent Plasma Data vs ID Algorithm Runs, '+t0+' to '+tf)
    axs.plot(run_count_list,region_len_list,label='Quiescent Data',color='tab:blue')
    axs.set_ylabel('Quiescent Region Data Counts')
    axs.set_xlabel('ID Algorithm Run Count')
    axs.axhline(median_counts,linestyle='dashed',linewidth=1.5,color='tab:orange', label='Median Quiescent Value')
    axs.axvline(300,linestyle='dashed',linewidth=1.5,color='tab:red', label='Chosen Run Value')
    
    # axs.set_ylim(-3,5000)
    
    leg = plt.legend(loc='upper right',fontsize=14)
    
    for legobj in leg.legend_handles:
        legobj.set_linewidth(2.0)
    
    plt.show()
    
    breakpoint()
