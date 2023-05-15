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


fields_user = os.environ['PSP_FIELDS_ID']
fields_pass = os.environ['PSP_FIELDS_PW']
sweap_user = os.environ['PSP_SWEAP_ID']
sweap_pass = os.environ['PSP_SWEAP_PW']

Rs = 6.957e5 #solar radius in km
Rs_in_m = Rs*10**3
w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec

def quiescent_id(analysis='all', mode='csv',runs=20,enc_start=1,enc_end=13,thresh=0.95,bincount=499,full_run=False): # analysis chooses type of analysis you want to do, mode is broken, 
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
    
    hpos_path = CONFIG['local_data_dir']+'/data/sci/fields/l1/ephem_eclipj2000/full_mission/'
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
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        title = 'Encounter '+str(enc_num)+': '+t0p[0:20]+' to '+tfp[0:20]
        
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
        
        
        # nanwhere = np.where(np.isnan(z))
        # nanwhere = nanwhere[0]
        
        nanwhere = np.isnan(z)
        
        nanlst=[]
        # for i in range(len(nanwhere)-1):
        #     if nanwhere[i]+1==nanwhere[i+1]: 
        #         nanlst.append(False)
        #     else:
        #         nanlst.append(True)
        
        for i in range(len(nanwhere)-1):
            if nanwhere[i]==nanwhere[i+1]: 
                nanlst.append(False)
            else:
                nanlst.append(True)
        
        if np.isnan(z[-1]):
            nanlst.append(True)
        else:
            nanlst.append(False)
        
        nanlst_arr = np.array(nanlst)
        
        truewhere = np.where(nanlst_arr)
        
        bindices = truewhere[0] #bin indices, or BINdices
        
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
                for jj in range(runs): #runs = number of random runs
                    i_max = len(z_time)
                    # wndw_bars_i = np.array(z_time[0])
                    
                    csv_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/random_regions/'
                    csv_savename = 'enc_'+str(enc_num)+'_quiescent_random.csv'
                    
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
            
            # breakpoint()
            
            # if atype=='rand':
            #     breakpoint()
            
            if atype == 'rand' and runs > 1: #this should only be used for random bars.
                
            
            
                q_z_lst = []
                q_z_add = np.zeros(len(z_time))
                for kk in range(len(wndw_list)):
                    # print(kk)
                    wnd_strs = wndw_list[kk][0:-1]
                    wnd_ends = wndw_list[kk][1:]
                    
                    # csv_arr = np.array([])
                    q_z_arr = np.array([])
                    
                    range_check = np.arange(10)/10*runs
                    
                    if jj in range_check:
                        print(jj)
                    
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
        
        if mode in ['zplot','both']:
        
            fis1 = plt.figure(figsize=(9,12))
            
            # times = [b_time,qual_r_time,qual_l_time,qual_t_time]
            # datas = [Br,qual_r,qual_l,qual_t]
            # labels = ['Br (nT)','random qual','long qual','time qual']
            
            times = [b_time,qual_r_time,qual_r_time,qual_r_time]
            datas = [Br,qual_r_one,qual_r_five,qual_r]
            labels = ['nT','quality flag, 1 run','quality flag, 5 run','quality flag, 5 run']
            names = ['Br','1 iteration','5 iterations','50 iterations']
            
            
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
                    
                    axs.set_title('Encounter '+str(i)+', Run count:'+str(runs[0]))
                    a = axs.plot(quiet_time,quiet_b,linewidth=0.1,label='Quiescent Br')
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
            
def quiescent_prop_enc(enc='all'):
    if enc == 'all':
        enc_arr = enc_flt[0:12]
    elif len(enc)!=1:
        encs = [enc]
        enc_arr = enc_flt[encs]
    else:
        enc_arr = [enc_flt[enc]]
    
    hpos_path = CONFIG['local_data_dir']+'/data/sci/fields/l1/ephem_eclipj2000/full_mission/'
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    # enc_ind = (enc-1)
    
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
        
        quiescent_prop(t0=t0,tf=tf)
        
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
    csv_path='/Users/besh2109/Desktop/psp_regions/region_data/'

    psp.fields(trange=[t0,tf], datatype='mag_RTN_4_Sa_per_Cyc', level='l2',last_version=True)
    mag_data = pyt.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')

    if encounter in [1,12]:

        psp.spc(trange=[t0,tf], level='L3')
        vel_data = pyt.get_data('vp_fit_RTN')
    
    else:
        psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00')
        vel_data = pyt.get_data('VEL_RTN_SUN')
    
    psp.fields(trange=[t0,tf], datatype='ephem_spp_hg', level='l1',last_version=True)
    pos_data = pyt.get_data('position') #retrieve PSP position data from tplot variable
    
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
    what IPs are hogging data. Particularly the public NASA servers. If I pull too much information for too long
    (like looping through all the Parker Solar Probe data)
    then the servers will seriously slow down how much I can pull after a while. 
    
    Here I'm pulling straight from the Berkeley Servers so that shouldnt
    be an issue but its good practice to check whether you have the data already before unnecessarily making a
    a request of the server.
    
    Incidentally, this will make it so that you can run the code without using the internet (assuming you already have the data onhand).
    """

    
    mag_file_name = pys.time_string(i_day,fmt='psp_fld_l2_mag_RTN_4_Sa_per_Cyc_%Y%m%d_v02.cdf')
    # mag_file_name = pys.time_string(i_day,fmt='psp_fld_l2_mag_RTN_%Y%m%d_v02.cdf')
    # mag_file_name = pys.time_string(i_day,fmt='psp_fld_l2_mag_RTN_1min_%Y%m%d_v02.cdf')
    mag_isfile = os.path.isfile(mag_file_path+mag_file_name)     
    mag_infile = mag_file_path+mag_file_name
    
    psp.fields(trange=[t0,tf], datatype='mag_RTN_4_Sa_per_Cyc', level='l2',last_version=True,\
               username=fields_user,password=fields_pass)
    # psp.fields(trange=[t0,tf], datatype='mag_RTN', level='l2',last_version=True)
    

    
    
    # mag_data = pyt.get_data('psp_fld_l2_mag_RTN_1min')
    mag_data = pyt.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
    # mag_data = pyt.get_data('psp_fld_l2_mag_RTN')
    
    ephem_isfile = os.path.isfile(ephem_file_path+ephem_file_name)
    
    spc_file_name = pys.time_string(i_day,fmt='psp_swp_spc_l3i_%Y%m%d_v01.cdf')
    spc_isfile = os.path.isfile(spc_file_path+spc_file_name)  
    spc_infile = spc_file_path+spc_file_name
    

    
    if encounter in [1]:

        psp.spc(trange=[t0,tf], level='L3',\
                   username=sweap_user,password=sweap_pass)
        vel_data = pyt.get_data('vp_fit_RTN')
    
    else:
        psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',\
                   username=sweap_user,password=sweap_pass)
        vel_data = pyt.get_data('VEL_RTN_SUN')
    
    psp.fields(trange=[t0,tf], datatype='ephem_spp_hg', level='l1',last_version=True,\
               username=fields_user,password=fields_pass)
    pos_data = pyt.get_data('position') #retrieve PSP position data from tplot variable
        
    mag_time_arr = mag_data[0]
    mag_data_arr = mag_data[1]
    
    b_mag = np.sqrt(mag_data_arr[:,0]**2+mag_data_arr[:,1]**2+mag_data_arr[:,2]**2)
    
    br_meas = -mag_data_arr[:,0]/b_mag  #define the magnetic field to be opposite what it is in RTN to be consistent with DeWit 2020
    bt_meas = -mag_data_arr[:,1]/b_mag
    bn_meas = -mag_data_arr[:,2]/b_mag
    

    mag_len = len(b_mag)
    
    br_where = np.where(br_meas < 0)
    
    # br_meas[br_where] = -br_meas[br_where]
    # bt_meas[br_where] = -bt_meas[br_where]
    # bn_meas[br_where] = -bn_meas[br_where]

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
    
    vmag = 400 #km/s
    
    vel_time_arr = vel_data[0]
    vel_data_arr = vel_data[1]
    
    v_r_data = vel_data_arr[:,0]
    
    """
    Useful to fill in NaNs in SPC data in order to make parker calculation.
    Slice day into 102 slices to take the median of to make a median array.
    Fill in nans with values from this median array.
    This ought to be okay because we're only looking for a rough estimate.
    """
    v_mean_arr = np.ones(len(v_r_data))
    v_len = len(v_r_data)
    v_bin_size = int(v_len/102)
    ii=0
    while ii<103:
        v_nan_check = v_r_data[ii*v_bin_size:(ii+1)*v_bin_size]
        slice_mean = np.nanmedian(v_nan_check)
        v_mean_arr[ii*v_bin_size:(ii+1)*v_bin_size]=slice_mean
        ii+=1
    
    if v_mean_arr[-1]==1:
        v_mean_arr[-1] = v_mean_arr[-2]
        
    v_r_data[np.isnan(v_r_data)]=v_mean_arr[np.isnan(v_r_data)]
    
    v_r_xp = np.arange(len(v_r_data))
    v_r_x = np.arange(mag_len)/mag_len*(len(v_r_data)-1)
    v_r_i = np.interp(v_r_x,v_r_xp,v_r_data)
    
    v_r=v_r_i
    # v_r = vmag
    
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
    
    if pickle:
        
        pickle_path = '/Users/besh2109/Desktop/z_pickle/'
        pickle_name = ''
        
    
    return {'z_time':z_time_arr,'z':z,'quiet_z':quiet_z,\
            'b_time':mag_time_arr,'Br':-br_meas*b_mag,'Bt':-bt_meas*b_mag,'Bn':-bn_meas*b_mag,'quiet_B':quiet_b,'B_mag':b_mag, \
            'carr_time':carr_time,'carr_long':carr_long,'carr_lat':carr_lat,'carr_dif':carr_dif_i,'time_dif':time_dif_i}
        
def quiescent_calc_enc(enc_num,enc_radius=65,save=False):
    # enc_num = enc_start
    
    hpos_path = CONFIG['local_data_dir']+'/data/sci/fields/l1/ephem_eclipj2000/full_mission/'
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
        
        save_path = '/Users/besh2109/Desktop/z_save/'
        
        pickle_tag = 'pickle/'
        numpy_tag = 'numpy/'
        tplot_tag = 'tplot/'
        
        numpy_save_arr = np.array([z_time,z])
        
        pickle_name = 'z_pickle_enc_'+str(enc_num)+'.pkl'
        
        with open(save_path+pickle_tag+pickle_name,'wb') as f:
            pkl.dump(numpy_save_arr,f)
        
        
        numpy_name = 'z_numpy_enc_'+str(enc_num)
        
        
        np.save(save_path+numpy_tag+numpy_name,numpy_save_arr)
        
        
        
        tplot_name = 'z_tplot_enc'+str(enc_num)+'.cdf'
        

        
        pyt.store_data("z", data={'x':z_time, 'y':z})
        
        pyt.tplot_save('z',save_path+tplot_tag+tplot_name)
        
    return varis
       
def quiescent_plot(t0='2018-11-04',tf='2018-11-05',title=None,save=False,other_comp=True,enc_save=False): #plots z and other parameters.
    
    t0_flt = pys.time_float(t0)
    tf_flt = pys.time_float(tf)

    enci = 0
    for enc in enc_flt:          
        if (t0_flt>enc[0]) and (t0_flt<enc[1]): 
            enc_num = enci+1 #determine which encounter today is in
            # per_date = per_flt[enci]
            # per_dist = per_dist_lst[enci]
        enci+=1

    varis = quiescent_calc(t0=t0,tf=tf)
    
    z_time = varis['z_time']
    z = varis['z']
    quiet_z = varis['quiet_z']
    b_time = varis['b_time']
    Br = varis['Br']
    Bt = varis['Bt']
    Bn = varis['Bn']
    
    quiet_B = varis['quiet_B']
    B_mag = varis['B_mag']
    
    carr_time = varis['carr_time']
    carr_long = varis['carr_long']
    carr_dif = varis['carr_dif']
    
    zlen = len(z_time)

    z_where = np.where((z_time>=t0_flt)&(z_time<tf_flt))

    z_time = z_time[z_where]
    z = z[z_where]  
    quiet_z = quiet_z[z_where]
    
    b_where = np.where((b_time>=t0_flt)&(b_time<tf_flt))

    b_time = b_time[b_where]
    Br = Br[b_where]
    Bt = Bt[b_where]
    Bn = Bn[b_where]
    quiet_B = quiet_B[b_where]
    B_mag = B_mag[b_where]
    
    carr_where = np.where((carr_time>=t0_flt)&(carr_time<tf_flt))
    carr_time = carr_time[carr_where]
    carr_long = carr_long[carr_where]
    
    i_max = len(carr_time)
    wndw_bars = np.array(carr_time[0])
    carr_int = 0
    for i in range(i_max):
        
        carr_int+=carr_dif[i]
        
        if np.abs(carr_int) > 0.82: #0.082 for granulation scale, 0.82 is supergranulation scale
            wndw_bars = np.append(wndw_bars,carr_time[i])
            carr_int = 0
        if i==i_max:
            wndw_bars = np.append(wndw_bars,carr_time[i])

    #--------------#
    # i_max = len(z_time)
    # wndw_bars = np.array(z_time[0])
    
    # bins = np.array([0])
    # bins = np.append(bins,random.sample(range(i_max),499))
    # bins = np.append(bins,np.array(i_max-1))
    
    # # breakpoint()
    
    # wndw_bars = z_time[bins]
    
    #--------------#

    fis1 = plt.figure(figsize=(30,15))
    
    data1 = [Br,z]
    data2 = [quiet_B, quiet_z]
    
    label1 = ['Br','z']
    label2 = ['Br (z<0.05)','z<0.05']
    
    y_labs = ['B Field (nT)','z-parameter']
    
    for ii in range(2):
        
        axs = fis1.add_subplot(2,1,ii+1)
        axs.plot(z_time,data1[ii], color='blue',linewidth=0.2,label=label1[ii],zorder=1)
        axs.plot(z_time,data2[ii], color='orange',linewidth=0.22,label=label2[ii],zorder=2)
        
        if ii==0:
            axs.plot(z_time,B_mag,color='black',linewidth=0.22,label='|B|',zorder=3)
            axs.plot(carr_time,carr_long,color='purple',linewidth=1,label='Carrington Long',zorder=5)
            
            if other_comp:
                axs.plot(z_time,Bt,color='red',linewidth=0.12,label='Bt',zorder=3)
                axs.plot(z_time,Bn,color='green',linewidth=0.12,label='Bn',zorder=3)
            
            if title==None:
                axs.set_title(t0+' to '+tf)
            else:
                axs.set_title(title)
                
            # axs.vlines(wndw_bars,-120,120, color='green',zorder=4,linewidth=0.5)
            # axs.set_ylim([16,17])
        
        axs.set_xticks([z_time[0],z_time[round(len(z_time)/3)],\
                       z_time[round(2*len(z_time)/3)],z_time[(len(z_time)-1)]])
        
        x_label_list = [pys.time_string(z_time[0],fmt='%Y-%m-%d/%H:%M:%S'),\
                        pys.time_string(z_time[round(len(z_time)/3)],fmt='%Y-%m-%d/%H:%M:%S'), \
                        pys.time_string(z_time[round(2*len(z_time)/3)],fmt='%Y-%m-%d/%H:%M:%S'), \
                        pys.time_string(z_time[len(z_time)-1],fmt='%Y-%m-%d/%H:%M:%S')]
            
        axs.set_xticklabels(x_label_list)
        axs.set_ylabel(y_labs[ii],fontsize=18)
        
        if ii==1:
            axs.set_ylim([-0.1,1.1])
            
        axs.set_xlim([z_time[0],z_time[(len(z_time)-1)]])
        
        leg = axs.legend(loc='upper right')
        leg.legendHandles[0].set_linewidth(1.5)
        leg.legendHandles[1].set_linewidth(1.5)
        if ii==0:
            leg.legendHandles[2].set_linewidth(1.5) 
            leg.legendHandles[3].set_linewidth(1.5) 
    
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
    
    if save:    
        plt.savefig(savepath+savename,bbox_inches='tight')
    else:
        plt.show()
        
    plt.clf()
    plt.cla()

    plt.close(fis1)
    plt.close('all')
        
def quiescent_enc_plot(other_comp=False):
    enc_num = 1
    
    hpos_path = CONFIG['local_data_dir']+'/data/sci/fields/l1/ephem_eclipj2000/full_mission/'
    #print(hpos_path)
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    #print(pyt.tplot_names())
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    for enc in enc_flt[0:13]:
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
        
        title = 'Encounter '+str(enc_num)+': '+t0p[0:20]+' to '+tfp[0:20]
        
        if other_comp:
            title = title+' All Components'
        
        quiescent_plot(t0=t0,tf=tf,title=title,save=True,enc_save=True,other_comp=other_comp)
        
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