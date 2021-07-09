#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 10:50:30 2021

@author: besh2109
"""

import numpy as np
from numpy.linalg import inv
from scipy.io import readsav
import pyspedas as pys
import pytplot as pyt
import os
import pandas as pd
from statistics import median
import math
import matplotlib.pyplot as plt
import cdflib

from .config import CONFIG
from .config import enc_flt
from .config import per_flt
from .config import per_dist_lst

Rs_grps = [[50,45],[45,40],[40,35],[35,30],[30,25],[25,20],[20,1]]

def mag_epoch(plot='epoch',quick=True,no_enc_7=False, no_n_hat = False, win_len = 1,by_Rs=False):

    i=0
    enc_num = len(per_flt)
    while i <= enc_num:
        
        Rs = 6.957e5 #solar radius in km
        Rs_in_m = Rs*10**3
        w = 2*np.pi/(25.38*86400) # angular frequency of the sun in degrees/sec
        
        v = 100000 #m/s typical slow solar wind speed, may replace later with measurement values
        
        
        if not by_Rs:
            if i==0:
                csv_filename = 'harmwave_master_arch.csv'
                csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
                if no_enc_7:
                    name = 'All Encounters sans 7'
                else:
                    name = 'All Encounters'
            else:
                csv_filename = 'enc_'+str(i)+'_harmwave_arch.csv'
                csv_path = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/Enc'+str(i)+'/'
                savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/histograms/'
                savename = 'Enc_'+str(i)+'_mag_epoch.png'
                name = 'Encounter '+str(i)
        
        else:
            csv_filename = 'harmwave_master_arch.csv'
            csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
            if i==0:
                if no_enc_7:
                    name = 'All Radial Distances sans Enc 7'
                else:
                    name = 'All Radial Distances'
            else:
                name = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+' Rs'
        
        isfile = os.path.isfile(csv_path+csv_filename)
        
        if isfile:
            
            df = pd.read_csv(csv_path+csv_filename)
            bf = df.to_numpy()

            af = np.delete(bf,bf[:,1]<90,0)
            
            if by_Rs and i !=0:
                af = np.delete(af,af[:,2]>Rs_grps[i-1][0],0)
                af = np.delete(af,af[:,2]<Rs_grps[i-1][1],0)
            
            if no_enc_7 and i != enc_num: # this portion of code kills events in encounter 7
                dates = list(af[:,0])     # when doing the overview picture w/ all encounters
                for j in range(len(dates)):
                    dates[j] = dates[j][0:4]
                npdates = np.array(dates)
                af = np.delete(af,npdates=='2021',0)
            
            wave_start = np.array(pys.time_float(af[:,0]))
            wave_end = np.array(wave_start+af[:,1])
            
            window_start = np.array(wave_start - win_len*af[:,1])#*(1/3)) #complete epoch analysis for window larger than wave itself
            window_end = np.array(wave_end + win_len*af[:,1])#*(1/3))     #want to see if theres a difference between times during and before/after events
            
            dates = list(af[:,0])
            for j in range(len(dates)):
                dates[j] = dates[j][0:10]

            uniq_dates = np.unique(dates)
            date_flt = pys.time_float(uniq_dates)
            uniq_next = pys.time_string(np.array(date_flt)+86400.)
            
            duration = np.array(af[:,1])
            
            mag_time = []
            mag_r_data = []
            mag_t_data = []
            mag_n_data = []
            
            vel_time = []
            vel_r_data = []
            vel_t_data = []
            vel_n_data = [] 
            
            dens_time = []
            density_data = []
            
            r_data = []
            r_data_mod = []
            p_theta = []
            mag_len_arr = []
            
            vel_len_arr = []
            dens_len_arr = []

            for j in range(len(uniq_dates)): #gather data for each day range(2):#
                
                
                """ PSP wave data """
                if quick==True:
                    pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]], datatype='mag_RTN_4_Sa_per_Cyc', level='l2',last_version=True)
                    mag_data = pyt.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
                else:
                    pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]], datatype='mag_RTN', level='l2',last_version=True)
                    mag_data = pyt.get_data('psp_fld_l2_mag_RTN')
                
                mag_time_arr = mag_data[0]
                mag_data_arr = mag_data[1]
                
                """ PSP Positional data """
                pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]], datatype='ephem_eclipj2000', level='l1') #going to be used to plot parker position
                pos_data = pyt.get_data('position')

                pos_time_arr = pos_data[0]
                pos_data_arr = pos_data[1]
                
                """ PSP particle data """
                pys.psp.spi(trange=[uniq_dates[j],uniq_next[j]], datatype='spi_sf00', level='L3')
                dens_data = pyt.get_data('DENS')
                                
                dens_time_arr = dens_data[0]
                dens_data_arr = dens_data[1]
                
                idlpath = '/Users/besh2109/Desktop/psp_processed_SPANi_data/'
                idlfile = 'psp_spani_L3_rtn_sc_velocities_'+pys.time_string(date_flt[j],fmt='%Y_%m_%d')+'_00_00_00__20210614.sav'
                
                vel_raw = readsav(idlpath+idlfile)
                vel_data = vel_raw.psp_swp_spi_sf00_l3_vel_rtn
                
                vel_time_arr = vel_data[0][0]
                vel_data_arr = np.transpose(vel_data[0][1])
                
                """ date management """
                
                date_strt_flt = pys.time_float(uniq_dates[j])
                date_end_flt = pys.time_float(uniq_next[j])
                
                where = np.where(np.logical_and(wave_start>date_strt_flt,wave_start<date_end_flt))
                
                win_start_tmp = np.array(window_start[where])
                win_end_tmp = np.array(window_end[where])
                
                for k in range(len(win_start_tmp)): 
                    
                    
                    
                    """ #magnetic field data """
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
                    mag_len_arr.append(len(mag_ti))
                    
                    """ #bulk velocity data """
                    vel_where = np.where((vel_time_arr > win_start_tmp[k]) & (vel_time_arr < win_end_tmp[k]))
                    vel_where = vel_where[0]
                    
                    vel_ti = np.array(vel_time_arr[vel_where])
                    vel_r = np.array(vel_data_arr[vel_where,0])
                    vel_t = np.array(vel_data_arr[vel_where,1])
                    vel_n = np.array(vel_data_arr[vel_where,2])
    
                    vel_time.append(vel_ti)
                    vel_r_data.append(vel_r)
                    vel_t_data.append(vel_t)
                    vel_n_data.append(vel_n)
                    vel_len_arr.append(len(vel_ti))
                    
                    """ #bulk density data """
                    dens_where = np.where((dens_time_arr > win_start_tmp[k]) & (dens_time_arr < win_end_tmp[k]))
                    dens_where = dens_where[0]
                    
                    dens_ti = np.array(dens_time_arr[dens_where])
                    dens = np.array(dens_data_arr[dens_where])
                    
                    dens_time.append(dens_ti)
                    density_data.append(dens)
                    dens_len_arr.append(len(dens_ti))
                    
                    """ #position data """
                    
                    wave_center = (win_end_tmp[k]+win_start_tmp[k])/2
                    time_dif = np.array(abs(pos_time_arr-wave_center))
                    pos_where = np.where(time_dif == min(time_dif))
                    pos_where = pos_where[0][0]

                    r_km = np.sqrt(pos_data_arr[pos_where,0]**2+pos_data_arr[pos_where,1]**2+pos_data_arr[pos_where,2]**2)
                    r_m = r_km*10**3
                    r_data.append(r_m)
                    
                    r_data_mod.append(r_m-Rs_in_m)
                    
                    rho = np.sqrt(pos_data_arr[pos_where,0]**2+pos_data_arr[pos_where,1]**2)
                    parker_theta = np.arctan(pos_data_arr[pos_where,2]/rho)
                    p_theta.append(np.pi/2 - parker_theta) # in radians
            
            """
            r = np.array(r_data)/Rs      
            plt.figure(figsize=(10,7))
            plt.scatter(r,duration)
            plt.ylabel("Duration in Seconds")
            plt.yscale("Log")
            plt.xlabel("Distance in Rs")
            plt.show()
            """

            min_mag_len = min(mag_len_arr) #minimum length of mag data
            n_mag_bins = int(min_mag_len)
            
            min_vel_len = min([num for num in vel_len_arr if num != 0])
            n_vel_bins = min_vel_len
            
            mag_b_data = []
            r_b_data = []
            t_b_data = []
            n_b_data = []
            vel_v_data = []
            r_v_data = []
            t_v_data = []
            n_v_data = []
            θ_def = [] #cosine of the deflection angle
            phi_def = []
            for l in range(len(mag_time)): #this block seeks to normalize all the magnetic field data 
                                           #to the length of the shortest window
                
                bin_size = mag_len_arr[l]/n_mag_bins
                ind_arr = np.arange(mag_len_arr[l])
                norm_mag_time = np.zeros((n_mag_bins,))
                norm_mag_r_val = np.zeros((n_mag_bins,))
                norm_mag_t_val = np.zeros((n_mag_bins,))
                norm_mag_n_val = np.zeros((n_mag_bins,))
                for m in range(n_mag_bins):
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
                  
                
                b_mag = np.sqrt(norm_mag_r_val**2+norm_mag_t_val**2+norm_mag_n_val**2)
                r_b = np.array(norm_mag_r_val/b_mag)
                t_b = np.array(norm_mag_t_val/b_mag)
                n_b = np.array(norm_mag_n_val/b_mag)
                
                
                if np.median(r_b) <= 0: #control for magnetic field polarity
                    r_b = -r_b
                    t_b = -t_b
                    n_b = -n_b
                    
                    norm_mag_r_val = -norm_mag_r_val
                    norm_mag_t_val = -norm_mag_t_val
                    norm_mag_n_val = -norm_mag_n_val
                    
                mag_b_data.append(b_mag)
                r_b_data.append(r_b)#mag_r/|B|
                t_b_data.append(t_b)#mag_t/|B|
                n_b_data.append(n_b)#mag_n/|B|
                mag_time[l] = np.array(norm_mag_time)

                mag_r_data[l] =  (np.array(norm_mag_r_val)-np.median(norm_mag_r_val))#
                mag_t_data[l] =  (np.array(norm_mag_t_val)-np.median(norm_mag_t_val))#
                mag_n_data[l] =  (np.array(norm_mag_n_val)-np.median(norm_mag_n_val))#
                
                vel_bin_size = vel_len_arr[l]/n_vel_bins
                vel_ind_arr = np.arange(vel_len_arr[l])
                norm_vel_time = np.zeros((n_vel_bins,))
                norm_vel_r_val = np.zeros((n_vel_bins,))
                norm_vel_t_val = np.zeros((n_vel_bins,))
                norm_vel_n_val = np.zeros((n_vel_bins,))
                for m in range(n_vel_bins):
                    win_str = m*vel_bin_size
                    win_end = (m+1)*vel_bin_size
                    wind_where = np.where((vel_ind_arr>=win_str)&(vel_ind_arr<win_end))
                    wind_where = wind_where[0]
                    vel_time_val_tmp = np.median(vel_time[l][wind_where])
                    vel_r_val_tmp = np.median(vel_r_data[l][wind_where])
                    vel_t_val_tmp = np.median(vel_t_data[l][wind_where])
                    vel_n_val_tmp = np.median(vel_n_data[l][wind_where])
                    norm_vel_time[m] = vel_time_val_tmp
                    norm_vel_r_val[m] = vel_r_val_tmp
                    norm_vel_t_val[m] = vel_t_val_tmp
                    norm_vel_n_val[m] = vel_n_val_tmp
                
                v_mag = np.sqrt(norm_vel_r_val**2+norm_vel_t_val**2+norm_vel_n_val**2)
                r_v = np.array(norm_vel_r_val/v_mag)
                t_v = np.array(norm_vel_t_val/v_mag)
                n_v = np.array(norm_vel_n_val/v_mag)
                
                vel_v_data.append(v_mag)
                r_v_data.append(r_v)#mag_r/|B|
                t_v_data.append(t_v)#mag_t/|B|
                n_v_data.append(n_v)#mag_n/|B|
                
                vel_time[l] = np.array(norm_vel_time)                
                vel_r_data[l] = np.array(norm_vel_r_val)
                vel_t_data[l] = np.array(norm_vel_t_val)
                vel_n_data[l] = np.array(norm_vel_n_val)
                
                v = vel_r_data[l]*10**3 #m/s
                dot_term = 1/np.sqrt(1+((w/v)**2)*(r_data_mod[l])**2*np.sin(p_theta[l])**2)
                #r = 1.496*10**11
                
                phi_p = np.arctan(-w*r_data_mod[l]/v)
                phi_m = np.arctan(np.array(norm_mag_t_val)/np.array(norm_mag_r_val))
                phi_dif = np.zeros((len(r_b),))
                
                cos_θ = np.zeros((len(r_b),))
                norm_vel_r_val = np.zeros((n_vel_bins,))
                vel_ind_arr = np.arange(vel_len_arr[l])
                cos_bin_size = len(r_b)/len(v)
                
                
                #print(v[0])
                for n in range(len(v)):
                    win_str = math.ceil(n*cos_bin_size)
                    win_end = math.ceil((n+1)*cos_bin_size)
                    
                    #print(win_str,win_end)
                    phi_dif[win_str:win_end] = phi_m[win_str:win_end] - phi_p[n]
                    
                    dot_term = 1/np.sqrt(1+((w/v[n])**2)*(r_data_mod[l])**2*np.sin(p_theta[l])**2)
                    cos_θ[win_str:win_end] = r_b[win_str:win_end]*dot_term - t_b[win_str:win_end]*dot_term*w*r_data_mod[l]/v[n]*np.sin(p_theta[l])
                    
                #cos_θ = np.array(r_b*dot_term - t_b*dot_term*w*r_data[l]/v*np.sin(p_theta[l]))

                θ_def.append(np.arccos(cos_θ)*180/np.pi)
                phi_def.append(phi_dif*180/np.pi)

            if plot =='epoch' or plot == 'both':
                
                savepath = '/Users/besh2109/Desktop/PSP_epoch/mag_epoch/'
                
                if not by_Rs:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_mag_epoch.png'
                        else:
                            savename = 'All_Events_mag_epoch.png'       
                    else:
                        savename = 'Enc_'+str(i)+'_mag_epoch.png'
                else:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_mag_epoch.png'
                        else:
                            savename = 'All_Events_mag_epoch.png'       
                    else:
                        savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs_mag_epoch.png'
                
                fis1 = plt.figure(figsize=(15,15))
                fis1.suptitle('Magnetic Field Unit Vector '+name, fontsize=16,y=0.92)
                axs1 = fis1.add_subplot(411)
                
                ind_hist = []
                data_hist = []
                for o in r_b_data:
                    ind = list(range(len(o)))
                    data = list(o)
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs1.plot(o)
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                            
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_mag_bins,40])
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
    
                r_color = axs1.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                
                #height = [-0.6,-0.6]
                #endpoints = [n_mag_bins/3,2*n_mag_bins/3]
                #axs1.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_r_data)+1)
                axs1.set(title='mag_r, normalized time')
                #plt.title()
                #plt.show()
                box = axs1.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(r_color,cax=axColor, label='Counts')
                #fis2 = plt.figure(figsize=(15,10))
                axs2 = fis1.add_subplot(412)
                
                ind_hist = []
                data_hist = []
                for o in t_b_data:
                    ind = list(range(len(o)))
                    data = list(o)
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs2.plot(o)
                
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_mag_bins,40])
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
    
                t_color = axs2.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                
                box = axs2.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(t_color,cax=axColor, label='Counts')
                
                #height = [0,0]
                #endpoints = [n_vel_bins/3,2*n_vel_bins/3]
                #axs2.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_t_data)+1)
                axs2.set(title='mag_t, normalized time')
                #plt.title('mag_t first 73 waves, normalized time')
                #plt.show()
                
                #fis3 = plt.figure(figsize=(15,10))
                axs3 = fis1.add_subplot(413)
                
                ind_hist = []
                data_hist = []
                for o in n_b_data:
                    ind = list(range(len(o)))
                    data = list(o)
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs2.plot(o)
                
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist) 
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_mag_bins,40]) 
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
    
                n_color = axs3.pcolormesh(xedge,yedge,histo,cmap='jet')  #,np.log10(histo)
                
                box = axs3.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(n_color,cax=axColor, label='Counts')
                
                #height = [0,0]
                #endpoints = [n_vel_bins/3,2*n_vel_bins/3]
                #axs3.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_n_data)+1)
                axs3.set(title='mag_n, normalized time')
                #plt.title('mag_n first 73 waves, normalized time')
                
                axs4 = fis1.add_subplot(414)
                
                ind_hist = []
                data_hist = []
                for o in phi_def:
                    ind = list(range(len(o)))
                    data = list(o)#np.arccos(o)) #convert to deflection angle
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs2.plot(o)
                
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist) 
                
                #axs4.scatter(ind_hist_arr,data_hist_arr,s=1)
                #plt.show()
                #yes
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_mag_bins,40]) 
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
                
                n_color = axs4.pcolormesh(xedge,yedge,histo,cmap='jet')  #,np.log10(histo)
                
                box = axs4.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(n_color,cax=axColor, label='Counts')
                
                #height = [0,0]
                #endpoints = [n_mag_bins/3+30,2*n_mag_bins/3-30]
                #axs4.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_n_data)+1)
                axs4.set_ylim([-80,80])
                axs4.set(title='Deflection Angle θ, normalized time')
                
                plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                plt.clf()
                plt.cla()
                plt.close('all')
                plt.close(fis1)
            
            if plot == 'dist' or plot == 'both':
                
                savepath = '/Users/besh2109/Desktop/PSP_epoch/mag_dist/'
                
                if not by_Rs:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_mag_dist.png'
                        else:
                            savename = 'All_Events_mag_dist.png'      
                    else:
                        savename = 'Enc_'+str(i)+'_mag_dist.png'
                else:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_mag_dist.png'
                        else:
                            savename = 'All_Events_mag_dist.png'      
                    else:
                        savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs_mag_dist.png'
                
                fig1 = plt.figure(figsize=(15,15))
                fig1.suptitle('Magnetic Field Unit Vector Distributions '+name, fontsize=16,y=0.92)
                ax1 = fig1.add_subplot(221)
                ax2 = fig1.add_subplot(222)
                ax3 = fig1.add_subplot(223)
                ax4 = fig1.add_subplot(224)
                
                ind_hist = []
                data_hist = []
                for o in r_b_data:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_mag_bins,125])
                histo = np.transpose(histo)

                k=0
                dist1r = np.zeros(histo[:,0].shape)
                while k < round(n_mag_bins/3): #distribution 1
                    dist1r = dist1r + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_mag_bins/3)
                dist2r = np.zeros(histo[:,0].shape)
                while k >= round(n_mag_bins/3) and k < round(2*n_mag_bins/3): #distribution 2
                    dist2r = dist2r + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_mag_bins/3)
                dist3r = np.zeros(histo[:,0].shape)
                while k >= round(2*n_mag_bins/3) and k<n_mag_bins: #distribution 3
                    dist3r = dist3r + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax1.set_aspect(1)
                #ax1.set_adjustable('box')
                norm_val = np.max(dist2r)
                ax1.plot(yedge[1:len(yedge)],dist1r/norm_val,color='red',label='before wave')
                ax1.plot(yedge[1:len(yedge)],dist2r/norm_val,color='green',label='during wave')
                ax1.plot(yedge[1:len(yedge)],dist3r/norm_val,color='blue',label='after wave')
                ax1.legend()
                #ax1.set_ylim([0,120])
                ax1.set_title('r-hat')
                ax1.set_ylabel('Normalized Counts Distribution')
                ax1.set_xlabel('Br/|B|')
                ax1.set_adjustable('box')
                #ax1.set_xlim(-25,25)
                
                #ax2 = fig1.add_subplot(2,2,(1,2))
                
                ind_hist = []
                data_hist = []
                for o in t_b_data:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_mag_bins,125])
                histo = np.transpose(histo)
                
                

                k=0
                dist1t = np.zeros(histo[:,0].shape)
                while k < round(n_mag_bins/3): #distribution 1
                    dist1t = dist1t + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_mag_bins/3)
                dist2t = np.zeros(histo[:,0].shape)
                while k >= round(n_mag_bins/3) and k < round(2*n_mag_bins/3): #distribution 2
                    dist2t = dist2t + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_mag_bins/3)
                dist3t = np.zeros(histo[:,0].shape)
                while k >= round(2*n_mag_bins/3) and k<n_mag_bins: #distribution 3
                    dist3t = dist3t + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax2.set_aspect(1)
                #ax2.set_adjustable('box')
                norm_val = np.max(dist2t)
                ax2.plot(yedge[1:len(yedge)],dist1t/norm_val,color='red',label='before wave')
                ax2.plot(yedge[1:len(yedge)],dist2t/norm_val,color='green',label='during wave')
                ax2.plot(yedge[1:len(yedge)],dist3t/norm_val,color='blue',label='after wave')
                ax2.legend()
                #ax2.set_ylim([0,26])
                ax2.set_title('t-hat')
                ax2.set_ylabel('Normalized Counts Distribution')
                ax2.set_xlabel('Bt/|B|')
                ax2.set_adjustable('box')
                
                #ax3 = fig1.add_subplot(2,2,(2,1))
                
                ind_hist = []
                data_hist = []
                for o in n_b_data:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_mag_bins,125])
                histo = np.transpose(histo)

                k=0
                dist1n = np.zeros(histo[:,0].shape)
                while k < round(n_mag_bins/3): #distribution 1
                    dist1n = dist1n + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_mag_bins/3)
                dist2n = np.zeros(histo[:,0].shape)
                while k >= round(n_mag_bins/3) and k < round(2*n_mag_bins/3): #distribution 2
                    dist2n = dist2n + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_mag_bins/3)
                dist3n = np.zeros(histo[:,0].shape)
                while k >= round(2*n_mag_bins/3) and k<n_mag_bins: #distribution 3
                    dist3n = dist3n + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax3.set_aspect(1)
                #ax3.set_adjustable('box')
                norm_val = np.max(dist2n)
                ax3.plot(yedge[1:len(yedge)],dist1n/norm_val,color='red',label='before wave')
                ax3.plot(yedge[1:len(yedge)],dist2n/norm_val,color='green',label='during wave')
                ax3.plot(yedge[1:len(yedge)],dist3n/norm_val,color='blue',label='after wave')
                ax3.legend()
                #ax3.set_ylim([0,34])
                ax3.set_title('n-hat')
                ax3.set_ylabel('Normalized Counts Distribution')
                ax3.set_xlabel('Bn/|B|')
                ax3.set_adjustable('box')
                
                #ax4 = fig1.add_subplot(2,2,(2,2))
                
                ind_hist = []
                data_hist = []
                for o in phi_def:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_mag_bins,125])
                histo = np.transpose(histo)

                k=0
                dist1cos = np.zeros(histo[:,0].shape)
                while k < round(n_mag_bins/3): #distribution 1
                    dist1cos = dist1cos + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_mag_bins/3)
                dist2cos = np.zeros(histo[:,0].shape)
                while k >= round(n_mag_bins/3) and k < round(2*n_mag_bins/3): #distribution 2
                    dist2cos = dist2cos + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_mag_bins/3)
                dist3cos = np.zeros(histo[:,0].shape)
                while k >= round(2*n_mag_bins/3) and k<n_mag_bins: #distribution 3
                    dist3cos = dist3cos + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax3.set_aspect(1)
                #ax3.set_adjustable('box')
                norm_val = np.max(dist2cos)
                ax4.plot(yedge[1:len(yedge)],dist1cos/norm_val,color='red',label='before wave')
                ax4.plot(yedge[1:len(yedge)],dist2cos/norm_val,color='green',label='during wave')
                ax4.plot(yedge[1:len(yedge)],dist3cos/norm_val,color='blue',label='after wave')
                ax4.legend()
                ax4.set_xlim([-40,40])
                #ax4.set_ylim([0,50])
                ax4.set_title('Parker-Measurement Deflection Angle')
                ax4.set_ylabel('Normalized Counts Distribution')
                ax4.set_xlabel('θ degrees')
                ax4.set_adjustable('box')
                
                
                plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                plt.clf()
                plt.cla()
                plt.close('all')
                plt.close(fig1)
            #plt.savefig(savepath+savename)
            
        i+=1
        
def vel_epoch(plot='epoch',no_enc_7=False, win_len=1, by_Rs=False,resolution=10):

    i=0
    enc_num = len(per_flt)
    while i <= enc_num:
        
        Rs = 6.957e5 #solar radius in km
        w = 2*np.pi/(25.38*86400) # angular frequency of the sun in degrees/sec
        
        v = 400000 #m/s typical slow solar wind speed, may replace later with measurement values
        
        
        if not by_Rs:
            if i==0:
                csv_filename = 'harmwave_master_arch.csv'
                csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
                if no_enc_7:
                    name = 'All Encounters sans 7'
                else:
                    name = 'All Encounters'
            else:
                csv_filename = 'enc_'+str(i)+'_harmwave_arch.csv'
                csv_path = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/Enc'+str(i)+'/'
                savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/histograms/'
                savename = 'Enc_'+str(i)+'_mag_epoch.png'
                name = 'Encounter '+str(i)
        
        else:
            csv_filename = 'harmwave_master_arch.csv'
            csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
            if i==0:
                if no_enc_7:
                    name = 'All Radial Distances sans Enc 7'
                else:
                    name = 'All Radial Distances'
            else:
                name = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+' Rs'
        
        isfile = os.path.isfile(csv_path+csv_filename)
        
        if isfile:
            
            df = pd.read_csv(csv_path+csv_filename)
            bf = df.to_numpy()

            af = np.delete(bf,bf[:,1]<90,0)
            
            if by_Rs and i !=0:
                af = np.delete(af,af[:,2]>Rs_grps[i-1][0],0)
                af = np.delete(af,af[:,2]<Rs_grps[i-1][1],0)
            
            if no_enc_7 and i != enc_num: #this portion of code kills events in encounter 7
                dates = list(af[:,0])
                for j in range(len(dates)):
                    dates[j] = dates[j][0:4]
                npdates = np.array(dates)
                af = np.delete(af,npdates=='2021',0)
            
            wave_start = np.array(pys.time_float(af[:,0]))
            wave_end = np.array(wave_start+af[:,1])
            
            window_start = np.array(wave_start - win_len*af[:,1])#*(1/3)) #complete epoch analysis for window larger than wave itself
            window_end = np.array(wave_end + win_len*af[:,1])#*(1/3))     #want to see if theres a difference between times during and before/after events
            
            dates = list(af[:,0])
            for j in range(len(dates)):
                dates[j] = dates[j][0:10]

            uniq_dates = np.unique(dates)
            date_flt = pys.time_float(uniq_dates)
            uniq_next = pys.time_string(np.array(date_flt)+86400.)
            
            duration = np.array(af[:,1])
            
            vel_time = []
            vel_r_data = []
            vel_t_data = []
            vel_n_data = []
            vel_mag_data = []
            
            vel_r_med = []
            vel_t_med = []
            vel_n_med = []
            vel_mag_med = []
            
            r_wave_med = []
            t_wave_med = []
            n_wave_med = []
            mag_wave_med = []
            
            dens_time = []
            density_data = []
            
            r_data = []
            
            vel_len_arr = []
            dens_len_arr = []
            
            for j in range(len(uniq_dates)): #gather data for each day range(2):#
                
                """ PSP Positional data """
                pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]], datatype='ephem_eclipj2000', level='l1') #going to be used to plot parker position
                pos_data = pyt.get_data('position')

                pos_time_arr = pos_data[0]
                pos_data_arr = pos_data[1]
                
                """ PSP particle data """
                pys.psp.spi(trange=[uniq_dates[j],uniq_next[j]], datatype='spi_sf00', level='L3')
                dens_data = pyt.get_data('DENS')
                                
                dens_time_arr = dens_data[0]
                dens_data_arr = dens_data[1]
                
                idlpath = '/Users/besh2109/Desktop/psp_processed_SPANi_data/'
                idlfile = 'psp_spani_L3_rtn_sc_velocities_'+pys.time_string(date_flt[j],fmt='%Y_%m_%d')+'_00_00_00__20210614.sav'
                
                vel_raw = readsav(idlpath+idlfile)
                vel_data = vel_raw.psp_swp_spi_sf00_l3_vel_rtn
                
                vel_time_arr = vel_data[0][0]
                vel_data_arr = np.transpose(vel_data[0][1])
                
                """ date management """
                
                date_strt_flt = pys.time_float(uniq_dates[j])
                date_end_flt = pys.time_float(uniq_next[j])
                
                where = np.where(np.logical_and(wave_start>date_strt_flt,wave_start<date_end_flt))
                
                win_start_tmp = np.array(window_start[where])
                win_end_tmp = np.array(window_end[where])
                
                wave_start_tmp = np.array(wave_start[where])
                wave_end_tmp = np.array(wave_end[where])
                
                for k in range(len(win_start_tmp)): 
                    
                    
                    
                    """ #bulk velocity data """
                    vel_where = np.where((vel_time_arr > win_start_tmp[k]) & (vel_time_arr < win_end_tmp[k]))
                    vel_where = vel_where[0]
                    
                    vel_ti = np.array(vel_time_arr[vel_where])
                    vel_r = np.array(vel_data_arr[vel_where,0])
                    vel_t = np.array(vel_data_arr[vel_where,1])
                    vel_n = np.array(vel_data_arr[vel_where,2])
    
                    vel_time.append(vel_ti)
                    vel_r_data.append(vel_r)
                    vel_t_data.append(vel_t)
                    vel_n_data.append(vel_n)
                    vel_mag_data.append(np.sqrt(vel_data_arr[vel_where,0]**2+vel_data_arr[vel_where,1]**2+vel_data_arr[vel_where,2]**2))
                    vel_len_arr.append(len(vel_ti))
                    
                    wave_where = np.where((vel_time_arr > wave_start_tmp[k]) & (vel_time_arr < wave_end_tmp[k]))
                    wave_where = wave_where[0]
                    
                    r_wave_med.append(np.median(vel_data_arr[wave_where,0]))
                    t_wave_med.append(np.median(vel_data_arr[wave_where,1]))
                    n_wave_med.append(np.median(vel_data_arr[wave_where,2]))
                    mag_wave_med.append(np.median(np.sqrt(vel_data_arr[wave_where,0]**2+vel_data_arr[wave_where,1]**2+vel_data_arr[wave_where,2]**2)))
                    
                    
                    """ #bulk density data """
                    dens_where = np.where((dens_time_arr > win_start_tmp[k]) & (dens_time_arr < win_end_tmp[k]))
                    dens_where = dens_where[0]
                    
                    dens_ti = np.array(dens_time_arr[dens_where])
                    dens = np.array(dens_data_arr[dens_where])
                    
                    dens_time.append(dens_ti)
                    density_data.append(dens)
                    dens_len_arr.append(len(dens_ti))
                    
                    """ #position data """
                    
                    wave_center = (win_end_tmp[k]+win_start_tmp[k])/2
                    time_dif = np.array(abs(pos_time_arr-wave_center))
                    pos_where = np.where(time_dif == min(time_dif))
                    pos_where = pos_where[0][0]

                    r_km = np.sqrt(pos_data_arr[pos_where,0]**2+pos_data_arr[pos_where,1]**2+pos_data_arr[pos_where,2]**2)
                    r_data.append(r_km)
            
            
            vel_time_tmp = []
            vel_r_tmp = []
            vel_t_tmp = []
            vel_n_tmp = []
            vel_mag_tmp = []
            vel_len_tmp = []
            vel_med_tmp = []
            for m in range(len(vel_len_arr)):
                if vel_len_arr[m] >= resolution:
                    vel_time_tmp.append(vel_time[m])
                    vel_r_tmp.append(vel_r_data[m])
                    vel_t_tmp.append(vel_t_data[m])
                    vel_n_tmp.append(vel_n_data[m])
                    vel_mag_tmp.append(vel_mag_data[m])
                    vel_len_tmp.append(vel_len_arr[m])
            
            vel_time = list(vel_time_tmp)
            vel_r_data = list(vel_r_tmp)
            vel_t_data = list(vel_t_tmp)
            vel_n_data = list(vel_n_tmp)
            vel_len_arr = np.array(vel_len_tmp)
            
            min_vel_len = min([num for num in vel_len_arr if num != 0])
            n_vel_bins = min_vel_len

            vel_v_data = []
            r_v_data = []
            t_v_data = []
            n_v_data = []
            
            for l in range(len(vel_time)): #this block seeks to normalize all the magnetic field data 
                                           #to the length of the shortest window

                vel_bin_size = vel_len_arr[l]/n_vel_bins
                vel_ind_arr = np.arange(vel_len_arr[l])
                norm_vel_time = np.zeros((n_vel_bins,))
                norm_vel_r_val = np.zeros((n_vel_bins,))
                norm_vel_t_val = np.zeros((n_vel_bins,))
                norm_vel_n_val = np.zeros((n_vel_bins,))
                for m in range(n_vel_bins):
                    win_str = m*vel_bin_size
                    win_end = (m+1)*vel_bin_size
                    wind_where = np.where((vel_ind_arr>=win_str)&(vel_ind_arr<win_end))
                    wind_where = wind_where[0]
                    vel_time_val_tmp = np.median(vel_time[l][wind_where])
                    vel_r_val_tmp = np.median(vel_r_data[l][wind_where])
                    vel_t_val_tmp = np.median(vel_t_data[l][wind_where])
                    vel_n_val_tmp = np.median(vel_n_data[l][wind_where])
                    norm_vel_time[m] = vel_time_val_tmp
                    norm_vel_r_val[m] = vel_r_val_tmp
                    norm_vel_t_val[m] = vel_t_val_tmp
                    norm_vel_n_val[m] = vel_n_val_tmp
                
                v_mag = np.sqrt(norm_vel_r_val**2+norm_vel_t_val**2+norm_vel_n_val**2)
                r_v = np.array(norm_vel_r_val/v_mag)
                t_v = np.array(norm_vel_t_val/v_mag)
                n_v = np.array(norm_vel_n_val/v_mag)
                
                vel_v_data.append(v_mag)
                r_v_data.append(r_v)#mag_r/|B|
                t_v_data.append(t_v)#mag_t/|B|
                n_v_data.append(n_v)#mag_n/|B|
                
                vel_time[l] = np.array(norm_vel_time)                
                vel_r_data[l] = np.array(norm_vel_r_val)
                vel_t_data[l] = np.array(norm_vel_t_val)
                vel_n_data[l] = np.array(norm_vel_n_val)
                
                vel_r_med.append(np.array(norm_vel_r_val-r_wave_med[l]))
                vel_t_med.append(np.array(norm_vel_t_val-t_wave_med[l]))
                vel_n_med.append(np.array(norm_vel_n_val-n_wave_med[l]))
                vel_mag_med.append(np.array(v_mag-mag_wave_med[l]))
                
            
            if plot=='epoch' or plot=='both':
            
                savepath = '/Users/besh2109/Desktop/PSP_epoch/vel_epoch/'
                    
                if not by_Rs:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_vel_epoch.png'
                        else:
                            savename = 'All_Events_vel_epoch.png'      
                    else:
                        savename = 'Enc_'+str(i)+'_vel_epoch.png'
                else:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_vel_epoch.png'
                        else:
                            savename = 'All_Events_vel_epoch.png'       
                    else:
                        savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs_vel_epoch.png'
                    
                fis1 = plt.figure(figsize=(15,15))
                axs1 = fis1.add_subplot(411)
                
                ind_hist = []
                data_hist = []
                for o in vel_r_data: #vel_r_data
                    ind = list(range(len(o)))
                    data = list(o)
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs1.plot(o)
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                            
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_vel_bins,75])
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
    
                r_color = axs1.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                
                #height = [-0.6,-0.6]
                #endpoints = [n_vel_bins/3,2*n_vel_bins/3]
                #axs1.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_r_data)+1)
                axs1.set(title='vel_r, SPAN-I, normalized time')
                #plt.title()
                #plt.show()
                box = axs1.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(r_color,cax=axColor, label='Counts')
                #fis2 = plt.figure(figsize=(15,10))
                axs2 = fis1.add_subplot(412)
                
                ind_hist = []
                data_hist = []
                for o in vel_t_data: #vel_t_data
                    ind = list(range(len(o)))
                    data = list(o)
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs2.plot(o)
    
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_vel_bins,75])
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
    
                t_color = axs2.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                
                box = axs2.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(t_color,cax=axColor, label='Counts')
                
                #height = [0,0]
                #endpoints = [n_vel_bins/3,2*n_vel_bins/3]
                #axs2.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_t_data)+1)
                axs2.set(title='vel_t, SPAN-I, normalized time')
                #plt.title('mag_t first 73 waves, normalized time')
                #plt.show()
                
                #fis3 = plt.figure(figsize=(15,10))
                axs3 = fis1.add_subplot(413)
                
                ind_hist = []
                data_hist = []
                for o in vel_n_data: #vel_n_data
                    ind = list(range(len(o)))
                    data = list(o)
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs2.plot(o)
                
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist) 
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_vel_bins,75]) 
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
    
                n_color = axs3.pcolormesh(xedge,yedge,histo,cmap='jet')  #,np.log10(histo)
                
                box = axs3.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(n_color,cax=axColor, label='Counts')
                
                #height = [0,0]
                #endpoints = [n_vel_bins/3,2*n_vel_bins/3]
                #axs3.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_n_data)+1)
                axs3.set(title='vel_n, SPAN-I, normalized time')
                #plt.title('mag_n first 73 waves, normalized time')
                
                axs4 = fis1.add_subplot(414)
                
                ind_hist = []
                data_hist = []
                for o in vel_v_data:
                    ind = list(range(len(o)))
                    data = list(o)#np.arccos(o)) #convert to deflection angle
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs2.plot(o)
                
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist) 
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_vel_bins,75]) 
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
    
                n_color = axs4.pcolormesh(xedge,yedge,histo,cmap='jet')  #,np.log10(histo)
                
                box = axs4.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(n_color,cax=axColor, label='Counts')
                
                #height = [0,0]
                #endpoints = [n_mag_bins/3+30,2*n_mag_bins/3-30]
                #axs4.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_n_data)+1)
                axs4.set(title='v_mag, SPAN-I, normalized time')
                
                plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                plt.clf()
                plt.cla()
                plt.close('all')
                plt.close(fis1)
                #plt.savefig(savepath+savename)
                
            if plot == 'dist' or plot == 'both':
                
                savepath = '/Users/besh2109/Desktop/PSP_epoch/vel_dist/'
                    
                if not by_Rs:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_vel_dist.png'
                        else:
                            savename = 'All_Events_vel_dist.png'      
                    else:
                        savename = 'Enc_'+str(i)+'_vel_dist.png'
                else:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_vel_dist.png'
                        else:
                            savename = 'All_Events_vel_dist.png'     
                    else:
                        savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs_vel_dist.png'
                
                fig1 = plt.figure(figsize=(15,15))
                fig1.suptitle('Velocity components minus median value '+name, fontsize=16,y=0.92)
                ax1 = fig1.add_subplot(221)
                ax2 = fig1.add_subplot(222)
                ax3 = fig1.add_subplot(223)
                ax4 = fig1.add_subplot(224)
                
                ind_hist = []
                data_hist = []
                for o in vel_r_med:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_vel_bins,125])
                histo = np.transpose(histo)

                k=0
                dist1r = np.zeros(histo[:,0].shape)
                while k < round(n_vel_bins/3): #distribution 1
                    dist1r = dist1r + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_vel_bins/3)
                dist2r = np.zeros(histo[:,0].shape)
                while k >= round(n_vel_bins/3) and k < round(2*n_vel_bins/3): #distribution 2
                    dist2r = dist2r + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_vel_bins/3)
                dist3r = np.zeros(histo[:,0].shape)
                while k >= round(2*n_vel_bins/3) and k<n_vel_bins: #distribution 3
                    dist3r = dist3r + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax1.set_aspect(1)
                #ax1.set_adjustable('box')
                norm_val = np.max(dist2r)
                ax1.plot(yedge[1:len(yedge)],dist1r/norm_val,color='red',label='before wave')
                ax1.plot(yedge[1:len(yedge)],dist2r/norm_val,color='green',label='during wave')
                ax1.plot(yedge[1:len(yedge)],dist3r/norm_val,color='blue',label='after wave')
                ax1.legend()
                #ax1.set_ylim([0,120])
                ax1.set_xlim([-50,50])
                ax1.set_title('r-component of velocity')
                ax1.set_ylabel('Normalized Counts Distribution')
                ax1.set_xlabel('km/s')
                ax1.set_adjustable('box')
                #ax1.set_xlim(-25,25)
                
                #ax2 = fig1.add_subplot(2,2,(1,2))
                
                ind_hist = []
                data_hist = []
                for o in vel_t_med:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_vel_bins,125])
                histo = np.transpose(histo)
                
                

                k=0
                dist1t = np.zeros(histo[:,0].shape)
                while k < round(n_vel_bins/3): #distribution 1
                    dist1t = dist1t + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_vel_bins/3)
                dist2t = np.zeros(histo[:,0].shape)
                while k >= round(n_vel_bins/3) and k < round(2*n_vel_bins/3): #distribution 2
                    dist2t = dist2t + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_vel_bins/3)
                dist3t = np.zeros(histo[:,0].shape)
                while k >= round(2*n_vel_bins/3) and k<n_vel_bins: #distribution 3
                    dist3t = dist3t + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax2.set_aspect(1)
                #ax2.set_adjustable('box')
                norm_val = np.max(dist2t)
                ax2.plot(yedge[1:len(yedge)],dist1t/norm_val,color='red',label='before wave')
                ax2.plot(yedge[1:len(yedge)],dist2t/norm_val,color='green',label='during wave')
                ax2.plot(yedge[1:len(yedge)],dist3t/norm_val,color='blue',label='after wave')
                ax2.legend()
                #ax2.set_ylim([0,26])
                ax2.set_xlim([-50,50])
                ax2.set_title('t-component of velocity')
                ax2.set_ylabel('Normalized Counts Distribution')
                ax2.set_xlabel('km/s|')
                ax2.set_adjustable('box')
                
                #ax3 = fig1.add_subplot(2,2,(2,1))
                
                ind_hist = []
                data_hist = []
                for o in vel_n_med:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_vel_bins,125])
                histo = np.transpose(histo)

                k=0
                dist1n = np.zeros(histo[:,0].shape)
                while k < round(n_vel_bins/3): #distribution 1
                    dist1n = dist1n + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_vel_bins/3)
                dist2n = np.zeros(histo[:,0].shape)
                while k >= round(n_vel_bins/3) and k < round(2*n_vel_bins/3): #distribution 2
                    dist2n = dist2n + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_vel_bins/3)
                dist3n = np.zeros(histo[:,0].shape)
                while k >= round(2*n_vel_bins/3) and k<n_vel_bins: #distribution 3
                    dist3n = dist3n + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax3.set_aspect(1)
                #ax3.set_adjustable('box')
                norm_val = np.max(dist2n)
                ax3.plot(yedge[1:len(yedge)],dist1n/norm_val,color='red',label='before wave')
                ax3.plot(yedge[1:len(yedge)],dist2n/norm_val,color='green',label='during wave')
                ax3.plot(yedge[1:len(yedge)],dist3n/norm_val,color='blue',label='after wave')
                ax3.legend()
                ax3.set_xlim([-50,50])
                #ax3.set_ylim([0,34])
                ax3.set_title('n-component of velocity')
                ax3.set_ylabel('Normalized Counts Distribution')
                ax3.set_xlabel('km/s')
                ax3.set_adjustable('box')
                
                #ax4 = fig1.add_subplot(2,2,(2,2))
                
                ind_hist = []
                data_hist = []
                for o in vel_mag_med:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_vel_bins,125])
                histo = np.transpose(histo)

                k=0
                dist1cos = np.zeros(histo[:,0].shape)
                while k < round(n_vel_bins/3): #distribution 1
                    dist1cos = dist1cos + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_vel_bins/3)
                dist2cos = np.zeros(histo[:,0].shape)
                while k >= round(n_vel_bins/3) and k < round(2*n_vel_bins/3): #distribution 2
                    dist2cos = dist2cos + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_vel_bins/3)
                dist3cos = np.zeros(histo[:,0].shape)
                while k >= round(2*n_vel_bins/3) and k<n_vel_bins: #distribution 3
                    dist3cos = dist3cos + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax3.set_aspect(1)
                #ax3.set_adjustable('box')
                norm_val = np.max(dist2cos)
                ax4.plot(yedge[1:len(yedge)],dist1cos/norm_val,color='red',label='before wave')
                ax4.plot(yedge[1:len(yedge)],dist2cos/norm_val,color='green',label='during wave')
                ax4.plot(yedge[1:len(yedge)],dist3cos/norm_val,color='blue',label='after wave')
                ax4.legend()
                ax4.set_xlim([-50,50])
                #ax4.set_ylim([0,50])
                ax4.set_title('Velocity Magnitude minus median value')
                ax4.set_ylabel('Normalized Counts Distribution')
                ax4.set_xlabel('km/s')
                ax4.set_adjustable('box')
                
                
                plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                plt.clf()
                plt.cla()
                plt.close('all')
                plt.close(fig1)
            
        i+=1

def spec_epoch(plot='epoch',no_enc_7=False, win_len=1, by_Rs=False, norm_freq='fce'):

    i=0
    enc_num = len(per_flt)
    while i <= enc_num:
        
        
        q = 1.60218*10e-20 #Coulombs
        me =  9.10938*10e-32 #kg
        mi = 1.6726219*10e-27 #kg
        Rs = 6.957e5 #solar radius in km
        eps = 8.8541878128*10e-12 # epsilon naught, s^2 C^2/m^3 kg

        w = 2*np.pi/(25.38*86400) # angular frequency of the sun in degrees/sec
        v = 400000 #m/s typical slow solar wind speed, may replace later with measurement values

        

        if not by_Rs:
            if i==0:
                csv_filename = 'harmwave_master_arch.csv'
                csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
                if no_enc_7:
                    name = 'All Encounters sans 7'
                else:
                    name = 'All Encounters'
            else:
                csv_filename = 'enc_'+str(i)+'_harmwave_arch.csv'
                csv_path = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/Enc'+str(i)+'/'
                savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/histograms/'
                savename = 'Enc_'+str(i)+'_mag_epoch.png'
                name = 'Encounter '+str(i)
        
        else:
            csv_filename = 'harmwave_master_arch.csv'
            csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
            if i==0:
                if no_enc_7:
                    name = 'All Radial Distances sans Enc 7'
                else:
                    name = 'All Radial Distances'
            else:
                name = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+' Rs'
                
        isfile = os.path.isfile(csv_path+csv_filename)
        
        if isfile:
            
            df = pd.read_csv(csv_path+csv_filename)
            bf = df.to_numpy()

            af = np.delete(bf,bf[:,1]<90,0)
            
            if by_Rs and i !=0:
                af = np.delete(af,af[:,2]>Rs_grps[i-1][0],0)
                af = np.delete(af,af[:,2]<Rs_grps[i-1][1],0)
            
            if no_enc_7 and i != enc_num: #this portion of code kills events in encounter 7
                dates = list(af[:,0])
                for j in range(len(dates)):
                    dates[j] = dates[j][0:4]
                npdates = np.array(dates)
                af = np.delete(af,npdates=='2021',0)
                
            
            wave_start = np.array(pys.time_float(af[:,0]))
            wave_end = np.array(wave_start+af[:,1])
            
            window_start = np.array(wave_start - win_len*af[:,1])#*(1/3)) #complete epoch analysis for window larger than wave itself
            window_end = np.array(wave_end + win_len*af[:,1])#*(1/3))     #want to see if theres a difference between times during and before/after events
            
            dates = list(af[:,0])
            for j in range(len(dates)):
                dates[j] = dates[j][0:10]

            uniq_dates = np.unique(dates)
            date_flt = pys.time_float(uniq_dates)
            uniq_next = pys.time_string(np.array(date_flt)+86400.+86400./4)
            
            duration = np.array(af[:,1])
            
            ac12_time = []
            ac12_spec = []
            ac12_freq = []
            
            bef_max_data = []
            duri_max_data = []
            after_max_data = []
            
            ac34_time = []
            ac34_spec = []
            ac34_freq = []   
            
            r_km_data = []
            r_m_data = []
            fce_data = []
            f_fce_data = []
            f_fce_max_data = []
            
            fpi_data = []
            f_fpi_data = []
            
            ac12_len_arr = []
            iso_waves = [] #dates of isolated waves
            iso_duration = [] #duration of isolated waves
            #ac34_len_arr = []
            
            for j in range(len(uniq_dates)): #gather data for each day range(2):#
                
                """ PSP Positional data """
                pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]], datatype='ephem_eclipj2000', level='l1') #going to be used to plot parker position
                pos_data = pyt.get_data('position')

                pos_time_arr = pos_data[0]
                pos_data_arr = pos_data[1]
                
                """ PSP spectral data """
                pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]], datatype='dfb_ac_spec', level='l2')
                ac12_data = pyt.get_data('psp_fld_l2_dfb_ac_spec_dV12hg')
                
                ac12_time_arr = ac12_data[0]
                ac12_tmp = ac12_data[1]
                ac12_tmp[ac12_tmp==0] = np.nan
                ac12_data_arr = ac12_tmp
            
                for f in range(len(ac12_tmp[0,:])):
                    noise = median(ac12_tmp[:,f])
                    ac12_data_arr[:,f] = 10*np.log10(ac12_tmp[:,f]/noise) # np.array(ac12_tmp[:,f]/noise)#
            
                ac12_data_arr = np.transpose(ac12_data_arr)
                ac12_freq_arr = np.transpose(ac12_data[2])
                ac12_freq_lst = ac12_freq_arr[:,0]
                
                """ PSP density data """
                
                pys.psp.spi(trange=[uniq_dates[j],uniq_next[j]], datatype='spi_sf00', level='L3')
                dens_data = pyt.get_data('DENS')
                
                dens_time_arr = dens_data[0]
                dens_data_arr = dens_data[1]
                
                """ PSP low res data """
                pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]],datatype='mag_RTN_1min',level='l2')
                mag_data = pyt.get_data('psp_fld_l2_mag_RTN_1min')
                
                mag_time_arr = mag_data[0]
                mag_data_arr = mag_data[1]
                
                
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

                    """ spectral data """
                    ac12_where = np.where((ac12_time_arr > win_start_tmp[k]) & (ac12_time_arr < win_end_tmp[k]))
                    ac12_where = ac12_where[0]
                    
                    ac12_ti = np.array(ac12_time_arr[ac12_where])
                    ac12_data_tmp = np.array(ac12_data_arr[:,ac12_where])
                    ac12_freq_tmp = np.array(ac12_freq_arr[:,ac12_where[0]])
                    
                    ac12_time.append(ac12_ti)
                    ac12_spec.append(ac12_data_tmp)
                    ac12_freq.append(ac12_freq_tmp)
                    ac12_len_arr.append(len(ac12_ti))

                    before_where = np.where((ac12_time_arr > win_start_tmp[k]) & (ac12_time_arr < wv_start[k]))
                    before_where = before_where[0]
                    before_data = np.array(ac12_data_arr[:,before_where])
                    
                    duri_where = np.where((ac12_time_arr > wv_start[k]) & (ac12_time_arr < wv_end[k]))
                    duri_where = duri_where[0]
                    #print(duri_where)
                    
                    duri_data = np.array(ac12_data_arr[:,duri_where])
                    #print(ac12_data_arr.shape)
                    after_where = np.where((ac12_time_arr > wv_end[k]) & (ac12_time_arr < win_end_tmp[k]))
                    after_where = after_where[0]
                    after_data = np.array(ac12_data_arr[:,after_where])
                    
                    #print(pys.time_string(wv_end[k]))
                    before_max = np.max(before_data)
                    dura_max = np.max(duri_data) #get it? Like the diesel engine? I have achieved comedy.
                    after_max = np.max(after_data)
                    

 
                    bef_max_data.append(before_max)
                    duri_max_data.append(dura_max)
                    after_max_data.append(after_max)
                    
                    if (before_max<0.5*dura_max) and (after_max<0.5*dura_max): #check which waves have minimal wave
                        iso_waves.append(pys.time_string(wv_start[k]))         #power outside the marked interval
                        iso_duration.append(wv_dur[k])
                        #print('yessir')
            
                    """ #position data """
                    
                    wave_center = (win_end_tmp[k]+win_start_tmp[k])/2
                    time_dif = np.array(abs(pos_time_arr-wave_center))
                    pos_where = np.where(time_dif == min(time_dif))
                    pos_where = pos_where[0][0]

                    r_km = np.sqrt(pos_data_arr[pos_where,0]**2+pos_data_arr[pos_where,1]**2+pos_data_arr[pos_where,2]**2)
                    r_km_data.append(r_km)
                    r_m_data.append(r_km*10**3)
                    
                    """ mag data """
                    mag_time_dif = np.array(abs(mag_time_arr-wave_center))
                    mag_where = np.where(mag_time_dif == min(mag_time_dif))
                    mag_where = mag_where[0][0]
                    
                    Bmag = np.sqrt(mag_data_arr[mag_where,0]**2+mag_data_arr[mag_where,1]**2+mag_data_arr[mag_where,2]**2)
                    Bmag = Bmag*10e-10 #T
                    fce = q*Bmag/(2*np.pi*me) #Hz
                    fce_data.append(fce)
                    f_fce_data.append(ac12_freq_tmp/fce)  
                    
                    duri_max_where = np.argmax(duri_data)
                    invert = np.unravel_index(duri_max_where,duri_data.shape)

                    f_fce_max_data.append(ac12_freq_tmp[invert[0]]/fce)
                    
                    """ dens data """
                    
                    dens_time_dif = np.array(abs(dens_time_arr-wave_center))
                    dens_where = np.where(dens_time_dif == min(dens_time_dif))
                    dens_where = dens_where[0][0]
                    
                    dens = dens_data_arr[dens_where]*10e6 #1/m^3
                    fpi = (1/(2*np.pi))*np.sqrt(dens*q**2/(mi*eps))
                    fpi_data.append(fpi)
                    f_fpi_data.append(ac12_freq_tmp/fpi)
            
            iso_arr = np.transpose([iso_waves,iso_duration])
            iso_df = pd.DataFrame(iso_arr,columns=['Wave date','duration'])
            #print(iso_df)
            #print(np.array(f_fce_data).shape)
            min_ac12_len = min(ac12_len_arr)
            ac12_bins = min_ac12_len

         
            index_dif = []
            for l in range(len(ac12_time)): #this block seeks to normalize all the magnetic field data 
                                           #to the length of the shortest window
                                           
                fce_dif = np.array(abs(f_fce_data[l]-1))
                fce_where = np.where(fce_dif==min(fce_dif))
                fce_where = fce_where[0][0]
                
                index_dif.append(fce_where)
                
                
                ac12_bin_size = ac12_len_arr[l]/ac12_bins
                ac12_ind_arr = np.arange(ac12_len_arr[l])
                
                norm_ac12_time = np.zeros((ac12_bins,))
                norm_ac12_tmp = np.zeros((ac12_bins,len(ac12_freq_lst)))
                norm_ac12_freq = np.zeros((ac12_bins,len(ac12_freq_lst)))
                
                ac12_t = ac12_time[l]
                ac12_dat = ac12_spec[l]
                ac12_fq = ac12_freq[l]
                
                for m in range(ac12_bins): #hopefully these are the same
                    ac12_win_str = m*ac12_bin_size
                    ac12_win_end = (m+1)*ac12_bin_size
                    
                    ac12_wind_where = np.where((ac12_ind_arr>=ac12_win_str)&(ac12_ind_arr<ac12_win_end))
                    ac12_wind_where = ac12_wind_where[0]
                    
                    #print(ac12_freq.shape)
                    
                    ac12_time_val_tmp = np.median(ac12_t[ac12_wind_where])
                    ac12_val_tmp = np.median(ac12_dat[:,ac12_wind_where],axis=1)
                    ac12_freq_tmp = ac12_fq
                    
                    norm_ac12_time[m] = ac12_time_val_tmp
                    norm_ac12_tmp[m,:] = ac12_val_tmp
                    norm_ac12_freq[m,:] = ac12_freq_tmp
                    

                ac12_time[l] = np.array(norm_ac12_time)                
                ac12_spec[l] = np.transpose(norm_ac12_tmp)
                ac12_freq[l] = np.transpose(norm_ac12_freq)
            
            if plot == 'epoch' or plot == 'both':

                if norm_freq=='fce':
                    f_norm = f_fce_data
                    name = name+', normalized to fce'
                    save_tag = '_fce'
                else:
                    f_norm = f_fpi_data
                    name = name+', normalized to fpi'
                    save_tag = '_fpi'

                f_norm_range = np.mean(f_norm,axis=0)
                for l in range(len(ac12_time)):
                    spec = np.array(ac12_spec[l])
                    f_norm_lst = f_norm[l]
    
                    for m in range(len(spec[0,:])):
                        interspec = np.interp(f_norm_range,f_norm_lst,spec[:,m])
                        spec[:,m] = interspec
                    
                    ac12_spec[l] = spec
                    
                
                fis1 = plt.figure(figsize=(15,10))
                axs1 = fis1.add_subplot(111)
                
                savepath = '/Users/besh2109/Desktop/PSP_epoch/spec_epoch/'
                    
                if not by_Rs:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7'+save_tag+'_spec_epoch.png'
                        else:
                            savename = 'All_Events'+save_tag+'_spec_epoch.png'      
                    else:
                        savename = 'Enc_'+str(i)+save_tag+'_spec_epoch.png'
                else:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events'+save_tag+'_no_7_spec_epoch.png'
                        else:
                            savename = 'All_Events'+save_tag+'_spec_epoch.png'   
                    else:
                        savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs'+save_tag+'_spec_epoch.png'
                
                time_list = np.array(range(min_ac12_len))
                
                sum_arr = np.zeros(ac12_spec[0].shape)
    
                for o in ac12_spec:
                    sum_arr = sum_arr + o
                    #axs1.plot(o)
                
                freq_list = ac12_freq[0][:,0]
    
                spectra = axs1.pcolormesh(time_list,f_norm_range,sum_arr, cmap='jet')
                
                axs1.set(title='power spectrum, '+name)
                axs1.set_yscale("Log")
                axs1.set_ylim([0.2,10])
                box = axs1.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(spectra,cax=axColor, label='Counts')
                #fis2 = plt.figure(figsize=(15,10))
    
    
                plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                plt.clf()
                plt.cla()
                plt.close('all')
                plt.close(fis1)
                #plt.savefig(savepath+savename)
            if plot =='radial' or plot == 'both':
                
                fig = plt.figure(figsize=(15,10))
                ax1 = fig.add_subplot(111)
                Rs_data = np.array(r_km_data)/Rs
                max_data = np.array(duri_max_data)
                f_fce_max = np.array(f_fce_max_data)

                
                histo,xedge,yedge = np.histogram2d(Rs_data,f_fce_max, bins=[60,60]) #,bins=[n_mag_bins,40]
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
                #breakpoint()
                ax1.set_xlim([15,50])
                ax1.set_ylim([0,1])
                ax1.set_ylabel('f/fce')
                ax1.set_xlabel('Distance in Rs')
                hist2d = ax1.pcolormesh(xedge,yedge,histo,cmap='jet')
                box = ax1.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fig.colorbar(hist2d,cax=axColor, label='Counts')
                plt.show()
            
            if plot == 'none':
                pass
        i+=1
        
def ion_epoch(plot='epoch',no_enc_7=False, no_enc_1=False, win_len=1, by_Rs=False):
    
    i=0
    
    enc_num = len(per_flt)
    
    while i <= enc_num:
        
        Rs = 6.957e5 #solar radius in km
        w = 2*np.pi/(25.38*86400) # angular frequency of the sun in degrees/sec
        
        v = 400000 #m/s typical slow solar wind speed, may replace later with measurement values
        mu = 4*np.pi*1e-7 #mu naught
        eVtoJ = 1.60218*1e-19 #eV to J
        #kb = 1.380649*10e-23 #boltzmann constant, J/K
        
        if not by_Rs:
            if i==0:
                csv_filename = 'harmwave_master_arch.csv'
                csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
                if no_enc_7:
                    name = 'All Encounters sans 7'
                elif no_enc_1:
                    name = 'All Encounters sans 1'
                else:
                    name = 'All Encounters'
            else:
                csv_filename = 'enc_'+str(i)+'_harmwave_arch.csv'
                csv_path = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/Enc'+str(i)+'/'
                savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/histograms/'
                savename = 'Enc_'+str(i)+'_mag_epoch.png'
                name = 'Encounter '+str(i)
        
        else:
            csv_filename = 'harmwave_master_arch.csv'
            csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
            if i==0:
                if no_enc_7:
                    name = 'All Radial Distances sans Enc 7'
                elif no_enc_1:
                    name = 'All Radial Distances sans Enc 1'
                else:
                    name = 'All Radial Distances'
            else:
                name = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+' Rs'
        
        isfile = os.path.isfile(csv_path+csv_filename)

        if isfile:
            
            df = pd.read_csv(csv_path+csv_filename)
            bf = df.to_numpy()

            af = np.delete(bf,bf[:,1]<90,0)
            
            if by_Rs and i !=0:
                af = np.delete(af,af[:,2]>Rs_grps[i-1][0],0)
                af = np.delete(af,af[:,2]<Rs_grps[i-1][1],0)
            
            if no_enc_7 and i != enc_num: #this portion of code kills events in encounter 7
                dates = list(af[:,0])
                for j in range(len(dates)):
                    dates[j] = dates[j][0:4]
                npdates = np.array(dates)
                af = np.delete(af,npdates=='2021',0)
            
            if no_enc_1 and i != 1: #this portion of code kills events in encounter 1
                dates = list(af[:,0])
                for j in range(len(dates)):
                    dates[j] = dates[j][0:4]
                npdates = np.array(dates)
                af = np.delete(af,npdates=='2018',0)
                
            
            wave_start = np.array(pys.time_float(af[:,0]))
            wave_end = np.array(wave_start+af[:,1])
            
            window_start = np.array(wave_start - win_len*af[:,1])#*(1/3)) #complete epoch analysis for window larger than wave itself
            window_end = np.array(wave_end + win_len*af[:,1])#*(1/3))     #want to see if theres a difference between times during and before/after events
            
            dates = list(af[:,0])
            for j in range(len(dates)):
                dates[j] = dates[j][0:10]

            uniq_dates = np.unique(dates)
            date_flt = pys.time_float(uniq_dates)
            uniq_next = pys.time_string(np.array(date_flt)+86400.)
            
            duration = np.array(af[:,1])
            
            
            temp_time = []
            temperature_data = []
            temp_len_arr = []
            
            dens_time = []
            density_data = []
            dens_len_arr = []
            
            tens_time = []
            tens_sc_data = [] #tensor in sc coords
            
            pa_time = []
            pa_spec = []
            pa_angle = []
            pa_len_arr = []
            
            mag_time = []
            mag_scx_data = []
            mag_scy_data = []
            mag_scz_data = []
            mag_len_arr = []
            
            r_data = []
            
            beftens_sc_data = []
            durtens_sc_data = []
            afttens_sc_data = []
            bef_time = []
            dur_time = []
            aft_time = []
            bef_len_arr = []
            dur_len_arr = []
            aft_len_arr = []


            
            for j in range(len(uniq_dates)): #gather data for each day range(2):#

                """ PSP Positional data """
                pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]], datatype='ephem_eclipj2000', level='l1') #going to be used to plot parker position
                pos_data = pyt.get_data('position')

                pos_time_arr = pos_data[0]
                pos_data_arr = pos_data[1]
                
                """ PSP ion data """
                
                date_form_1 = pys.time_string(date_flt[j],fmt='/%Y/%m/')
                date_form_2 = pys.time_string(date_flt[j],fmt='%Y%m%d')
                rot_path = '/Users/besh2109/spedas_data/psp/data/sci/sweap/spi/L3/spi_sf00'+date_form_1
                rot_file = 'psp_swp_spi_sf00_L3_mom_INST_'+date_form_2+'_v02.cdf' 
                
                if os.path.isfile(rot_path+rot_file):
                    pyt.cdf_to_tplot(rot_path+rot_file)
                else:
                    pys.psp.spi(trange=[uniq_dates[j],uniq_next[j]], datatype='spi_sf00', level='L3')
                
                temp_data = pyt.get_data('TEMP')
                
                temp_time_arr = temp_data[0]
                temp_data_arr = temp_data[1]
                
                tens_data = pyt.get_data('T_TENSOR')
                
                tens_time_arr = tens_data[0]
                tens_data_arr = tens_data[1]
                
                rot_cdf = cdflib.CDF(rot_path+rot_file)
                
                rot_mat_ins_sc = rot_cdf.varget('ROTMAT_SC_INST')
                rot_mat_ins_sc_inv = inv(rot_mat_ins_sc)
                
                dens_data = pyt.get_data('DENS')
                
                dens_time_arr = dens_data[0]
                dens_data_arr = dens_data[1]
                
                """ PSP electron data """
                
                pys.psp.spe(trange=[uniq_dates[j],uniq_next[j]], datatype='spe_sf0_pad', level='L3')
                
                pa_data = pyt.get_data('EFLUX_VS_PA_E')
                
                pa_time_arr = pa_data[0]
                pa_data_arr = pa_data[1]
                pa_angle_arr = pa_data[2]
                pa_energy_arr = pa_data[3]
                
                ener = 10 #which energy bin to plot for pitch angle
                
                pa_data_arr = np.transpose(pa_data_arr[:,:,ener])
                pa_angle_arr = np.transpose(pa_angle_arr)
                pa_angle_lst = pa_angle_arr[:,0]
                
                energy = pa_energy_arr[0,ener]

                """ magnetic field data """
                pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]], datatype='mag_SC_4_Sa_per_Cyc', level='l2', last_version=True) #magnetic field, this time in SC coords
                
                mag_data = pyt.get_data('psp_fld_l2_mag_SC_4_Sa_per_Cyc')
                
                mag_time_arr = mag_data[0]
                mag_data_arr = mag_data[1]

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

                    
                    """ #temp data """
                    temp_where = np.where((temp_time_arr > win_start_tmp[k]) & (temp_time_arr < win_end_tmp[k]))
                    temp_where = temp_where[0]
                    
                    temp_ti = np.array(temp_time_arr[temp_where])
                    temp = np.array(temp_data_arr[temp_where])
                    
                    temp_time.append(temp_ti)
                    temperature_data.append(temp)
                    temp_len_arr.append(len(temp_ti))
                                        
                    tens_ti = np.array(tens_time_arr[temp_where])
                    tens = np.array(tens_data_arr[temp_where,:])
                    tensor = []
                    for p in range(len(temp_where)):
                        
                        Txx = tens[p,0]
                        Tyy = tens[p,1]
                        Tzz = tens[p,2]
                        Txy = tens[p,3]
                        Txz = tens[p,4]
                        Tyz = tens[p,5]
                        
                        tens_tmp = np.array([[Txx,Txy,Txz],[Txy,Tyy,Tyz],[Txz,Tyz,Tzz]])
                        
                        tens_tmp_1 = np.dot(rot_mat_ins_sc,tens_tmp)
                        tensor.append(np.dot(tens_tmp_1,rot_mat_ins_sc_inv))

                    tensor = np.array(tensor)
                    tens_time.append(tens_ti)
                    tens_sc_data.append(tensor)
                    
                    before_where = np.where((temp_time_arr > win_start_tmp[k]) & (temp_time_arr < wv_start[k]))
                    before_where = before_where[0]
                    
                    bef_ti = np.array(tens_time_arr[before_where])
                    before_data = np.array(tens_data_arr[before_where,:])
                    
                    beftens = []
                    for p in range(len(before_where)):
                        
                        Txx = before_data[p,0]
                        Tyy = before_data[p,1]
                        Tzz = before_data[p,2]
                        Txy = before_data[p,3]
                        Txz = before_data[p,4]
                        Tyz = before_data[p,5]
                        
                        tens_tmp = np.array([[Txx,Txy,Txz],[Txy,Tyy,Tyz],[Txz,Tyz,Tzz]])
                        
                        tens_tmp_1 = np.dot(rot_mat_ins_sc,tens_tmp)
                        beftens.append(np.dot(tens_tmp_1,rot_mat_ins_sc_inv))
                        
                    beftens = np.array(beftens)
                    beftens_sc_data.append(beftens)

                    
                    duri_where = np.where((temp_time_arr > wv_start[k]) & (temp_time_arr < wv_end[k]))
                    duri_where = duri_where[0]
                    dur_ti = np.array(tens_time_arr[duri_where])
                    duri_data = np.array(tens_data_arr[duri_where,:])
                    
                    durtens = []
                    for p in range(len(duri_where)):
                        
                        Txx = duri_data[p,0]
                        Tyy = duri_data[p,1]
                        Tzz = duri_data[p,2]
                        Txy = duri_data[p,3]
                        Txz = duri_data[p,4]
                        Tyz = duri_data[p,5]
                        
                        tens_tmp = np.array([[Txx,Txy,Txz],[Txy,Tyy,Tyz],[Txz,Tyz,Tzz]])
                        
                        tens_tmp_1 = np.dot(rot_mat_ins_sc,tens_tmp)
                        durtens.append(np.dot(tens_tmp_1,rot_mat_ins_sc_inv))

                    durtens = np.array(durtens)
                    durtens_sc_data.append(durtens)


                    after_where = np.where((temp_time_arr > wv_end[k]) & (temp_time_arr < win_end_tmp[k]))
                    after_where = after_where[0]
                    aft_ti = np.array(tens_time_arr[after_where])
                    after_data = np.array(tens_data_arr[after_where,:])
                    
                    afttens = []
                    for p in range(len(after_where)):
                        
                        Txx = after_data[p,0]
                        Tyy = after_data[p,1]
                        Tzz = after_data[p,2]
                        Txy = after_data[p,3]
                        Txz = after_data[p,4]
                        Tyz = after_data[p,5]
                        
                        tens_tmp = np.array([[Txx,Txy,Txz],[Txy,Tyy,Tyz],[Txz,Tyz,Tzz]])
                        
                        tens_tmp_1 = np.dot(rot_mat_ins_sc,tens_tmp)
                        afttens.append(np.dot(tens_tmp_1,rot_mat_ins_sc_inv))
        
                    afttens = np.array(afttens)
                    afttens_sc_data.append(afttens)
                    
                    """ density data """
                    dens_where = np.where((dens_time_arr > win_start_tmp[k]) & (dens_time_arr < win_end_tmp[k]))
                    dens_where = dens_where[0]
                    
                    dens_ti = np.array(dens_time_arr[dens_where])
                    dens = np.array(dens_data_arr[dens_where])*1e6 #1/m^3
                    
                    dens_time.append(dens_ti)
                    density_data.append(dens)
                    dens_len_arr.append(len(dens_ti))
                    
                    """ pitch angle data """
                    pa_where = np.where((pa_time_arr > win_start_tmp[k]) & (pa_time_arr < win_end_tmp[k]))
                    pa_where = pa_where[0]
                    
                    pa_ti = np.array(pa_time_arr[pa_where])
                    pa_data_tmp = np.array(pa_data_arr[:,pa_where])
                    pa_angle_tmp = np.array(pa_angle_arr[:,pa_where[0]])
                    
                    pa_time.append(pa_ti)
                    pa_spec.append(pa_data_tmp)
                    pa_angle.append(pa_angle_tmp)
                    pa_len_arr.append(len(pa_ti))
            
                    """ #position data """
                    
                    wave_center = (win_end_tmp[k]+win_start_tmp[k])/2
                    time_dif = np.array(abs(pos_time_arr-wave_center))
                    pos_where = np.where(time_dif == min(time_dif))
                    pos_where = pos_where[0][0]

                    r_km = np.sqrt(pos_data_arr[pos_where,0]**2+pos_data_arr[pos_where,1]**2+pos_data_arr[pos_where,2]**2)
                    r_data.append(r_km)
            
                    """ #magnetic field data """
                    mag_where = np.where((mag_time_arr > win_start_tmp[k]) & (mag_time_arr < win_end_tmp[k]))
                    mag_where = mag_where[0]
                    
                    mag_ti = np.array(mag_time_arr[mag_where])
                    mag_scx = np.array(mag_data_arr[mag_where,0])
                    mag_scy = np.array(mag_data_arr[mag_where,1])
                    mag_scz = np.array(mag_data_arr[mag_where,2])
    
                    mag_time.append(mag_ti)
                    mag_scx_data.append(mag_scx)
                    mag_scy_data.append(mag_scy)
                    mag_scz_data.append(mag_scz)
                    mag_len_arr.append(len(mag_ti))
                    
                    bef_time.append(bef_ti)
                    bef_len_arr.append(len(bef_ti))
                    
                    dur_time.append(dur_ti)
                    dur_len_arr.append(len(dur_ti))
                    
                    aft_time.append(aft_ti)
                    aft_len_arr.append(len(aft_ti))
            
            min_temp_len = min(temp_len_arr)
            n_temp_bins = int(min_temp_len)
  
            min_pa_len = min(pa_len_arr)
            pa_bins = min_pa_len
            
            min_mag_len = min(mag_len_arr) #minimum length of mag data
            n_mag_bins = int(min_mag_len)
            
            mag_b_data = []
            rot_mat_sc_fa_lst = []
            rot_mat_inv_lst = []
            
            t_tensor_FA_coords = []
            
            T_perp_I = []
            T_par_I = []
            T_anis_I = []
            Beta_par_I = []
            
            T_anis_bef = []
            T_anis_dur = []
            T_anis_aft = []
            
            Beta_par_bef = []
            Beta_par_dur = []
            Beta_par_aft = []
            
            Beta_par_bef_med = []
            Beta_par_dur_med = []
            Beta_par_aft_med = []
            
            T_anis_bef_med = []
            T_anis_dur_med = []
            T_anis_aft_med = []
            
            for l in range(len(temp_time)): #this block seeks to normalize all the magnetic field data 
                                           #to the length of the shortest window

                temp_bin_size = temp_len_arr[l]/n_temp_bins
                temp_ind_arr = np.arange(temp_len_arr[l])
                norm_temp_time = np.zeros((n_temp_bins,))
                norm_temp_val = np.zeros((n_temp_bins,))
                norm_tens_time = np.zeros((n_temp_bins,))
                norm_tens_val = np.zeros((n_temp_bins,3,3))
                norm_dens_val = np.zeros((n_temp_bins,))

                rot_mat_sc_fa = []
                rot_mat_inv = []
                
                bin_size = mag_len_arr[l]/n_temp_bins
                ind_arr = np.arange(mag_len_arr[l])
                norm_mag_time = np.zeros((n_temp_bins,))
                norm_mag_scx_val = np.zeros((n_temp_bins,))
                norm_mag_scy_val = np.zeros((n_temp_bins,))
                norm_mag_scz_val = np.zeros((n_temp_bins,))
                
                tens_fa = []
                
                for m in range(n_temp_bins):
                    
                    win_str = m*temp_bin_size
                    win_end = (m+1)*temp_bin_size
                    wind_where = np.where((temp_ind_arr>=win_str)&(temp_ind_arr<win_end))
                    wind_where = wind_where[0]
                    temp_time_val_tmp = np.median(temp_time[l][wind_where])
                    temp_val_tmp = np.median(temperature_data[l][wind_where])
                    dens_val_tmp = np.median(density_data[l][wind_where])
                    tens_val_tmp = np.median(tens_sc_data[l][wind_where,:,:],axis=0)
                    
                    norm_temp_time[m] = temp_time_val_tmp
                    norm_temp_val[m] = temp_val_tmp
                    
                    norm_dens_val[m] = dens_val_tmp
                    
                    norm_tens_time[m] = temp_time_val_tmp
                    norm_tens_val[m] = tens_val_tmp
                    
                    
                    win_str_mag = m*bin_size
                    win_end_mag = (m+1)*bin_size
                    wind_where = np.where((ind_arr>=win_str_mag)&(ind_arr<win_end_mag))
                    wind_where = wind_where[0]
                    mag_time_val_tmp = np.median(mag_time[l][wind_where])
                    mag_scx_val_tmp = np.median(mag_scx_data[l][wind_where])
                    mag_scy_val_tmp = np.median(mag_scy_data[l][wind_where])
                    mag_scz_val_tmp = np.median(mag_scz_data[l][wind_where])
                    norm_mag_time[m] = mag_time_val_tmp
                    norm_mag_scx_val[m] = mag_scx_val_tmp
                    norm_mag_scy_val[m] = mag_scy_val_tmp
                    norm_mag_scz_val[m] = mag_scz_val_tmp
                    
                    Bx = mag_scx_val_tmp
                    By = mag_scy_val_tmp
                    Bz = mag_scz_val_tmp
                    B = np.sqrt(Bx**2+By**2+Bz**2)
                    Bx_By = np.sqrt(Bx**2+By**2)
                    
                    rot_mat = np.array([[Bx/B,    By/B,                         Bz/B],\
                                       [0,        Bx_By*Bz/(By**2+Bz**2),      -Bx_By*By/(By**2+Bz**2)],\
                                       [-Bx_By/B, Bx*By*Bx_By/(B*(By**2+Bz**2)),Bx*Bz*Bx_By/(B*(By**2+Bz**2))]])#rotation matrix from SC to FA
                        
                    rot_mat_sc_fa.append(rot_mat)
                    rot_inv = inv(rot_mat)
                    rot_mat_inv.append(rot_inv)
                    tmp_tens = np.dot(rot_mat,tens_val_tmp) #first calculation of T' = R T R^-1
                    tens_fa.append(np.dot(tmp_tens,rot_inv)) #temperature tensor rotated to FA coords.

                tens_fa = np.array(tens_fa)
    
                temp_time[l] = np.array(norm_temp_time)                
                temperature_data[l] = np.array(norm_temp_val)
                density_data[l] = np.array(norm_dens_val)
                tens_time[l] = np.array(norm_tens_time)
                tens_sc_data[l] = np.array(norm_tens_val)
                
                rot_mat_sc_fa = np.array(rot_mat_sc_fa)
                rot_mat_inv = np.array(rot_mat_inv)
                
                
                rot_mat_sc_fa_lst.append(rot_mat_sc_fa)
                rot_mat_inv_lst.append(rot_mat_inv)
                mag_time[l] = np.array(norm_mag_time)
                
                b_mag = np.sqrt(norm_mag_scx_val**2+norm_mag_scy_val**2+norm_mag_scz_val**2)*1e-9 #convert to Tesla from nT
                mag_b_data.append(b_mag)
                mag_scx_data[l] =  np.array(norm_mag_scx_val)
                mag_scy_data[l] =  np.array(norm_mag_scy_val)
                mag_scz_data[l] =  np.array(norm_mag_scz_val)
                
                t_tensor_FA_coords.append(np.array(tens_fa))

                T_perp_I_tmp = np.array(tens_fa[:,1,1]+tens_fa[:,2,2])/2
                T_par_I_tmp = np.array(tens_fa[:,0,0])
                T_anis = T_perp_I_tmp/T_par_I_tmp
                
                T_perp_I.append(T_perp_I_tmp/temperature_data[l])
                T_par_I.append(T_par_I_tmp/temperature_data[l])
                T_anis_I.append(T_anis)
                
                T_par_J = T_par_I_tmp*eVtoJ #converts eV temperature to Joules
                
                beta = 2*mu*density_data[l]*T_par_J/b_mag**2
                Beta_par_I.append(beta)
                
                bef_beta = []
                dur_beta = []
                aft_beta = []
                
                bef_anis = []
                dur_anis = []
                aft_anis = []
                for m in range(n_temp_bins):
                    if m < n_temp_bins/3:
                        bef_beta.append(beta[m])
                        bef_anis.append(T_anis[m])
                    elif m >= n_temp_bins/3 and m < 2*n_temp_bins/3:
                        dur_beta.append(beta[m])
                        dur_anis.append(T_anis[m])
                    elif m >= 2*n_temp_bins/3:
                        aft_beta.append(beta[m])
                        aft_anis.append(T_anis[m])
                
                Beta_par_bef.append(bef_beta)
                Beta_par_dur.append(dur_beta)
                Beta_par_aft.append(aft_beta)
                
                T_anis_bef.append(bef_anis)
                T_anis_dur.append(dur_anis)
                T_anis_aft.append(aft_anis)
                
                Beta_par_bef_med.append(bef_beta[round(len(bef_beta)/2)])
                Beta_par_dur_med.append(dur_beta[round(len(dur_beta)/2)])
                Beta_par_aft_med.append(aft_beta[round(len(aft_beta)/2)]) #aft_beta
                
                T_anis_bef_med.append(bef_anis[round(len(bef_anis)/2)]) #bef_anis
                T_anis_dur_med.append(dur_anis[round(len(dur_anis)/2)]) #dur_anis
                T_anis_aft_med.append(aft_anis[round(len(aft_anis)/2)]) #aft_anis
                
                pa_bin_size = pa_len_arr[l]/pa_bins
                pa_ind_arr = np.arange(pa_len_arr[l])
                
                norm_pa_time = np.zeros((pa_bins,))
                norm_pa_tmp = np.zeros((pa_bins,len(pa_angle_lst)))
                norm_pa_angle = np.zeros((pa_bins,len(pa_angle_lst)))
                
                #print(norm_pa_tmp.shape)
                #print(norm_pa_angle.shape)
                
                pa_t = pa_time[l]
                pa_dat = pa_spec[l]
                pa_ang = pa_angle[l]
                
                for m in range(pa_bins):
                    pa_win_str = m*pa_bin_size
                    pa_win_end = (m+1)*pa_bin_size
                    
                    pa_wind_where = np.where((pa_ind_arr>=pa_win_str)&(pa_ind_arr<pa_win_end))
                    pa_wind_where = pa_wind_where[0]
                    
                    pa_time_val_tmp = np.median(pa_t[pa_wind_where])
                    pa_val_tmp = np.nanmedian(pa_dat[:,pa_wind_where],axis=1)
                    pa_ang_tmp = pa_ang
                    
                    #print(pa_val_tmp.shape)
                    #print(pa_ang_tmp.shape)
                    
                    norm_pa_time[m] = pa_time_val_tmp
                    norm_pa_tmp[m,:] = pa_val_tmp
                    norm_pa_angle[m,:] = pa_ang_tmp
                    
                pa_time[l] = np.array(norm_pa_time)
                pa_spec[l] = np.array(norm_pa_tmp)
                pa_angle[l] = np.array(norm_pa_angle)
            
            
            
            if plot == 'epoch' or plot == 'all':

                savepath = '/Users/besh2109/Desktop/PSP_epoch/temp_epoch/'

                if not by_Rs:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_temp_epoch.png'
                        elif no_enc_1:
                            savename = 'All_Events_no_1_temp_epoch.png'
                        else:
                            savename = 'All_Events_temp_epoch.png'      
                    else:
                        savename = 'Enc_'+str(i)+'_temp_epoch.png'
                else:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_temp_epoch.png'
                        elif no_enc_1:
                            savename = 'All_Events_no_1_temp_epoch.png'
                        else:
                            savename = 'All_Events_temp_epoch.png'      
                    else:
                        savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs_temp_epoch.png'
            
                fis1 = plt.figure(figsize=(15,12))
                axs1 = fis1.add_subplot(411)
                
                ind_hist = []
                data_hist = []
                for o in T_perp_I:
                    ind = list(range(len(o)))
                    data = list(o)
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs1.plot(o)
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                            
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_temp_bins,125])
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
    
                r_color = axs1.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                
                #height = [-0.6,-0.6]
                #endpoints = [n_vel_bins/3,2*n_vel_bins/3]
                #axs1.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_r_data)+1)
                axs1.set(title='Ion Tperp/Tinst')
                #axs1.set_ylim(-25,25)
                axs1.set_ylabel('Tperp/Tinst')
                #axs1.set_xlabel('Normalized time')
                #plt.title()
                #plt.show()
                box = axs1.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(r_color,cax=axColor, label='Counts')
                
                axs2 = fis1.add_subplot(412)
                
                ind_hist = []
                data_hist = []
                for o in T_par_I:
                    ind = list(range(len(o)))
                    data = list(o)
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs1.plot(o)
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                            
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_temp_bins,125])
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
    
                r_color = axs2.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                
                #height = [-0.6,-0.6]
                #endpoints = [n_vel_bins/3,2*n_vel_bins/3]
                #axs1.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_r_data)+1)
                axs2.set(title='Ion Tpar/Tinst')
                #axs2.set_ylim(-25,25)
                axs2.set_ylabel('Tpar/Tinst')
                #axs2.set_xlabel('Normalized time')
                #plt.title()
                #plt.show()
                box = axs2.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(r_color,cax=axColor, label='Counts')
                
                axs3 = fis1.add_subplot(413)
                
                ind_hist = []
                data_hist = []
                for o in T_anis_I:
                    ind = list(range(len(o)))
                    data = list(o)
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs1.plot(o)
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                # ind_hist_arr = np.delete(ind_hist_arr,data_hist_arr<-1)
                # data_hist_arr = np.delete(data_hist_arr,data_hist_arr<-1)
                
                # ind_hist_arr = np.delete(ind_hist_arr,data_hist_arr>4)
                # data_hist_arr = np.delete(data_hist_arr,data_hist_arr>4)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_temp_bins,200])
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
    
                r_color = axs3.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                
                #height = [-0.6,-0.6]
                #endpoints = [n_vel_bins/3,2*n_vel_bins/3]
                #axs1.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_r_data)+1)
                axs3.set(title='Ion temperature anisotropy')
                axs3.set_ylim(0,3)
                axs3.set_ylabel('Tperp/Tpar')
                #axs3.set_xlabel('Normalized time')
                #plt.title()
                #plt.show()
                box = axs3.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(r_color,cax=axColor, label='Counts')
                
                
                axs4 = fis1.add_subplot(414)
                
                ind_hist = []
                data_hist = []
                for o in Beta_par_I:
                    ind = list(range(len(o)))
                    data = list(o)
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                    #axs1.plot(o)
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_temp_bins,125])
                histo = np.transpose(histo)
                histo[histo==0]=np.nan
    
                r_color = axs4.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                
                #height = [-0.6,-0.6]
                #endpoints = [n_vel_bins/3,2*n_vel_bins/3]
                #axs4.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_r_data)+1)
                axs4.set(title='Plasma Beta Parallel')
                #axs4.set_ylim(-0.2,0.75)
                axs4.set_ylabel('P/Pb')
                #axs4.set_xlabel('Normalized time')
                #plt.title()
                #plt.show()
                box = axs4.get_position()
                axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                fis1.colorbar(r_color,cax=axColor, label='Counts')
                
                #fis2 = plt.figure(figsize=(15,10))
                #breakpoint()
                plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                plt.clf()
                plt.cla()
                plt.close('all')
                plt.close(fis1)
                #plt.show()
            
            if plot == 'dist' or plot == 'all':
                
                #fis1 = plt.figure(figsize=(15,10))
                #axs1 = fis1.add_subplot(131)

                savepath = '/Users/besh2109/Desktop/PSP_epoch/temp_dist/'
                
                if not by_Rs:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_temp_dist.png'
                        elif no_enc_1:
                            savename = 'All_Events_no_1_temp_dist.png'
                        else:
                            savename = 'All_Events_temp_dist.png'      
                    else:
                        savename = 'Enc_'+str(i)+'_temp_dist.png'
                else:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_temp_dist.png'
                        elif no_enc_1:
                            savename = 'All_Events_no_1_temp_dist.png'
                        else:
                            savename = 'All_Events_temp_dist.png'      
                    else:
                        if no_enc_1:
                            savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs_no_1_temp_dist.png'
                        else:
                            savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs_temp_dist.png'

                fig1 = plt.figure(figsize=(15,15))
                fig1.suptitle('Ion Temperature Distributions '+name, fontsize=16,y=0.92)
                ax1 = fig1.add_subplot(221)
                ax2 = fig1.add_subplot(222)
                ax3 = fig1.add_subplot(223)
                ax4 = fig1.add_subplot(224)
                
                ind_hist = []
                data_hist = []
                for o in T_perp_I:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_temp_bins,125])
                histo = np.transpose(histo)

                k=0
                dist1per = np.zeros(histo[:,0].shape)
                while k < round(n_temp_bins/3): #distribution 1
                    dist1per = dist1per + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_temp_bins/3)
                dist2per = np.zeros(histo[:,0].shape)
                while k >= round(n_temp_bins/3) and k < round(2*n_temp_bins/3): #distribution 2
                    dist2per = dist2per + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_temp_bins/3)
                dist3per = np.zeros(histo[:,0].shape)
                while k >= round(2*n_temp_bins/3) and k<n_temp_bins: #distribution 3
                    dist3per = dist3per + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax1.set_aspect(1)
                #ax1.set_adjustable('box')
                norm_val = np.max(dist2per)
                ax1.plot(yedge[1:len(yedge)],dist1per/norm_val,color='red',label='before wave')
                ax1.plot(yedge[1:len(yedge)],dist2per/norm_val,color='green',label='during wave')
                ax1.plot(yedge[1:len(yedge)],dist3per/norm_val,color='blue',label='after wave')
                ax1.legend()
                #ax1.set_ylim([0,120])
                ax1.set_title('Ion Temp Perp / Ion Temp INST')
                ax1.set_ylabel('Normalized Counts Distribution')
                ax1.set_xlabel('Tperp/Tinst')
                ax1.set_adjustable('box')
                #ax1.set_xlim(-25,25)
                
                #ax2 = fig1.add_subplot(2,2,(1,2))
                
                ind_hist = []
                data_hist = []
                for o in T_par_I:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_temp_bins,125])
                histo = np.transpose(histo)
                
                

                k=0
                dist1par = np.zeros(histo[:,0].shape)
                while k < round(n_temp_bins/3): #distribution 1
                    dist1par = dist1par + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_temp_bins/3)
                dist2par = np.zeros(histo[:,0].shape)
                while k >= round(n_temp_bins/3) and k < round(2*n_temp_bins/3): #distribution 2
                    dist2par = dist2par + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_temp_bins/3)
                dist3par = np.zeros(histo[:,0].shape)
                while k >= round(2*n_temp_bins/3) and k<n_temp_bins: #distribution 3
                    dist3par = dist3par + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax2.set_aspect(1)
                #ax2.set_adjustable('box')
                norm_val = np.max(dist2par)
                ax2.plot(yedge[1:len(yedge)],dist1par/norm_val,color='red',label='before wave')
                ax2.plot(yedge[1:len(yedge)],dist2par/norm_val,color='green',label='during wave')
                ax2.plot(yedge[1:len(yedge)],dist3par/norm_val,color='blue',label='after wave')
                ax2.legend()
                #ax2.set_ylim([0,26])
                ax2.set_title('Ion Temp Par / Ion Temp INST')
                ax2.set_ylabel('Normalized Counts Distribution')
                ax2.set_xlabel('Tpar/Tinst')
                ax2.set_adjustable('box')
                
                #ax3 = fig1.add_subplot(2,2,(2,1))
                
                ind_hist = []
                data_hist = []
                for o in T_anis_I:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_temp_bins,125])
                histo = np.transpose(histo)

                k=0
                dist1an = np.zeros(histo[:,0].shape)
                while k < round(n_temp_bins/3): #distribution 1
                    dist1an = dist1an + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_temp_bins/3)
                dist2an = np.zeros(histo[:,0].shape)
                while k >= round(n_temp_bins/3) and k < round(2*n_temp_bins/3): #distribution 2
                    dist2an = dist2an + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_temp_bins/3)
                dist3an = np.zeros(histo[:,0].shape)
                while k >= round(2*n_temp_bins/3) and k<n_temp_bins: #distribution 3
                    dist3an = dist3an + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax3.set_aspect(1)
                #ax3.set_adjustable('box')
                norm_val = np.max(dist2an)
                ax3.plot(yedge[1:len(yedge)],dist1an/norm_val,color='red',label='before wave')
                ax3.plot(yedge[1:len(yedge)],dist2an/norm_val,color='green',label='during wave')
                ax3.plot(yedge[1:len(yedge)],dist3an/norm_val,color='blue',label='after wave')
                ax3.legend()
                #ax3.set_ylim([0,34])
                ax3.set_xlim([0,3])
                ax3.set_title('Ion Temperature Anisotropy')
                ax3.set_ylabel('Normalized Counts Distribution')
                ax3.set_xlabel('Tperp/Tpar')
                ax3.set_adjustable('box')
                
                #ax4 = fig1.add_subplot(2,2,(2,2))
                
                ind_hist = []
                data_hist = []
                for o in Beta_par_I:
                    ind = list(range(len(o)))
                    data = list(o) #- np.median(o))
                    ind_hist = ind_hist + ind
                    data_hist = data_hist + data
                ind_hist_arr = np.array(ind_hist)
                data_hist_arr = np.array(data_hist)
                
                histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_temp_bins,125])
                histo = np.transpose(histo)

                k=0
                dist1B = np.zeros(histo[:,0].shape)
                while k < round(n_temp_bins/3): #distribution 1
                    dist1B = dist1B + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(n_temp_bins/3)
                dist2B = np.zeros(histo[:,0].shape)
                while k >= round(n_temp_bins/3) and k < round(2*n_temp_bins/3): #distribution 2
                    dist2B = dist2B + histo[:,k]/np.sum(histo[:,k])
                    k+=1

                k=round(2*n_temp_bins/3)
                dist3B = np.zeros(histo[:,0].shape)
                while k >= round(2*n_temp_bins/3) and k<n_temp_bins: #distribution 3
                    dist3B = dist3B + histo[:,k]/np.sum(histo[:,k])
                    k+=1
                
                #ax3.set_aspect(1)
                #ax3.set_adjustable('box')
                norm_val = np.max(dist2B)
                ax4.plot(yedge[1:len(yedge)],dist1B/norm_val,color='red',label='before wave')
                ax4.plot(yedge[1:len(yedge)],dist2B/norm_val,color='green',label='during wave')
                ax4.plot(yedge[1:len(yedge)],dist3B/norm_val,color='blue',label='after wave')
                ax4.legend()
                #ax4.set_xlim([0,2])
                #ax4.set_ylim([0,50])
                ax4.set_title('Plasma Beta Parallel')
                ax4.set_ylabel('Normalized Counts Distribution')
                ax4.set_xlabel('P/Pb')
                ax4.set_adjustable('box')
                
                
                plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                plt.clf()
                plt.cla()
                plt.close('all')
                plt.close(fig1)
                
            if plot == 'beta' or plot == 'all':
                
                #fis1 = plt.figure(figsize=(15,10))
                #axs1 = fis1.add_subplot(131)

                savepath = '/Users/besh2109/Desktop/PSP_epoch/temp_beta/'
                
                if not by_Rs:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_temp_beta.png'
                        elif no_enc_1:
                            savename = 'All_Events_no_1_temp_beta.png'
                        else:
                            savename = 'All_Events_temp_beta.png'      
                    else:
                        savename = 'Enc_'+str(i)+'_temp_beta.png'
                else:
                    if i==0:
                        if no_enc_7:
                            savename = 'All_Events_no_7_temp_beta.png'
                        elif no_enc_1:
                            savename = 'All_Events_no_1_temp_beta.png'
                        else:
                            savename = 'All_Events_temp_beta.png'      
                    else:
                        if no_enc_1:
                            savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs_no_1_temp_beta.png'
                        else:
                            savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs_temp_beta.png'

                fig1 = plt.figure(figsize=(15,15))
                fig1.suptitle('Ion Anisotropy vs Beta Para Distributions '+name, fontsize=16,y=0.92)
                ax1 = fig1.add_subplot(221)
                ax2 = fig1.add_subplot(222)
                ax3 = fig1.add_subplot(223)
                ax4 = fig1.add_subplot(224)
                
                histo,xedge,yedge = np.histogram2d(Beta_par_bef_med,T_anis_bef_med, range=[[0, 1], [0, 5]],bins=[25,25]) #bins=[25,25],
                ax1.scatter(Beta_par_bef_med,T_anis_bef_med,s=1,color='red',label='before wave')
                ax1.pcolormesh(xedge,yedge,np.transpose(histo)) #,extent=[xedge.min(),xedge.max(),yedge.min(),yedge.max()]
                
                ax1.set_title('Ion Temperature Anisotropy vs Beta Parallel Before Wave')
                ax1.set_ylabel('Tperp/Tpar')
                ax1.set_xlabel('Beta Parallel')
                ax1.set_ylim([0,5])
                ax1.set_xlim([0,1])
                ax1.set_adjustable('box')
                ax1.legend()
                
                histo,xedge,yedge = np.histogram2d(Beta_par_dur_med,T_anis_dur_med, range=[[0, 1], [0, 5]],bins=[25,25]) # bins=[25,25],
                ax2.scatter(Beta_par_dur_med,T_anis_dur_med,s=1,color='green',label='during wave')
                ax2.pcolormesh(xedge,yedge,np.transpose(histo)) #,extent=[xedge.min(),xedge.max(),yedge.min(),yedge.max()]
                
                #ax2.set_ylim([0,26])
                ax2.set_title('Ion Temperature Anisotropy vs Beta Parallel During Wave')
                ax2.set_ylabel('Tperp/Tpar')
                ax2.set_xlabel('Beta Parallel')
                ax2.set_ylim([0,5])
                ax2.set_xlim([0,1])
                ax2.set_adjustable('box')
                ax2.legend()
                
                histo,xedge,yedge = np.histogram2d(Beta_par_aft_med,T_anis_aft_med, range=[[0, 1], [0, 5]],bins=[25,25]) #bins=[25,25],
                ax3.scatter(Beta_par_aft_med,T_anis_aft_med,s=1,color='blue',label='after wave')
                ax3.pcolormesh(xedge,yedge,np.transpose(histo)) #,extent=[xedge.min(),xedge.max(),yedge.min(),yedge.max()]
                
                ax3.set_title('Ion Temperature Anisotropy vs Beta Parallel After Wave')
                ax3.set_ylabel('Tperp/Tpar')
                ax3.set_xlabel('Beta Parallel')
                ax3.set_ylim([0,5])
                ax3.set_xlim([0,1])
                ax3.set_adjustable('box')
                ax3.legend()
                
                beta_list = Beta_par_bef_med+Beta_par_dur_med+Beta_par_aft_med
                anis_list = T_anis_bef_med+T_anis_dur_med+T_anis_aft_med
                histo,xedge,yedge = np.histogram2d(beta_list,anis_list, range=[[0, 1], [0, 5]],bins=[25,25]) #bins=[25,25],
                ax4.scatter(Beta_par_bef_med,T_anis_bef_med,s=1,color='red',label='before wave')
                ax4.scatter(Beta_par_dur_med,T_anis_dur_med,s=1,color='green',label='during wave')
                ax4.scatter(Beta_par_aft_med,T_anis_aft_med,s=1,color='blue',label='after wave')
                ax4.pcolormesh(xedge,yedge,np.transpose(histo))#,extent=[xedge.min(),xedge.max(),yedge.min(),yedge.max()]
                
                ax4.set_title('Ion Temperature Anisotropy vs Beta Parallel After Wave')
                ax4.set_ylabel('Tperp/Tpar')
                ax4.set_xlabel('Beta Parallel')
                ax4.set_ylim([0,5])
                ax4.set_xlim([0,1])
                ax4.set_adjustable('box')
                ax4.legend()
                #ax4.set_xlim([0,2])
                #ax4.set_ylim([0,50])

                
                
                plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                plt.clf()
                plt.cla()
                plt.close('all')
                plt.close(fig1)
                
                #plt.show() n_mag_bins
            #plt.savefig(savepath+savename)
            
        i+=1
        
def elec_epoch(plot='epoch',no_enc_7=False, no_enc_1=False, win_len=1, by_Rs=False,resolution=10):
    
    i=0
    
    enc_num = len(per_flt)
    
    while i <= enc_num:
        
        Rs = 6.957e5 #solar radius in km
        w = 2*np.pi/(25.38*86400) # angular frequency of the sun in degrees/sec
        
        v = 400000 #m/s typical slow solar wind speed, may replace later with measurement values
        mu = 4*np.pi*1e-7 #mu naught
        eVtoJ = 1.60218*1e-19 #eV to J
        #kb = 1.380649*10e-23 #boltzmann constant, J/K
        
        if not by_Rs:
            
            if i==0:
                csv_filename = 'harmwave_master_arch.csv'
                csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
                strahl_path = '/Users/besh2109/Desktop/psp_strahl_width/'
                name = 'All Encounters'
                if no_enc_7:
                    name = name+' sans 7'
                if no_enc_1:
                    name = name+' sans 1'

            else:
                csv_filename = 'enc_'+str(i)+'_harmwave_arch.csv'
                csv_path = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/Enc'+str(i)+'/'
                strahl_path = '/Users/besh2109/Desktop/psp_strahl_width/'
                name = 'Encounter '+str(i)
        
        else:
            csv_filename = 'harmwave_master_arch.csv'
            csv_path='/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
            strahl_path = '/Users/besh2109/Desktop/psp_strahl_width/'
            strahl_file = 'Enc'+str(i)+'_PRELIMSTRAHLWIDTH.csv'
            
            if i==0:
                name = 'All Radial Distances'
                if no_enc_7:
                    name = name+' sans Enc 7'
                if no_enc_1:
                    name = name+' sans Enc 1'

            else:
                name = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+' Rs'
        
        isfile = os.path.isfile(csv_path+csv_filename)
        
        if isfile:
            
            df = pd.read_csv(csv_path+csv_filename)
            bf = df.to_numpy()

            af = np.delete(bf,bf[:,1]<90,0)
            
            if by_Rs and i !=0:
                af = np.delete(af,af[:,2]>Rs_grps[i-1][0],0)
                af = np.delete(af,af[:,2]<Rs_grps[i-1][1],0)
            
            if no_enc_7 and i != enc_num: #this portion of code kills events in encounter 7
                dates = list(af[:,0])
                for j in range(len(dates)):
                    dates[j] = dates[j][0:4]
                npdates = np.array(dates)
                af = np.delete(af,npdates=='2021',0)
            
            if no_enc_1 and i != 1: #this portion of code kills events in encounter 1
                dates = list(af[:,0])
                for j in range(len(dates)):
                    dates[j] = dates[j][0:4]
                npdates = np.array(dates)
                af = np.delete(af,npdates=='2018',0)
                
            
            wave_start = np.array(pys.time_float(af[:,0]))
            wave_end = np.array(wave_start+af[:,1])
            
            window_start = np.array(wave_start - win_len*af[:,1])#*(1/3)) #complete epoch analysis for window larger than wave itself
            window_end = np.array(wave_end + win_len*af[:,1])#*(1/3))     #want to see if theres a difference between times during and before/after events
            
            dates = list(af[:,0])
            for j in range(len(dates)):
                dates[j] = dates[j][0:10]

            uniq_dates = np.unique(dates)
            date_flt = pys.time_float(uniq_dates)
            uniq_next = pys.time_string(np.array(date_flt)+86400.)
            
            duration = np.array(af[:,1])
            
                            
            sw_time = []
            sw_data = []
            sw_len_arr = []
            sw_dur_med_arr = []
            
            drift_time = []
            drift_data = []
            drift_len_arr = []
            drift_dur_med_arr = []
            
            tpar_time = []
            tpar_data = []
            tpar_len_arr = []
            
            tper_time = []
            tper_data = []
            tper_len_arr = []
            
            temp_data = []
            temp_len_arr = []
            #print(len(af[:,0]))
            
            for j in range(len(uniq_dates)): #gather data for each day range(2):#

                """ PSP Positional data """
                # pys.psp.fields(trange=[uniq_dates[j],uniq_next[j]], datatype='ephem_eclipj2000', level='l1') #going to be used to plot parker position
                # pos_data = pyt.get_data('position')

                # pos_time_arr = pos_data[0]
                # pos_data_arr = pos_data[1]
                
                """ PSP electron data """
            
                #strahl widths, or sw
                if i==0 or by_Rs==True:
                    sw_time_arr = []
                    sw_data_arr = []
                    for l in range(1,8):
                        strahl_file = 'Enc'+str(l)+'_PRELIMSTRAHLWIDTH.csv'
                        #print(strahl_file)
                        sw_df = pd.read_csv(strahl_path+strahl_file)
                        sw_csv = sw_df.to_numpy()
                        sw_time_tmp0 = list(sw_csv[:,0])
                        sw_data_tmp0 = list(sw_csv[:,1])
                        
                        
                        sw_time_arr = sw_time_arr + sw_time_tmp0
                        sw_data_arr = sw_data_arr + sw_data_tmp0
                    sw_time_arr = np.array(sw_time_arr)
                    sw_data_arr = np.array(sw_data_arr)

                else:
                    strahl_file = 'Enc'+str(i)+'_PRELIMSTRAHLWIDTH.csv'
                    sw_df = pd.read_csv(strahl_path+strahl_file)
                    sw_csv = sw_df.to_numpy()
                    
                    sw_time_arr = sw_csv[:,0]
                    sw_data_arr = sw_csv[:,1]
                    
                #core drifts, or drift
                
                core_path = '/Users/besh2109/Desktop/psp_core_drift/'
                core_file = 'coredrift_e1toe8.tplot'
                
                pyt.tplot_restore(core_path+core_file)
                
                drift_data_tmp = pyt.get_data('coredrift')
                
                drift_time_arr = drift_data_tmp[0]
                drift_data_arr = drift_data_tmp[1]
                
                #core temperature, perp and parallel
                
                temp_path = '/Users/besh2109/Desktop/psp_core_temp/'
                temp_file = 'coret_e1toe8.tplot'
                
                par_data_tmp = pyt.get_data('coretpar')
                per_data_tmp = pyt.get_data('coretperp')
                
                tpar_time_arr = par_data_tmp[0]
                tpar_data_arr = par_data_tmp[1]
                
                tper_time_arr = per_data_tmp[0]
                tper_data_arr = per_data_tmp[1]

                """ date management """
                
                date_strt_flt = pys.time_float(uniq_dates[j])
                date_end_flt = pys.time_float(uniq_next[j])
                
                where = np.where(np.logical_and(wave_start>date_strt_flt,wave_start<date_end_flt))
                
                win_start_tmp = np.array(window_start[where])
                win_end_tmp = np.array(window_end[where])
                wv_start = np.array(wave_start[where])
                wv_end = np.array(wave_end[where])
                wv_dur = np.array(duration[where])

                
                #r_data = []
                #print(len(win_start_tmp))
                for k in range(len(win_start_tmp)): 
          
                    """ strahl data """
                    sw_where = np.where((sw_time_arr > win_start_tmp[k]) & (sw_time_arr < win_end_tmp[k]))
                    sw_where = sw_where[0]
                    
                    dur_where = np.where((sw_time_arr > wv_start[k]) & (sw_time_arr < wv_end[k]))
                    dur_where = dur_where[0]
                    #print(sw_where)
                    
                    sw_ti = np.array(sw_time_arr[sw_where])
                    sw_data_tmp = np.array(sw_data_arr[sw_where])
                    sw_dur_med = np.median(sw_data_arr[dur_where])
                    
                    sw_time.append(sw_ti)
                    sw_data.append(sw_data_tmp)
                    sw_len_arr.append(len(sw_ti))
                    sw_dur_med_arr.append(sw_dur_med)
                    
                    """ core drift data """
                    
                    drift_where = np.where((drift_time_arr > win_start_tmp[k]) & (drift_time_arr < win_end_tmp[k]))
                    drift_where = drift_where[0]
                    
                    dur_where = np.where((drift_time_arr > wv_start[k]) & (drift_time_arr < wv_end[k]))
                    dur_where = dur_where[0]
                    
                    
                    drift_ti = np.array(drift_time_arr[drift_where])
                    drift_data_tmp = np.array(drift_data_arr[drift_where])
                    drift_dur_med = np.median(drift_data_arr[dur_where])
                    
                    drift_time.append(drift_ti)
                    drift_data.append(drift_data_tmp)
                    drift_len_arr.append(len(drift_ti))
                    drift_dur_med_arr.append(drift_dur_med)
                    #print(len(drift_ti))
                    
                    """ core temp data """
                    
                    tpar_where = np.where((tpar_time_arr > win_start_tmp[k]) & (tpar_time_arr < win_end_tmp[k]))
                    tpar_where = tpar_where[0]
                    
                    tpar_ti = np.array(tpar_time_arr[tpar_where])
                    tpar_data_tmp = np.array(tpar_data_arr[tpar_where])
                    
                    tper_where = np.where((tper_time_arr > win_start_tmp[k]) & (tper_time_arr < win_end_tmp[k]))
                    tper_where = tper_where[0]
                    
                    tper_ti = np.array(tper_time_arr[tper_where])
                    tper_data_tmp = np.array(tper_data_arr[tper_where])
                    
                    tpar_time.append(tpar_ti)
                    tpar_data.append(tpar_data_tmp)
                    tpar_len_arr.append(len(tpar_ti))
                    
                    tper_time.append(tper_ti)
                    tper_data.append(tper_data_tmp)
                    tper_len_arr.append(len(tper_ti))
                    
                    temp_data.append((2*tper_data_tmp+tpar_data_tmp)/3)
                    temp_len_arr.append(len(tpar_ti))
            
                    """ #position data """
                    
                    # wave_center = (win_end_tmp[k]+win_start_tmp[k])/2
                    # time_dif = np.array(abs(pos_time_arr-wave_center))
                    # pos_where = np.where(time_dif == min(time_dif))
                    # pos_where = pos_where[0][0]

                    # r_km = np.sqrt(pos_data_arr[pos_where,0]**2+pos_data_arr[pos_where,1]**2+pos_data_arr[pos_where,2]**2)
                    # r_data.append(r_km)
            

            
            time_tmp = []
            data_tmp = []
            len_tmp = []
            med_tmp = []
            for m in range(len(sw_len_arr)):
                if sw_len_arr[m] >= resolution:
                    time_tmp.append(sw_time[m])
                    data_tmp.append(sw_data[m])
                    len_tmp.append(sw_len_arr[m])
                    med_tmp.append(sw_dur_med_arr[m])
            
            sw_time = list(time_tmp)
            sw_data = list(data_tmp)
            sw_len_arr = np.array(len_tmp)
            sw_dur_med_arr = np.array(med_tmp)
            
            min_sw_len = np.nanmin(sw_len_arr)
            n_sw_bins = min_sw_len
            #print(n_sw_bins)
            for l in range(len(sw_time)): #this block seeks to normalize all the magnetic field data 
                                           #to the length of the shortest window

                sw_bin_size = sw_len_arr[l]/n_sw_bins
                sw_ind_arr = np.arange(sw_len_arr[l])
                norm_sw_time = np.zeros((n_sw_bins,))
                norm_sw_val = np.zeros((n_sw_bins,))

                for m in range(n_sw_bins):
                    
                    win_str = m*sw_bin_size
                    win_end = (m+1)*sw_bin_size
                    wind_where = np.where((sw_ind_arr>=win_str)&(sw_ind_arr<win_end))
                    wind_where = wind_where[0]
                    sw_time_val_tmp = np.median(sw_time[l][wind_where])
                    sw_val_tmp = np.median(sw_data[l][wind_where])
                    
                    norm_sw_time[m] = sw_time_val_tmp
                    norm_sw_val[m] = sw_val_tmp

    
                sw_time[l] = np.array(norm_sw_time)                
                sw_data[l] = np.array(norm_sw_val)
            
            #print(drift_len_arr)
            dr_time_tmp = []
            dr_data_tmp = []
            dr_len_tmp = []
            dr_med_tmp = []
            for m in range(len(drift_len_arr)):
                if drift_len_arr[m] >= 3 and not np.isnan(np.max(drift_data[m])):
                    dr_time_tmp.append(drift_time[m])
                    dr_data_tmp.append(drift_data[m])
                    dr_len_tmp.append(drift_len_arr[m])
                    dr_med_tmp.append(drift_dur_med_arr[m])
            
            drift_time = list(dr_time_tmp)
            drift_data = list(dr_data_tmp)
            drift_len_arr = np.array(dr_len_tmp)
            drift_dur_med_arr = np.array(dr_med_tmp)
            
            tpa_time_tmp = []
            tpa_data_tmp = []
            tpa_len_tmp = []
            
            tpe_time_tmp = []
            tpe_data_tmp = []
            tpe_len_tmp = []

            tem_data_tmp = []
            tem_len_tmp = []

            for m in range(len(tpar_len_arr)):
                if tpar_len_arr[m] >= 3 and not np.isnan(np.max(tpar_data[m])):
                    tpa_time_tmp.append(tpar_time[m])
                    tpa_data_tmp.append(tpar_data[m])
                    tpa_len_tmp.append(tpar_len_arr[m])

                    tpe_time_tmp.append(tper_time[m])
                    tpe_data_tmp.append(tper_data[m])
                    tpe_len_tmp.append(tper_len_arr[m])

                    tem_data_tmp.append(temp_data[m])
                    tem_len_tmp.append(temp_len_arr[m])
            
            tpar_time = list(tpa_time_tmp)
            tpar_data = list(tpa_data_tmp)
            tpar_len_arr = np.array(tpa_len_tmp)
            
            tper_time = list(tpe_time_tmp)
            tper_data = list(tpe_data_tmp)
            tper_len_arr = np.array(tpe_len_tmp)
            
            temp_data = list(tem_data_tmp)
            temp_len_arr = np.array(tem_len_tmp)

            
            if len(drift_len_arr) != 0:
                    
                
                min_drift_len = np.nanmin(drift_len_arr)
                n_drift_bins = min_drift_len
                
                min_tpar_len = np.nanmin(tpar_len_arr)
                n_tpar_bins = min_tpar_len
                
                min_tper_len = np.nanmin(tper_len_arr)
                n_tper_bins = min_tper_len
                
                min_temp_len = np.nanmin(temp_len_arr)
                n_temp_bins = min_temp_len
    
                for l in range(len(drift_time)): #this block seeks to normalize all the magnetic field data 
                                               #to the length of the shortest window
    
                    drift_bin_size = drift_len_arr[l]/n_drift_bins
                    drift_ind_arr = np.arange(drift_len_arr[l])
                    norm_drift_time = np.zeros((n_drift_bins,))
                    norm_drift_val = np.zeros((n_drift_bins,))
    
                    for m in range(n_drift_bins):
                        
                        win_str = m*drift_bin_size
                        win_end = (m+1)*drift_bin_size
                        wind_where0 = np.where((drift_ind_arr>=win_str)&(drift_ind_arr<win_end))
                        wind_where0 = wind_where0[0]
                        drift_time_val_tmp = np.median(drift_time[l][wind_where0])
                        drift_val_tmp = np.median(drift_data[l][wind_where0])
                        
                        norm_drift_time[m] = drift_time_val_tmp
                        norm_drift_val[m] = drift_val_tmp
    
        
                    drift_time[l] = np.array(norm_drift_time)                
                    drift_data[l] = np.array(norm_drift_val)
                    
                for l in range(len(tpar_time)):
                    
                    tpar_bin_size = tpar_len_arr[l]/n_tpar_bins
                    tpar_ind_arr = np.arange(tpar_len_arr[l])
                    norm_tpar_time = np.zeros((n_tpar_bins,))
                    norm_tpar_val = np.zeros((n_tpar_bins,))
                    
                    for m in range(n_tpar_bins):
                        
                        win_str = m*tpar_bin_size
                        win_end = (m+1)*tpar_bin_size
                        wind_where1 = np.where((tpar_ind_arr>=win_str)&(tpar_ind_arr<win_end))
                        wind_where1 = wind_where1[0]
                        tpar_time_val_tmp = np.median(tpar_time[l][wind_where1])
                        tpar_val_tmp = np.median(tpar_data[l][wind_where1])
                        
                        norm_tpar_time[m] = tpar_time_val_tmp
                        norm_tpar_val[m] = tpar_val_tmp
                        
                    
                    tpar_time[l] = np.array(norm_tpar_time)                
                    tpar_data[l] = np.array(norm_tpar_val)
                    
                    tper_bin_size = tper_len_arr[l]/n_tper_bins
                    tper_ind_arr = np.arange(tper_len_arr[l])
                    norm_tper_time = np.zeros((n_tper_bins,))
                    norm_tper_val = np.zeros((n_tper_bins,))
                    
                    for m in range(n_tper_bins):
                        
                        win_str = m*tper_bin_size
                        win_end = (m+1)*tper_bin_size
                        wind_where2 = np.where((tper_ind_arr>=win_str)&(tper_ind_arr<win_end))
                        wind_where2 = wind_where2[0]
                        tper_time_val_tmp = np.median(tper_time[l][wind_where2])
                        tper_val_tmp = np.median(tper_data[l][wind_where2])
                        
                        norm_tper_time[m] = tper_time_val_tmp
                        norm_tper_val[m] = tper_val_tmp
                    
                    
                    tper_time[l] = np.array(norm_tper_time)                
                    tper_data[l] = np.array(norm_tper_val)
                    
                    temp_bin_size = temp_len_arr[l]/n_temp_bins
                    temp_ind_arr = np.arange(temp_len_arr[l])
                    norm_temp_time = np.zeros((n_temp_bins,))
                    norm_temp_val = np.zeros((n_temp_bins,))
                    
                    for m in range(n_temp_bins):
                        
                        win_str = m*temp_bin_size
                        win_end = (m+1)*temp_bin_size
                        wind_where3 = np.where((temp_ind_arr>=win_str)&(temp_ind_arr<win_end))
                        wind_where3 = wind_where3[0]
                        # temp_time_val_tmp = np.median(tpar_time[l][wind_where3])
                        temp_val_tmp = np.median(temp_data[l][wind_where3])
                        
                        #norm_temp_time[m] = temp_time_val_tmp
                        norm_temp_val[m] = temp_val_tmp
                                        
                    temp_data[l] = np.array(norm_temp_val)
                    
                if plot == 'epoch' or plot == 'all':
    
                    savepath = '/Users/besh2109/Desktop/PSP_epoch/strahl_epoch/'
    
                    if not by_Rs:
                        if i==0:
                            savename = 'All_Events_strahl_epoch'
                            if no_enc_7:
                                savename = savename+'_no_7'
                            if no_enc_1:
                                savename = savename+'_no_1'
                            
                            savename = savename+'.png'    
                        else:
                            savename = 'Enc_'+str(i)+'_strahl_epoch.png'
                    else:
                        if i==0:
                            savename = 'All_Events_strahl_epoch'
                            if no_enc_7:
                                savename = savename+'_no_7'
                            if no_enc_1:
                                savename = savename+'_no_1'
                            
                            savename = savename+'.png'       
                        else:
                            savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs_strahl_epoch.png'
                
                    fig1 = plt.figure(figsize=(15,25))
                    axs1 = fig1.add_subplot(511)
                    
                    ind_hist = []
                    data_hist = []
                    for o in range(len(sw_data)):
                        ind = list(range(len(sw_data[o])))
                        data = list(sw_data[o])#-sw_dur_med_arr[o])
                        ind_hist = ind_hist + ind
                        data_hist = data_hist + data
                        #axs1.plot(o)
                    ind_hist_arr = np.array(ind_hist)
                    data_hist_arr = np.array(data_hist)
                    
                                
                    histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_sw_bins,60])
                    histo = np.transpose(histo)
                    histo[histo==0]=np.nan
        
                    r_color = axs1.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                    
                    #height = [-0.6,-0.6]
                    #endpoints = [n_vel_bins/3,2*n_vel_bins/3]
                    #axs1.scatter(endpoints,height, marker='|',s=75000, color='lime',linewidths=4, zorder=len(mag_r_data)+1)
                    axs1.set(title='Strahl Widths minus median '+name)
                    #axs1.set_ylim(-25,25)
                    axs1.set_ylabel('Degrees')
                    #axs1.set_xlabel('Normalized time')
                    #plt.title()
                    #plt.show()
                    box = axs1.get_position()
                    axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                    fig1.colorbar(r_color,cax=axColor, label='Counts')
                    
                    axs2 = fig1.add_subplot(512)
                    
                    ind_hist = []
                    data_hist = []
                    for o in range(len(drift_data)):
                        ind = list(range(len(drift_data[o])))
                        data = list(np.abs(drift_data[o]))#-sw_dur_med_arr[o])
                        ind_hist = ind_hist + ind
                        data_hist = data_hist + data
                        #axs1.plot(o)
                    ind_hist_arr = np.array(ind_hist)
                    data_hist_arr = np.array(data_hist)
                    
                                
                    histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_drift_bins,60])
                    histo = np.transpose(histo)
                    histo[histo==0]=np.nan
        
                    r_color = axs2.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                    
                    axs2.set(title='Core Drift '+name)
                    axs2.set_ylabel('km/s')
    
                    box = axs2.get_position()
                    axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                    fig1.colorbar(r_color,cax=axColor, label='Counts')
                    
                    
                    axs3 = fig1.add_subplot(513)
                    
                    ind_hist = []
                    data_hist = []
                    for o in range(len(tpar_data)):
                        ind = list(range(len(tpar_data[o])))
                        data = list(tpar_data[o]/temp_data[o])#-sw_dur_med_arr[o])
                        ind_hist = ind_hist + ind
                        data_hist = data_hist + data
                        #axs1.plot(o)
                    ind_hist_arr = np.array(ind_hist)
                    data_hist_arr = np.array(data_hist)
                    
                                
                    histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_tpar_bins,50])
                    histo = np.transpose(histo)
                    histo[histo==0]=np.nan
        
                    r_color = axs3.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                    
                    axs3.set(title='Tpar/Ttot '+name)
                    axs3.set_ylabel('Tpar/Ttot')
    
                    box = axs3.get_position()
                    axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                    fig1.colorbar(r_color,cax=axColor, label='Counts')   
                    
                    axs4 = fig1.add_subplot(514)
                    
                    ind_hist = []
                    data_hist = []
                    for o in range(len(tper_data)):
                        ind = list(range(len(tper_data[o])))
                        data = list(tper_data[o]/temp_data[o])#-sw_dur_med_arr[o])
                        ind_hist = ind_hist + ind
                        data_hist = data_hist + data
                        #axs1.plot(o)
                    ind_hist_arr = np.array(ind_hist)
                    data_hist_arr = np.array(data_hist)
                    
                                
                    histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_tper_bins,50])
                    histo = np.transpose(histo)
                    histo[histo==0]=np.nan
        
                    r_color = axs4.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                    
                    axs4.set(title='Tper/Ttot '+name)
                    axs4.set_ylabel('Tper/Ttot')
    
                    box = axs4.get_position()
                    axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                    fig1.colorbar(r_color,cax=axColor, label='Counts')
                    
                    axs5 = fig1.add_subplot(515)
                    
                    ind_hist = []
                    data_hist = []
                    for o in range(len(temp_data)):
                        ind = list(range(len(temp_data[o])))
                        data = list(tper_data[o]/tpar_data[o])#-sw_dur_med_arr[o])
                        ind_hist = ind_hist + ind
                        data_hist = data_hist + data
                        #axs1.plot(o)
                    ind_hist_arr = np.array(ind_hist)
                    data_hist_arr = np.array(data_hist)
                    
                                
                    histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_tpar_bins,50])
                    histo = np.transpose(histo)
                    histo[histo==0]=np.nan
        
                    r_color = axs5.pcolormesh(xedge,yedge,histo, cmap='jet')  #,np.log10(histo)
                    
                    axs5.set(title='Temperature Anisotropy '+name)
                    axs5.set_ylabel('Tper/Tpar')
    
                    box = axs5.get_position()
                    axColor= plt.axes([box.x0*1.01 + box.width * 1.01, box.y0, 0.01, box.height])
                    fig1.colorbar(r_color,cax=axColor, label='Counts')
                    
                    plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                    plt.clf()
                    plt.cla()
                    plt.close('all')
                    plt.close(fig1)
                    
                if plot == 'dist' or plot == 'all':
    
                    savepath = '/Users/besh2109/Desktop/PSP_epoch/strahl_dist/'
    
                    if not by_Rs:
                        if i==0:
                            savename = 'All_Events_strahl_dist'
                            if no_enc_7:
                                savename = savename+'_no_7'
                            if no_enc_1:
                                savename = savename+'_no_1'
                            
                            savename = savename+'.png'    
                        else:
                            savename = 'Enc_'+str(i)+'_strahl_dist.png'
                    else:
                        if i==0:
                            savename = 'All_Events_strahl_dist'
                            if no_enc_7:
                                savename = savename+'_no_7'
                            if no_enc_1:
                                savename = savename+'_no_1'
                            
                            savename = savename+'.png'       
                        else:
                            savename = str(Rs_grps[i-1][0])+'-'+str(Rs_grps[i-1][1])+'_Rs_strahl_dist.png'
                
                    fig1 = plt.figure(figsize=(15,15))
                    axs1 = fig1.add_subplot(221)
                    
                    ind_hist = []
                    data_hist = []
                    for o in range(len(sw_data)):
                        ind = list(range(len(sw_data[o])))
                        data = list(sw_data[o])#-sw_dur_med_arr[o])
                        ind_hist = ind_hist + ind
                        data_hist = data_hist + data
                        #axs1.plot(o)
                    ind_hist_arr = np.array(ind_hist)
                    data_hist_arr = np.array(data_hist)
                    
                                
                    histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_sw_bins,100])
                    histo = np.transpose(histo)
                    #histo[histo==0]=np.nan
        
                    
                    k1=0
                    j1=0
                    dist1sw = np.zeros(histo[:,0].shape)
                    while k1 < round(n_sw_bins/3): #distribution 1
                        dist1sw = dist1sw + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    
                    print(j1)
                    j1=0
                    k1=round(n_sw_bins/3)
                    dist2sw = np.zeros(histo[:,0].shape)
                    while k1 >= round(n_sw_bins/3) and k1 < round(2*n_sw_bins/3): #distribution 2
                        dist2sw = dist2sw + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    print(j1)
                    j1=0
                    k1=round(2*n_sw_bins/3)
                    dist3sw = np.zeros(histo[:,0].shape)
                    while k1 >= round(2*n_sw_bins/3) and k1<n_sw_bins: #distribution 3
                        dist3sw = dist3sw + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    print(j1)
                    
                    #ax2.set_aspect(1)
                    #ax2.set_adjustable('box')
                    norm_val = np.nanmax(dist2sw)
                    axs1.plot(yedge[1:len(yedge)],dist1sw/norm_val,color='red',label='before wave')
                    axs1.plot(yedge[1:len(yedge)],dist2sw/norm_val,color='green',label='during wave')
                    axs1.plot(yedge[1:len(yedge)],dist3sw/norm_val,color='blue',label='after wave')
                    axs1.legend()
                    #ax2.set_ylim([0,26])
                    axs1.set_title('Strahl Widths minus median '+name)
                    axs1.set_ylabel('Normalized Counts Distribution')
                    axs1.set_xlabel('Degrees')
                    axs1.set_adjustable('box')
                    
                    axs2 = fig1.add_subplot(222)
                    
                    ind_hist = []
                    data_hist = []
                    for o in range(len(drift_data)):
                        ind = list(range(len(drift_data[o])))
                        data = list(np.abs(drift_data[o]))#-sw_dur_med_arr[o])
                        ind_hist = ind_hist + ind
                        data_hist = data_hist + data
                        #axs1.plot(o)
                    ind_hist_arr = np.array(ind_hist)
                    data_hist_arr = np.array(data_hist)
                    
                                
                    histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_drift_bins,100])
                    histo = np.transpose(histo)
                    #histo[histo==0]=np.nan
        
                    
                    k1=0
                    j1=0
                    dist1dr = np.zeros(histo[:,0].shape)
                    while k1 < math.ceil(n_drift_bins/3): #distribution 1
                        dist1dr = dist1dr + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    
                    print(j1)
                    j1=0
                    k1=math.ceil(n_drift_bins/3)
                    dist2dr = np.zeros(histo[:,0].shape)
                    while k1 >= math.floor(n_drift_bins/3) and k1 < math.ceil(2*n_drift_bins/3): #distribution 2
                        dist2dr = dist2dr + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    print(j1)
                    j1=0
                    k1=math.ceil(2*n_drift_bins/3)
                    dist3dr = np.zeros(histo[:,0].shape)
                    while k1 >= math.floor(2*n_drift_bins/3) and k1<n_drift_bins: #distribution 3
                        dist3dr = dist3dr + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    print(j1)
                    
                    #ax2.set_aspect(1)
                    #ax2.set_adjustable('box')
                    norm_val = np.nanmax(dist2dr)
                    axs2.plot(yedge[1:len(yedge)],dist1dr/norm_val,color='red',label='before wave')
                    axs2.plot(yedge[1:len(yedge)],dist2dr/norm_val,color='green',label='during wave')
                    axs2.plot(yedge[1:len(yedge)],dist3dr/norm_val,color='blue',label='after wave')
                    axs2.legend()
                    #ax2.set_ylim([0,26])
                    axs2.set_title('Core Drift '+name)
                    axs2.set_ylabel('Normalized Counts Distribution')
                    axs2.set_xlabel('km/s')
                    axs2.set_adjustable('box')
                    
                    
                    axs3 = fig1.add_subplot(223)
                    
                    ind_hist = []
                    data_hist = []
                    for o in range(len(tpar_data)):
                        ind = list(range(len(tpar_data[o])))
                        data = list(tpar_data[o]/temp_data[o])#-sw_dur_med_arr[o])
                        ind_hist = ind_hist + ind
                        data_hist = data_hist + data
                        #axs1.plot(o)
                    ind_hist_arr = np.array(ind_hist)
                    data_hist_arr = np.array(data_hist)
                    
                                
                    histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_tpar_bins,50])
                    histo = np.transpose(histo)
                    #histo[histo==0]=np.nan
        
                    
                    k1=0
                    j1=0
                    dist1dr = np.zeros(histo[:,0].shape)
                    while k1 < math.ceil(n_tpar_bins/3): #distribution 1
                        dist1dr = dist1dr + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    
                    print(j1)
                    j1=0
                    k1=math.ceil(n_tpar_bins/3)
                    dist2dr = np.zeros(histo[:,0].shape)
                    while k1 >= math.floor(n_tpar_bins/3) and k1 < math.ceil(2*n_tpar_bins/3): #distribution 2
                        dist2dr = dist2dr + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    print(j1)
                    j1=0
                    k1=math.ceil(2*n_tpar_bins/3)
                    dist3dr = np.zeros(histo[:,0].shape)
                    while k1 >= math.floor(2*n_tpar_bins/3) and k1<n_tpar_bins: #distribution 3
                        dist3dr = dist3dr + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    print(j1)
                    
                    #ax2.set_aspect(1)
                    #ax2.set_adjustable('box')
                    norm_val = np.nanmax(dist2dr)
                    axs3.plot(yedge[1:len(yedge)],dist1dr/norm_val,color='red',label='before wave')
                    axs3.plot(yedge[1:len(yedge)],dist2dr/norm_val,color='green',label='during wave')
                    axs3.plot(yedge[1:len(yedge)],dist3dr/norm_val,color='blue',label='after wave')
                    axs3.legend()
                    #ax2.set_ylim([0,26])
                    axs3.set_title('Tpar/Ttot '+name)
                    axs3.set_ylabel('Normalized Counts Distribution')
                    axs3.set_xlabel('Tpar/Ttot')
                    axs3.set_adjustable('box')
                    
                    axs4 = fig1.add_subplot(224)
                    
                    ind_hist = []
                    data_hist = []
                    for o in range(len(tper_data)):
                        ind = list(range(len(tper_data[o])))
                        data = list(temp_data[o])#-sw_dur_med_arr[o])
                        ind_hist = ind_hist + ind
                        data_hist = data_hist + data
                        #axs1.plot(o)
                    ind_hist_arr = np.array(ind_hist)
                    data_hist_arr = np.array(data_hist)
                    
                                
                    histo,xedge,yedge = np.histogram2d(ind_hist_arr,data_hist_arr,bins=[n_tper_bins,50])
                    histo = np.transpose(histo)
                    #histo[histo==0]=np.nan
        
                    
                    k1=0
                    j1=0
                    dist1dr = np.zeros(histo[:,0].shape)
                    while k1 < math.ceil(n_tper_bins/3): #distribution 1
                        dist1dr = dist1dr + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    
                    print(j1)
                    j1=0
                    k1=math.ceil(n_tper_bins/3)
                    dist2dr = np.zeros(histo[:,0].shape)
                    while k1 >= math.floor(n_tper_bins/3) and k1 < math.ceil(2*n_tper_bins/3): #distribution 2
                        dist2dr = dist2dr + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    print(j1)
                    j1=0
                    k1=math.ceil(2*n_tper_bins/3)
                    dist3dr = np.zeros(histo[:,0].shape)
                    while k1 >= math.floor(2*n_tper_bins/3) and k1<n_tper_bins: #distribution 3
                        dist3dr = dist3dr + histo[:,k1]/np.sum(histo[:,k1])
                        k1+=1
                        j1+=1
                    print(j1)
                    
                    #ax2.set_aspect(1)
                    #ax2.set_adjustable('box')
                    norm_val = np.nanmax(dist2dr)
                    axs4.plot(yedge[1:len(yedge)],dist1dr/norm_val,color='red',label='before wave')
                    axs4.plot(yedge[1:len(yedge)],dist2dr/norm_val,color='green',label='during wave')
                    axs4.plot(yedge[1:len(yedge)],dist3dr/norm_val,color='blue',label='after wave')
                    axs4.legend()
                    #ax2.set_ylim([0,26])
                    axs4.set_title('Tper/Ttot '+name)
                    axs4.set_ylabel('Normalized Counts Distribution')
                    axs4.set_xlabel('Tper/Ttot')
                    axs4.set_adjustable('box')
                    
         
                    plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
                    plt.clf()
                    plt.cla()
                    plt.close('all')
                    plt.close(fig1)
        i+=1