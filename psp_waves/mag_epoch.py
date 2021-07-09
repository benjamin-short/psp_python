#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 18 13:35:06 2021

@author: besh2109
"""
import numpy as np
from scipy.io import readsav
import pyspedas as pys
import pytplot as pyt
import os
import pandas as pd
from statistics import median
import math
import matplotlib.pyplot as plt

# from .config import CONFIG
# from .config import enc_flt
# from .config import per_flt
# from .config import per_dist_lst

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

Rs_grps = [[50,45],[45,40],[40,35],[35,30],[30,25],[25,20],[20,1]]

plot='epoch'
quick=True
no_enc_7=False
no_n_hat = False
win_len = 1
by_Rs=False

i=0
enc_num = len(per_flt)
while i <= enc_num:
    
    Rs = 6.957e5 #solar radius in km
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
        
        Br_data = []
        Bt_data = []
        Bn_data = []
        
        vel_v_data = []
        r_v_data = []
        t_v_data = []
        n_v_data = []
        cos_θ_def = [] #cosine of the deflection angle
        θ_def = [] #deflection angle
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
            Br = np.array(norm_mag_r_val)
            Bt = np.array(norm_mag_t_val)
            Bn = np.array(norm_mag_n_val)
            
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

            mag_r_data[l] =  np.array(norm_mag_r_val)
            mag_t_data[l] =  np.array(norm_mag_t_val)
            mag_n_data[l] =  np.array(norm_mag_n_val)
            
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
            dot_term = 1/np.sqrt(1+((w/v)**2)*(r_data[l])**2*np.sin(p_theta[l])**2)
            #r = 1.496*10**11
            
            phi_p = np.arctan(-w*r_data[l]/v)
            phi_m = np.arctan(np.array(norm_mag_t_val)/np.array(norm_mag_r_val))
            
            cos_θ = np.zeros((len(r_b),))
            phi_dif = np.zeros((len(r_b),))
            norm_vel_r_val = np.zeros((n_vel_bins,))
            vel_ind_arr = np.arange(vel_len_arr[l])
            cos_bin_size = len(r_b)/len(v)
            
            #print(v[0])
            for n in range(len(v)):
                win_str = math.ceil(n*cos_bin_size)
                win_end = math.ceil((n+1)*cos_bin_size)

                dot_term = 1/np.sqrt(1+((w/v[n])**2)*(r_data[l])**2*np.sin(p_theta[l])**2)
                cos_θ[win_str:win_end] = r_b[win_str:win_end]*dot_term - t_b[win_str:win_end]*dot_term*w*r_data[l]/v[n]*np.sin(p_theta[l])
                
                phi_dif[win_str:win_end] = phi_p[n] - phi_m[win_str:win_end]
                
            #cos_θ = np.array(r_b*dot_term - t_b*dot_term*w*r_data[l]/v*np.sin(p_theta[l]))
            
            cos_θ_def.append(cos_θ)
            θ_def.append(np.arccos(cos_θ)*180/np.pi)
            phi_def.append(phi_dif*180/np.pi)

        if plot =='epoch' or plot == 'both':
            
            savepath = '/Users/besh2109/Desktop/PSP_epoch/mag_epoch/'
            
            if not by_Rs:
                if i==0:
                    if no_enc_7:
                        savename = 'All_Enc_no_7_mag_epoch.png'
                    else:
                        savename = 'All_Enc_mag_epoch.png'      
                else:
                    savename = 'Enc_'+str(i)+'_mag_epoch.png'
            else:
                if i==0:
                    if no_enc_7:
                        savename = 'All_Rs_no_7_mag_epoch.png'
                    else:
                        savename = 'All_Rs_mag_epoch.png'      
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
            for o in cos_θ_def:
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
            axs4.set_ylim([0,90])
            axs4.set(title='Deflection Angle θ, normalized time')
            
            plt.show()
            #plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)       
            #plt.clf()
            #plt.cla()
            #plt.close('all')
            #plt.close(fis1)
        
        if plot == 'dist' or plot == 'both':
            
            savepath = '/Users/besh2109/Desktop/PSP_epoch/mag_dist/'
            
            if not by_Rs:
                if i==0:
                    if no_enc_7:
                        savename = 'All_Enc_no_7_mag_dist.png'
                    else:
                        savename = 'All_Enc_mag_dist.png'      
                else:
                    savename = 'Enc_'+str(i)+'_mag_dist.png'
            else:
                if i==0:
                    if no_enc_7:
                        savename = 'All_Rs_no_7_mag_dist.png'
                    else:
                        savename = 'All_Rs_mag_dist.png'      
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
            for o in cos_θ_def:
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
            ax4.set_xlim([0,70])
            #ax4.set_ylim([0,50])
            ax4.set_title('Parker-Measurement Deflection Angle')
            ax4.set_ylabel('Normalized Counts Distribution')
            ax4.set_xlabel('θ degrees')
            ax4.set_adjustable('box')
            
            plt.show()
            #plt.savefig(savepath+savename, bbox_inches = 'tight',pad_inches = 0.2)
            #plt.clf()
            #plt.cla()
            #plt.close('all')
            #plt.close(fig1)
        #plt.savefig(savepath+savename)
        
    i+=10