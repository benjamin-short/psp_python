#!/usr/bin/env pyshon3
# -*- coding: utf-8 -*-
"""
Created on Thu May 22 12:48:32 2025

@author: besh2109
"""

import os
import pyspedas.projects.psp as psp
import pyspedas as pys
import matplotlib.pyplot as plt
import numpy as np
import astropy.units as u
import astropy.constants as const

from .config import CONFIG
from .config import enc_flt
from .config import per_flt
from .config import per_dist_lst

import psp_regions.utils as utils

from pathlib import Path

fields_id = os.environ['PSP_FIELDS_ID']
fields_pass = os.environ['PSP_FIELDS_PW']

# sweap_id = os.environ['PSP_SWEAP_ID']
# sweap_pass = os.environ['PSP_SWEAP_PW']

sweap_id = os.environ['PSP_SWEAP_ID_BERK']
sweap_pass = os.environ['PSP_SWEAP_PW_BERK']

jsoc_email = os.environ['JSOC_EMAIL']

Rs = 6.957e5 #solar radius in km
Rs_in_m = Rs*10**3 #solar radius in m
w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec

def delb_b(enc='all',enc_radius=65,save=False):
    
    print('calculating ∂B/B')
    mu = 4*np.pi*1e-7 #mu naught
    eVtoJ = 1.60218*1e-19 #eV to J
    #kb = 1.380649*10e-23 #boltzmann constant, J/K
    
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc = list(range(1,19))
        
        title_mod = 'for All Encounters'
        
    elif enc == 'no 13':
        enc = list(range(1,13))
        enc.extend(range(14,19))
        title_mod = 'for All Encounters, no 1 or 13'
        
    elif enc == 'no 1':
        enc = list(range(2,15)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        title_mod = 'for All Encounters, no 1'
        
    elif type(enc) is int:
        enc = [enc]
        title_mod = 'for Encounter '+str(enc[0])
        
    elif type(enc) is str and len(enc) == 1:
        enc = int(enc)
        enc = [enc]
        title_mod = 'for Encounter '+str(enc[0])
    
    else:
        title_mod = 'for Encounters '+str(enc[0])+' thru '+str(enc[-1])
    
    #----------------------Organizing CSVs for in Data-------------------------#
    
    csv_path = str(Path('~/Desktop/Quiescent Region Connectivity/psp_regions/region_data/').expanduser())
    
    #--------------------------Start importing PSP Data-----------------------------#
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/'
    pys.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    hpos = pys.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    duration_list = []
    
    non_q_dB_B = []
    non_q_t_scale = []
    
    q_dB_B = []
    q_t_scale = []
    
    q_rs = []
    nq_rs = []
    
    
    for i in enc:
        
        csv_name = 'enc_'+str(i)+'_regions_raw.csv'
        
        q_df = pd.read_csv(csv_path+csv_name) #quiescent region Pandas dataframe
        
        q_starts = q_df['start dates'].to_numpy()
        q_ends = q_df['end dates'].to_numpy()
        q_duras = q_df['duration'].to_numpy()
        
        q_start_flt = np.array(pys.time_float(q_starts))
        q_end_flt = np.array(pys.time_float(q_ends))
        
        # q_counts+=len(q_starts) #count number of q regions

        enc_ind = (i-1)
        
        perihelion_flt = per_flt[enc_ind] #float date of perihelion
        perihelion = pys.time_string(perihelion_flt) #string date of perihelion
        
        enc_ends = enc_flt[enc_ind]
        
        enc_str = pys.time_string(enc_ends[0])
        enc_end = pys.time_string(enc_ends[1])
        
        
        # radius_mod = '<'+str(enc_radius)
        
        hpos_where = np.where((hpos_time_arr>enc_ends[0])&(hpos_time_arr<enc_ends[1]))
        hpos_where = hpos_where[0]
        
        hpos_time = hpos_time_arr[hpos_where]
        
        hposx_data = hpos_data_arr[hpos_where,0]
        hposy_data = hpos_data_arr[hpos_where,1]
        hposz_data = hpos_data_arr[hpos_where,2]
        
        R = (np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs)/Rs
        
        R_where = np.where(R<enc_radius)
        R_where = R_where[0]
        
        time_select = hpos_time[R_where]
        # t0p = pys.time_string(time_select[0],fmt='%Y%m%d_%H%M%S')
        # tfp = pys.time_string(time_select[-1],fmt='%Y%m%d_%H%M%S')
        
        # if not tmp_check:
        
        t0 = pys.time_string(time_select[0])
        tf = pys.time_string(time_select[-1])
        
        psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)
        
        pos_data = pys.get_data('psp_spi_SUN_DIST')
        pos_time = pos_data[0]
        pos_rs = pos_data[1]/Rs
        
        psp.fields(trange=[t0,tf],level='l2',datatype='mag_RTN_4_Sa_per_Cyc',last_version=True,username=fields_id,password=fields_pass)
        # psp.fields(trange=[t0,tf],level='l2',datatype='mag_RTN',last_version=True,username=fields_id,password=fields_pass)
        mag_data = pys.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
        # mag_data = pys.get_data('psp_fld_l2_mag_RTN')
        
        mag_time_arr = mag_data[0]
        mag_data_arr = mag_data[1]
        
        Br = mag_data_arr[:,0]
        Bt = mag_data_arr[:,1]
        Bn = mag_data_arr[:,2]
        
        #-------------get rid of enc 13 ICME---------------#
        
        icme_start = pys.time_float('2022-09-06/15:25')
        icme_end = pys.time_float('2022-09-07/12:25')
        
        # icme_where = np.where(((mag_time_arr>icme_start)&(mag_time_arr<icme_end)))
        icme_where = np.where(~((mag_time_arr>icme_start)&(mag_time_arr<icme_end)))
        icme_where = icme_where[0]
        icme_where_check = len(icme_where)
        
        if icme_where_check > 0:
            
            mag_time_arr = mag_time_arr[icme_where]
            # position = position[icme_where]
            # data = data[icme_where]
            Br = Br[icme_where]
            Bt = Bt[icme_where]
            Bn = Bn[icme_where]
            
            
        #-----------------------------------------------#
        
        q_inds = []
        cut_times = []
        
        for j in range(len(q_starts)):
            
            q_time_where = np.where((mag_time_arr>q_start_flt[j]) & (mag_time_arr<q_end_flt[j]))
            q_inds.extend(q_time_where[0])
            q_time_tmp = mag_time_arr[q_time_where]
            cut_times.extend(q_time_tmp)
            q_Br_tmp = Br[q_time_where]
            q_Bt_tmp = Bt[q_time_where]
            q_Bn_tmp = Bn[q_time_where]
            
            q_δΒr = q_Br_tmp - np.nanmean(q_Br_tmp)
            q_δBt = q_Bt_tmp - np.nanmean(q_Bt_tmp)
            q_δBn = q_Bn_tmp - np.nanmean(q_Bn_tmp)
            
            q_δB = np.sqrt(np.nanmean(q_δΒr**2+q_δBt**2+q_δBn**2))
            q_B_mag = np.sqrt(q_Br_tmp**2+q_Bt_tmp**2+q_Bn_tmp**2)
            
            q_B_mag_av = np.nanmean(q_B_mag)
            
            q_dB_B.append(q_δB/q_B_mag_av)
            q_t_scale.append(q_duras[j])
            
            q_time_span_where = np.where((pos_time>q_start_flt[j]) & (pos_time<q_end_flt[j]))
            q_time_span_where = q_time_span_where[0]
            q_rs_pos = pos_rs[q_time_span_where]
            q_rs.append(np.nanmean(q_rs_pos))

        non_q_time_arr = np.delete(mag_time_arr,q_inds)
        non_q_Br = np.delete(Br,q_inds)
        non_q_Bt = np.delete(Bt,q_inds)
        non_q_Bn = np.delete(Bn,q_inds)
        
        
        n = 2
        iii = 0
        while iii<n:
            
            for j in range(len(q_starts)):
                
                incheck = [True]
                ii = 0
                while any(incheck): #checking for data that is not in the quiescent region list
                    random_index = np.random.choice(non_q_time_arr)
                    non_q_where = np.where((mag_time_arr>random_index) & (mag_time_arr<(random_index+q_duras[j])))
                    non_q_time_tmp = mag_time_arr[non_q_where]
                    incheck = np.isin(non_q_time_tmp,cut_times)
                    
                    non_q_span_where = np.where((pos_time>random_index) & (pos_time<(random_index+q_duras[j])))
                    non_q_span_time_tmp = pos_time[non_q_span_where]
                    
                    
                    if ii >= 15: #if the random check is taking too long, then just look for spots where data of the proper length can be found
                        zeros = np.zeros(mag_time_arr.shape)
                        index_arr = np.array(range(len(mag_time_arr)))
                        del_where = np.where(np.isin(mag_time_arr,cut_times))
                        del_inds = del_where[0]
                        # time_check = np.delete(mag_time_arr,del_inds)
                        # ind_check = np.delete(index_arr,del_inds)
                        zeros[del_where] = 1
                        groups = np.array(utils.group_zeros(zeros))
                        times = mag_time_arr[groups]
                        
                        lengths = []
                        for ijk in mag_time_arr[groups]:
                            lengths.extend(np.diff(ijk))
                            
                        lengthwhere = np.where(lengths>q_duras[j])
                        lengthwhere = lengthwhere[0]
                        
                        group_tmp = groups[lengthwhere]
                        times_tmp = times[lengthwhere]
                        
                        random_index = np.random.choice(np.arange(len(group_tmp)))
                        time_index = times_tmp[random_index,0]
                        
                        non_q_where = np.where((mag_time_arr>time_index) & (mag_time_arr<(time_index+q_duras[j])))
                        non_q_time_tmp = mag_time_arr[non_q_where]
                        incheck = np.isin(non_q_time_tmp,cut_times)
                        
                        non_q_span_where = np.where((pos_time>time_index) & (pos_time<(time_index+q_duras[j])))
                        non_q_span_time_tmp = pos_time[non_q_span_where]
                        # breakpoint()
                        
                        
                        
                    ii+=1
                        
                        
                cut_times.extend(non_q_time_tmp)
                # print('done',ii)
                
                # non_q_time_tmp = mag_time_arr[non_q_time_where]
                non_q_Br_tmp = Br[non_q_where]
                non_q_Bt_tmp = Bt[non_q_where]
                non_q_Bn_tmp = Bn[non_q_where]
                
                non_q_δΒr = non_q_Br_tmp - np.nanmean(non_q_Br_tmp)
                non_q_δBt = non_q_Bt_tmp - np.nanmean(non_q_Bt_tmp)
                non_q_δBn = non_q_Bn_tmp - np.nanmean(non_q_Bn_tmp)
                
                non_q_δB = np.sqrt(np.nanmean(non_q_δΒr**2+non_q_δBt**2+non_q_δBn**2))
                non_q_B_mag = np.sqrt(non_q_Br_tmp**2+non_q_Bt_tmp**2+non_q_Bn_tmp**2)
                
                non_q_B_mag_av = np.nanmean(non_q_B_mag)
                
                non_q_dB_B.append(non_q_δB/non_q_B_mag_av)
                
                
                nq_rs_pos = pos_rs[non_q_span_where]
                nq_rs.append(np.nanmean(nq_rs_pos))
                # non_q_t_scale.append(q_duras[j])
            iii+=1
            
    x_datas = [non_q_dB_B,q_dB_B]
    r_datas = [nq_rs,q_rs]
    
    radial_bins = [[8,15],[15,25],[25,35],[35,45],[45,55],[55,65],[65,75]]

    # radial_bins = [[8,15],[15,25],[25,35],[35,45]]
    n_rads = len(radial_bins)

    r_labs_in = ['8','15','25','35']
    r_labs_out = ['15','25','35','45']
    
    r_labs_in.reverse()
    r_labs_out.reverse()

    labels = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']
    titles = ['Non-Quiescent Solar Wind','Quiescent Solar Wind']

    cmaps = ['autumn','autumn']
    colors = ['tab:blue','tab:orange']
    
    
    # fig = plt.figure(figsize=(20,60))
    
    # q_rs_arr = np.array(q_rs)
    # nq_rs_arr = np.array(nq_rs)
    
    # q_dB_arr = np.array(q_dB_B)
    # nq_dB_arr = np.array(non_q_dB_B)
    
    # for plot in range(n_rads):
        
    #     rad_bins = radial_bins[plot]
    #     axs = fig.add_subplot(n_rads,1,plot+1)
        
    #     axs.set_title("δB/|B| Histograms",fontsize=38)
    #     axs.set_ylabel('Counts',fontsize=38)
    #     axs.set_xlabel('δB/|B|',fontsize=38)
        
    #     q_rad_where = np.where((q_rs_arr>rad_bins[0])&(q_rs_arr<rad_bins[1]))
    #     q_rad_where = q_rad_where[0]
    #     non_q_rad_where = np.where((nq_rs_arr>rad_bins[0])&(nq_rs_arr<rad_bins[1]))
    #     non_q_rad_where = non_q_rad_where[0]
        
    #     q_data = q_dB_arr[q_rad_where]
    #     nq_data = nq_dB_arr[non_q_rad_where]
        
    #     bins = np.linspace(0,0.8,30)
    #     hist1, _ = np.histogram(nq_data, bins=bins)
    #     hist2, _ = np.histogram(q_data, bins=bins)
        
    #     error1 = np.sqrt(hist1) 
    #     error2 = np.sqrt(hist2) 
        
    #     # Plot histograms with error bars
    #     axs.bar(bins[:-1], hist1, width=np.diff(bins), align='center', alpha=0.5, label=labels[0],edgecolor='black')
    #     axs.bar(bins[:-1], hist2, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
    #     axs.errorbar(bins[:-1], hist1, yerr=error1, fmt='none', color='k', capsize=3)
    #     axs.errorbar(bins[:-1], hist2, yerr=error2, fmt='none', color='k', capsize=3)
        
    #     axs.tick_params(axis='both', which='major', labelsize=34)
    #     leg = axs.legend(fontsize=30,loc='upper right',markerscale=5)
        
    # plt.show()
        
        # print(q_rad_where.shape)

    fig = plt.figure(figsize=(30,15))
    # axs = fig.add_subplot(1,1,1)
    # print(j+k+1)
    # index = n_types*j+1+k
    # if k ==0:
    axs = fig.add_subplot(111)
    # axs = fig.add_subplot(n_rads,1,j+1)
    axs.set_title("δB/|B| Histograms",fontsize=38)
    axs.set_ylabel('Counts',fontsize=38)
    axs.set_xlabel('δB/|B|',fontsize=38)
    
    bins = np.linspace(0,0.8,30)
    hist1, _ = np.histogram(x_datas[0], bins=bins)
    hist2, _ = np.histogram(x_datas[1], bins=bins)
    
    # hist1_norm = hist1 / np.sum(hist1)
    # hist2_norm = hist2 / np.sum(hist2)
    
    hist1_norm = hist1
    hist2_norm = hist2
    
    q_mean = np.nanmean(q_dB_B)
    q_std = np.nanstd(q_dB_B)
    q_median = np.nanmedian(q_dB_B)
    non_q_mean = np.nanmean(non_q_dB_B)
    non_q_std = np.nanstd(non_q_dB_B)
    non_q_median = np.nanmedian(non_q_dB_B)

    
    # Calculate error bars for each bin (normalized)
    # error1 = np.sqrt(hist1) / np.sum(hist1)
    # error2 = np.sqrt(hist2) / np.sum(hist2)
    
    error1 = np.sqrt(hist1) 
    error2 = np.sqrt(hist2) 
    
    # Plot histograms with error bars
    axs.bar(bins[:-1], hist1_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[0],edgecolor='black')
    axs.bar(bins[:-1], hist2_norm, width=np.diff(bins), align='center', alpha=0.5, label=labels[1],edgecolor='black')
    axs.errorbar(bins[:-1], hist1_norm, yerr=error1, fmt='none', color='k', capsize=3)
    axs.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
    
    axs.tick_params(axis='both', which='major', labelsize=34)
    leg = axs.legend(fontsize=30,loc='upper right',markerscale=5)
    
    axs.text(0.4,75,'Quiescent Mean: '+str(round(q_mean,2))+' ± '+str(round(q_std,2)),fontsize=30)
    # axs.text(0.4,75,'Quiescent Median: '+str(round(q_median,3)),fontsize=22)
    axs.text(0.4,70,'Non-Quiescent Mean: '+str(round(non_q_mean,1))+' ± '+str(round(non_q_std,1)),fontsize=30)
    # axs.text(0.4,65,'Non-Quiescent Median: '+str(round(non_q_median,3)),fontsize=22)
    
    plt.show()

def norm_cross_heli(enc='all',rlim=35,tau=30,overlap=0.5,tau_unit='min',save=False):
   
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        enc = list(range(1,25))
        
        title_mod = 'for All Encounters'
        
    elif enc == 'no 13':
        enc = list(range(1,13))
        enc.extend(range(14,25))
        title_mod = 'for All Encounters, no 1 or 13'
        
    elif enc == 'no 1':
        enc = list(range(2,25)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        title_mod = 'for All Encounters, no 1'
        
    elif type(enc) is int:
        enc = [enc]
        title_mod = 'for Encounter '+str(enc[0])
        
    elif type(enc) is str and len(enc) == 1:
        enc = int(enc)
        enc = [enc]
        title_mod = 'for Encounter '+str(enc[0])
    
    else:
        title_mod = 'for Encounters '+str(enc[0])+' thru '+str(enc[-1])
    
    #---------------------------Setting time units-----------------------------#
    if tau_unit == 's':
        unit = u.s
    elif tau_unit == 'min':
        unit = u.min
    elif tau_unit == 'hour':
        unit = u.hr
    elif tau_unit == 'day':
        unit = u.day
        
    else:
        print("Invalid tau_unit, using minutes.")
        unit = u.min
        
    tau = tau*unit #cross helicity scale in minutes
    
    #--------------------------------------------------------------------------#
    
    
    
    for i in enc:
        
        t0, tf = utils.encounter_dates(i,rlim=rlim)
        
        print("Encounter ",i)
        print(t0,tf)


        
        # psp.spc(trange=[t0,tf], level='L3',username=sweap_id,password=sweap_pass,last_version=True)
        # vel_data_spc = pys.get_data('psp_spc_vp_fit_RTN')
        
        # if vel_data_spc == None:
        #     vel_data_spc = pys.get_data('spp_spc_vp_fit_RTN')
        # vel_time_arr_spc = vel_data_spc[0]
        # vel_data_arr_spc = vel_data_spc[1]
        
        # spc_vr_clean = utils.sliding_median(vel_data_arr_spc[:,0],275) #about one minute long windows at max cadence
        # spc_vt_clean = utils.sliding_median(vel_data_arr_spc[:,1],275) #about one minute long windows at max cadence
        # spc_vn_clean = utils.sliding_median(vel_data_arr_spc[:,2],275) #about one minute long windows at max cadence
        
        psp.fields(trange=[t0,tf],datatype='sqtn_rfs_V1V2',level='l3',username=fields_id,password=fields_pass,last_version=True)
        
        dens_data = pys.get_data('electron_density')
        dens_time = dens_data[0]
        density = dens_data[1] #number density in 1/cm^3
        
        rho = (const.m_p*density/(u.cm)**3).to(u.kg/u.m**3) #mass density in kg/m^3
        res = (np.median(np.diff(dens_time))*u.s).to(unit)
        
        window_len = int(np.round(tau/res).value)
        
        time_lim_max = np.max(dens_time)
        time_lim_min = np.min(dens_time)
        
        psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)
        vel_data_spi = pys.get_data('psp_spi_VEL_RTN_SUN')
        vel_time_arr_spi = vel_data_spi[0]
        vel_data_arr_spi = vel_data_spi[1]
        
        spi_where = np.where((vel_time_arr_spi>time_lim_min)&(vel_time_arr_spi<time_lim_max))
        spi_where = spi_where[0]
        
        vel_time = vel_time_arr_spi[spi_where]
        vel_data = vel_data_arr_spi[spi_where,:]
        
        vel_nan_where = np.where(~np.isnan(vel_data[:,0]))
        vel_nan_where = vel_nan_where[0]
        
        vel_time_clean = vel_time[vel_nan_where]
        vel_data_clean = vel_data[vel_nan_where,:]
        
        Vr = vel_data_clean[:,0]
        Vt = vel_data_clean[:,1]
        Vn = vel_data_clean[:,2]
        V_mag_spi = np.sqrt(Vr**2+Vt**2+Vn**2)
        
        # Downsample array2 to match time_arr1
        Vr_down = np.interp(dens_time, vel_time_clean, Vr)*u.km/u.s
        Vt_down = np.interp(dens_time, vel_time_clean, Vt)*u.km/u.s
        Vn_down = np.interp(dens_time, vel_time_clean, Vn)*u.km/u.s
        
        psp.fields(trange=[t0,tf],datatype='mag_RTN_4_Sa_per_Cyc',level='l2',username=fields_id,password=fields_pass,last_version=True)
        mag_data = pys.get_data('psp_fld_l2_mag_RTN_4_Sa_per_Cyc')
        mag_time_arr = mag_data[0]
        mag_data_arr = mag_data[1]
        
        mag_where = np.where((mag_time_arr>time_lim_min)&(mag_time_arr<time_lim_max))
        mag_where = mag_where[0]
        
        mag_time = mag_time_arr[mag_where]
        mag_data = mag_data_arr[mag_where,:]
        
        mag_nan_where = np.where(~np.isnan(mag_data[:,0]))
        mag_nan_where = mag_nan_where[0]
        
        mag_time_clean = mag_time[mag_nan_where]
        mag_data_clean = mag_data[mag_nan_where,:]
        
        Br = mag_data_clean[:,0]
        Bt = mag_data_clean[:,1]
        Bn = mag_data_clean[:,2]
        B = np.sqrt(Br**2+Bt**2+Bn**2)
        
        # Downsample array2 to match time_arr1
        Br_down = np.interp(dens_time, mag_time_clean, Br)*u.nT
        Bt_down = np.interp(dens_time, mag_time_clean, Bt)*u.nT
        Bn_down = np.interp(dens_time, mag_time_clean, Bn)*u.nT
        
        Br_Va = (Br_down.to(u.T)/np.sqrt(const.mu0*rho)).to(u.km/u.s) #magnetic field strength in Alfven units
        Bt_Va = (Bt_down.to(u.T)/np.sqrt(const.mu0*rho)).to(u.km/u.s)
        Bn_Va = (Bn_down.to(u.T)/np.sqrt(const.mu0*rho)).to(u.km/u.s)
        B_Va = np.sqrt(Br_Va**2+Bt_Va**2+Bn_Va**2)
        
        pys.del_data()
        
        Br_Va_av, Br_ind = utils.sliding_average(Br_Va.value,window_len,overlap)
        Bt_Va_av, Bt_ind = utils.sliding_average(Bt_Va.value,window_len,overlap)
        Bn_Va_av, Bn_ind = utils.sliding_average(Bn_Va.value,window_len,overlap)
        
        delBr = np.array([])
        delBt = np.array([])
        delBn = np.array([])
        for ind in range(len(Br_ind)-1):
            start = Br_ind[ind]
            stop = Br_ind[ind+1]
            
            delBr_i = Br_Va[start:stop].value-Br_Va_av[ind]
            delBr = np.append(delBr,delBr_i)       
            
            delBt_i = Bt_Va[start:stop].value-Bt_Va_av[ind]
            delBt = np.append(delBt,delBt_i) 
            
            delBn_i = Bn_Va[start:stop].value-Bn_Va_av[ind]
            delBn = np.append(delBn,delBn_i) 
            
            if ind == len(Br_ind)-2:
                delBr_i = Br_Va[stop:].value-Br_Va_av[ind+1]
                delBr = np.append(delBr,delBr_i)
                
                delBt_i = Bt_Va[stop:].value-Bt_Va_av[ind+1]
                delBt = np.append(delBt,delBt_i) 
                
                delBn_i = Bn_Va[stop:].value-Bn_Va_av[ind+1]
                delBn = np.append(delBn,delBn_i) 
        
        delB= np.array([delBr,delBt,delBn]).T
        
        
        Vr_av, Vr_ind = utils.sliding_average(Vr_down.value,window_len,overlap)
        Vt_av, Vt_ind = utils.sliding_average(Vt_down.value,window_len,overlap)
        Vn_av, Vn_ind = utils.sliding_average(Vn_down.value,window_len,overlap)
        
        delVr = np.array([])
        delVt = np.array([])
        delVn = np.array([])
        for ind in range(len(Vr_ind)-1):
            start = Vr_ind[ind]
            stop = Vr_ind[ind+1]
            
            delVr_i = Vr_down[start:stop].value-Vr_av[ind]
            delVr = np.append(delVr,delVr_i)       
            
            delVt_i = Vt_down[start:stop].value-Vt_av[ind]
            delVt = np.append(delVt,delVt_i) 
            
            delVn_i = Vn_down[start:stop].value-Vn_av[ind]
            delVn = np.append(delVn,delVn_i) 
            
            if ind == len(Vr_ind)-2:
                delVr_i = Vr_down[stop:].value-Vr_av[ind+1]
                delVr = np.append(delVr,delVr_i)
                
                delVt_i = Vt_down[stop:].value-Vt_av[ind+1]
                delVt = np.append(delVt,delVt_i) 
                
                delVn_i = Vn_down[stop:].value-Vn_av[ind+1]
                delVn = np.append(delVn,delVn_i) 
        
        
        
        
        # delVr = Vr_down - np.nanmean(Vr_down)
        # delVt = Vt_down - np.nanmean(Vt_down)
        # delVn = Vn_down - np.nanmean(Vn_down)
        
        # delV = np.sqrt(np.nanmean(delVr**2+delVt**2+delVn**2))
        delV= np.array([delVr,delVt,delVn]).T
        
        breakpoint()
        
        delVdelB = np.sum(delV*delB, axis=1)
        delV_2 = np.sum(delV*delV, axis=1)
        delB_2 = np.sum(delB*delB, axis=1)
        
        cross_heli = 2*delVdelB/(delV_2+delB_2) #WTF???
        
        fig = plt.figure(figsize=(30,15))

        axs = fig.add_subplot(111)
        
        axs.set_title("Continuous Normalized Cross Helicity",fontsize=38)
        axs.set_ylabel('$\sigma_{c}$',fontsize=38)
        
        axs.plot(cross_heli)
        
        plt.show()