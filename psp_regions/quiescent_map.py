#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 20 16:28:14 2022

@author: besh2109
"""

from .config import enc_flt

import numpy as np
import os
import sys

import pandas as pd
import csv

import pytplot as pyt
import pyspedas.psp as psp
import pyspedas as pys

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
from matplotlib.markers import MarkerStyle
import matplotlib.dates as mdates

import _pickle as cpkl

from scipy.interpolate import interp1d

import astropy.units as u
import astropy.constants as const
from astropy.io import fits
from astropy.time import Time
from astropy.coordinates import SkyCoord

import sunpy.map
from sunpy.net import Fido, attrs as a
import sunpy.data.sample
from sunpy.map.header_helper import make_heliographic_header
from sunpy.coordinates import get_body_heliographic_stonyhurst, get_horizons_coord
from sunpy.time import parse_time
from sunpy.coordinates import frames

from reproject import reproject_interp, reproject_and_coadd
from sunkit_magex import pfss

#---solve parkers solution----#

import parkersolarwind as psw

#-----------------------------#

import glob

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

import utils

fields_id = os.environ['PSP_FIELDS_ID']
fields_pass = os.environ['PSP_FIELDS_PW']

sweap_id = os.environ['PSP_SWEAP_ID']
sweap_pass = os.environ['PSP_SWEAP_PW']

jsoc_email = os.environ['JSOC_EMAIL']

class MarkerSizeHandler(HandlerBase):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        marker_style = orig_handle.get_marker()
        marker_size = 4  # Define the desired marker size

        x = width // 2 - marker_size / 2
        y = height // 2 - marker_size / 2

        marker = MarkerStyle(marker_style)
        marker._transform = marker.get_transform().scale(marker_size)

        return [plt.Line2D([x], [y], marker=marker)]


def quiescent_map(t0='2020-01-29',tf=None,enc=None,rss=2.5,r_tmp=None,plot=False,save_coords=False,rlim=55,
                  source='hmi',adapt_source='gong',peri=True,low_res=True,full_run=True,test_plot=True,V_err=0.04,V_err_check=True,magneto_err=0):
    
    save_file_tag = '_'+source+'_rss_'+str(rss)[0]
    
    if len(str(rss))>1: #if rss is a decimal
        save_file_tag = save_file_tag +'_'+str(rss)[2]
    else: #if rss is an integer
        save_file_tag = save_file_tag +'_0'
        
    if magneto_err != 0:
        if np.sign(magneto_err) > 0:
            save_file_tag = save_file_tag + '_plus_'+str(magneto_err)+'_day'
        if np.sign(magneto_err) < 0:
            save_file_tag = save_file_tag + '_minus_'+str(magneto_err)+'_day'
            
        V_err_check = False
    
    Rs_km = 6.957e5*u.km #solar radius in km  
    
    JtoeV = 6.242*(10**18) # one joule equals 6.242*10^18 eV
    
    mp = 1.67262192*(10**(-27)) # mass of proton in kg
    
    if enc is not None:
        t0, tf = utils.encounter_dates(enc,rlim)

    if r_tmp==None:
        r_tmp=rss
    
    if tf==None:
        tf = pys.time_string(pys.time_float(t0)+86400)
        
    def set_axes_lims(ax):
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 180)
        
    if enc == None:
        enc = utils.encounter_check(t0)

    #-------------------------------IMPORT DATA-------------------------------#

    #------position data importing------#

    psp.fields(trange=[t0,tf], datatype='ephem_spp_hg', level='l1',username=fields_id,password=fields_pass,last_version=True) #going to be used to plot parker position
    pos_data = pyt.get_data('position')
    
    pos_time_arr = pos_data[0]
    pos_data_arr = pos_data[1]
    
    x = pos_data_arr[:,0]*u.km
    y = pos_data_arr[:,1]*u.km
    z = pos_data_arr[:,2]*u.km
    
    #----SPC velocity data importing-----#  
    
    psp.spc(trange=[t0,tf],level='L3',username=sweap_id,password=sweap_pass,last_version=True)
        
    spc_data = pyt.get_data('psp_spc_vp_fit_RTN')
    
    if spc_data == None:
        spc_data = pyt.get_data('spp_spc_vp_fit_RTN')

    spc_time_arr = spc_data[0]
    spc_data_arr = spc_data[1]

    vr_spc = spc_data_arr[:,0]
    
    vr_spc_clean = utils.sliding_median(vr_spc,275)

    interpolating_func = interp1d(spc_time_arr, vr_spc_clean, kind='linear', fill_value='extrapolate')
    
    # Create a new array of time values with fixed cadence
    spc_time_down = np.arange(spc_time_arr[0], spc_time_arr[-1], 3)

    # Interpolate arr_two to the new time values
    spc_v_down = interpolating_func(spc_time_down)
    
    spc_time_arr = spc_time_down
    vr_spc = spc_v_down
    
    spc_temp_data = pyt.get_data('psp_spc_wp1_fit')
    
    spc_temp_time_arr = spc_temp_data[0]
    spc_temp_data_arr = spc_temp_data[1]
    
    temp_spc_ev = ((1/2)*JtoeV*mp*spc_temp_data_arr**2)*(10**6) # should return the temperature of the protons in eV
    
    tp_spc_clean = utils.sliding_median(temp_spc_ev,275)
    
    interpolating_func = interp1d(spc_temp_time_arr, tp_spc_clean, kind='linear', fill_value='extrapolate')
    
    # Create a new array of time values with fixed cadence
    spc_time_down = np.arange(spc_temp_time_arr[0], spc_temp_time_arr[-1], 3)

    # Interpolate arr_two to the new time values
    spc_tp_down = interpolating_func(spc_time_down)
    
    tp_spc = spc_tp_down
    
    #------SPI velocity data importing------# 
    
    psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)

    vel_data = pyt.get_data('psp_spi_VEL_RTN_SUN')
    
    spi_time_arr = vel_data[0]
    spi_data_arr = vel_data[1]
    
    vr_spi = spi_data_arr[:,0]
    
    temp_data = pyt.get_data('psp_spi_TEMP')
    
    spi_temp_data_arr = temp_data[1]
    
    tp_spi = spi_temp_data_arr

    #------mag data from SPI product, really from FIELDS-------#
    
    psp.fields(trange=[t0,tf],level='l2',datatype='mag_RTN_1min',username=fields_id,password=fields_id,last_version=True)
    
    mag_data = pyt.get_data('psp_fld_l2_mag_RTN_1min')
    
    mag_time_arr = mag_data[0]*u.s
    mag_data_arr = mag_data[1]
    
    br_fields = mag_data_arr[:,0]
    bt_fields = mag_data_arr[:,1]
    bn_fields = mag_data_arr[:,2]
    
    sliding_med_br = utils.sliding_median(br_fields,120) #30 minute sliding window. 120 indices = 120 minutes.

    #---------------handling velocity data for Parker Spiral------------------#

    r0_time = pos_time_arr*u.s

    r0 = np.sqrt(x**2+y**2+z**2) #parker solar probe position height in kilometers
    r0_Rs = np.sqrt(x**2+y**2+z**2)/Rs_km*u.R_sun #height from center of sun in Solar Radii
    
    pos_len = len(pos_time_arr)
    
    hyb_time_vr, hyb_vr = utils.SPC_SPI_Construct(enc,spi_time_arr,vr_spi,spc_time_down,vr_spc) #hybrid (hyb) velocity arrays
    
    hyb_time_vr = np.array(hyb_time_vr)*u.s
    
    Vsw = hyb_vr*u.km/u.s
    Vsw_Rs = ((hyb_vr*u.km/u.s)/Rs_km)*u.R_sun

    Vsw_Rs_err_max = Vsw_Rs * (1+V_err)
    Vsw_Rs_err_min = Vsw_Rs * (1-V_err)
    
    #---------------handling temperature data------------------#
    
    pos_len = len(pos_time_arr)
    
    hyb_time_Tp, hyb_Tp = utils.SPC_SPI_Construct(enc,spi_time_arr,tp_spi,spc_time_arr,tp_spc) #hybrid (hyb) temperature arrays
    
    hyb_time_Tp = np.array(hyb_time_Tp)*u.s
    
    Tp = hyb_Tp*u.eV

    #---interpolate the position data up to velocity data rates---#
    r0_Rs_i = np.interp(hyb_time_vr,r0_time,r0_Rs)
    r0_i = np.interp(hyb_time_vr,r0_time,r0)
    x_i = np.interp(hyb_time_vr,r0_time,x)
    y_i = np.interp(hyb_time_vr,r0_time,y)
    z_i = np.interp(hyb_time_vr,r0_time,z)
    
    Tp_i = np.interp(hyb_time_vr,hyb_time_Tp,hyb_Tp)
    
    r0_time_arr = np.array(hyb_time_vr)
    r0 = np.array(r0_i)
    r0_Rs = np.array(r0_Rs_i)
    x_pos = np.array(x_i)
    y_pos = np.array(y_i)
    z_pos = np.array(z_i)

    Tp = np.array(Tp_i)
    
    pos_len = len(r0)
    # breakpoint()
    #-----------------------scale down the pfss model--------------------#
    
    if low_res:
        
        # Original indices (or times)
        # original_indices = np.linspace(r0_time_arr[0], r0_time_arr[-1], num=pos_len)
        # New indices for downsampling
        # low_res_times = np.linspace(r0_time_arr[0], r0_time_arr[-1], num=15000)
        low_res_times = np.linspace(r0_time_arr[0], r0_time_arr[-1], num=5000)
        
        
        r0_Rs_i = np.interp(low_res_times,r0_time_arr,r0_Rs)
        r0_i = np.interp(low_res_times,r0_time_arr,r0)
        
        x_i = np.interp(low_res_times,r0_time_arr,x_pos)
        y_i = np.interp(low_res_times,r0_time_arr,y_pos)
        z_i = np.interp(low_res_times,r0_time_arr,z_pos)
        
        Vsw_i = np.interp(low_res_times,r0_time_arr,Vsw)
        Vsw_Rs_i = np.interp(low_res_times,r0_time_arr,Vsw_Rs)
        Vsw_Rs_max_i = np.interp(low_res_times,r0_time_arr,Vsw_Rs_err_max)
        Vsw_Rs_min_i = np.interp(low_res_times,r0_time_arr,Vsw_Rs_err_min)
        
        
        Tp_i = np.interp(low_res_times,r0_time_arr,Tp)
        
        br_i = np.interp(low_res_times,mag_time_arr,br_fields)
        bt_i = np.interp(low_res_times,mag_time_arr,bt_fields)
        bn_i = np.interp(low_res_times,mag_time_arr,bn_fields)
        
        br_med_i = np.interp(low_res_times,mag_time_arr,sliding_med_br)
        
        r0_time_arr = low_res_times
        r0 = r0_i
        r0_Rs = r0_Rs_i
        x_pos = x_i
        y_pos = y_i
        z_pos = z_i
        Vsw = Vsw_i
        Vsw_Rs = Vsw_Rs_i
        Vsw_Rs_err_max = Vsw_Rs_max_i
        Vsw_Rs_err_min = Vsw_Rs_min_i
        
        br_fields = br_i
        bt_fields = bt_i
        bn_fields = bn_i
        
        sliding_med_br = br_med_i
        
        pos_len = len(r0)
    
    # fields_polarity_no_sbs = sliding_med_br/np.abs(sliding_med_br)
    # fields_polarity = br_fields/np.abs(br_fields) #just ±1 for positive/negative Br
    
    #------------------------Solve for Parker Spiral-------------------#

    # calculate time solar wind measured by PSP launched from the source surface.
    time_of_flight = np.abs(rss*u.R_sun-r0_Rs)/Vsw_Rs #should be in seconds. Time traveled from Rss
    
    corrected_time = r0_time_arr-time_of_flight #epoch time which the solar wind launched, assuming it was flying at the same speed the whole time. (ASSUMPTION)
    series = pd.Series(corrected_time)
    corrected_time = series.interpolate() #interpolates over the nans.

    carr_lon_psp = np.arctan2(y_pos, x_pos).to(u.deg)

    carr_lon_psp = np.where(carr_lon_psp < 0 * u.deg, carr_lon_psp + 360 * u.deg, carr_lon_psp)

    carr_lat_psp = np.arcsin(z_pos / r0).to(u.deg)

    w = 360/(25.38*86400)*u.deg/u.s # angular frequency of the sun in degrees/sec
    # w = 2*np.pi/(25.38*86400)*u.radian/u.s # angular frequency of the sun in radians/sec

    sintheta = np.sin((90*u.deg-carr_lat_psp).to(u.radian)) #sin of the azimuthal angle, which is 90 degrees minus the latitude

    src_lon = carr_lon_psp - (w*sintheta/Vsw_Rs)*(rss*u.R_sun-r0_Rs) #carrington longitude of the PSP traced by a Parker Spiral on the source surface
    
    err_max_lon = carr_lon_psp - (w*sintheta/Vsw_Rs_err_max)*(rss*u.R_sun-r0_Rs) #carrington longitude of the PSP traced by a Parker Spiral on the source surface
    err_min_lon = carr_lon_psp - (w*sintheta/Vsw_Rs_err_min)*(rss*u.R_sun-r0_Rs)
    
    t0_corr = pys.time_string(corrected_time[0].value)
    tf_corr = pys.time_string(corrected_time[-1].value)

    #----------------------------find date of perihelion----------------------#
    
    per_where = np.where(r0_Rs == np.min(r0_Rs))
    per_where = per_where[0]
    
    peri_time = r0_time_arr[per_where]
    
    peri_date_t0 = pys.time_string(peri_time[0].value-43200) #select the date for perihelion
    peri_date_tf = pys.time_string(peri_time[0].value+43200) 
    
    corr_peri_time = corrected_time[per_where]
    corr_peri_date = pys.time_string(corr_peri_time[0].value)
    
    corr_peri_date_t0 = pys.time_string(corr_peri_time[0].value-43200) #select the date when the plasma observed at perihelion left the sun
    corr_peri_date_tf = pys.time_string(corr_peri_time[0].value+43200)
    
    if magneto_err !=0:
        corr_peri_date_t0 = pys.time_string(corr_peri_time[0].value-43200+magneto_err*86400) #select the date when the plasma observed at perihelion left the sun
        corr_peri_date_tf = pys.time_string(corr_peri_time[0].value+43200+magneto_err*86400)
    
    #--------------------------PFSS MODEL START-----------------------------#
    
    if source=='hmi':
        
        if peri:
            t_start_hmi = parse_time(corr_peri_date_t0)
            t_end_hmi = parse_time(corr_peri_date_tf)
        # pfss_out = hmi2pfss(dt=t0_del_dt)
        else:
            t_start_hmi = parse_time(t0_corr)
            t_end_hmi = parse_time(tf_corr)

        results_hmi = Fido.search(a.jsoc.Notify(os.environ["JSOC_EMAIL"]),a.jsoc.Time(t_start_hmi.value,t_end_hmi.value),a.jsoc.Series('hmi.mrdailysynframe_small_720s'))
        filenames = Fido.fetch(results_hmi, path=os.environ['SUNPY_DATA_DIR']+'/hmi')
        
    if source=='gong':
        
        if peri:
            t0 = peri_date_t0
            tf = peri_date_tf
        # pfss_out = hmi2pfss(dt=t0_del_dt)
        else:
            pass
        
        filenames = pys.adapt.gong(trange=[t0,tf])
        
        # pfss_out = gong2pfss(dt=t0_del_dt)
    if source=='adapt':
        
        if peri:
            t0 = peri_date_t0
            tf = peri_date_tf
        # pfss_out = hmi2pfss(dt=t0_del_dt)
        else:
            pass
        
        filenames = pys.adapt.adapt(trange=[t0,tf],adapt_source=adapt_source)
    
    first_file = filenames[0]
    
    ###############################################################################
    # The PFSS solution is calculated on a regular 3D grid in (phi, s, rho), where
    # rho = ln(r), and r is the standard spherical radial coordinate. We need to
    # define the number of rho grid points, and the source surface radius.
    nrho = 35
    rss = rss
    
    if peri:
        
        hdul = fits.open(first_file)
        
        if source=='gong':
            dates = hdul[0].header["DATE-OBS"]+'/'+hdul[0].header["TIME-OBS"]
            
            pfss_map = sunpy.map.Map(first_file)
            pfss_map.meta['rsun'] = sunpy.sun.constants.radius.value
            pfss_in = pfss.Input(pfss_map, nrho, rss)
            pfss_out = pfss.pfss(pfss_in)
            
        if source=='adapt':
            dates = hdul[0].header["MAPTIME"][:10]+'/'+hdul[0].header["MAPTIME"][11:]
            
            pfss_map = pfss.utils.load_adapt(first_file)
            pfss_map_0 = pfss_map[0]

            pfss_map_cea = pfss.utils.car_to_cea(pfss_map_0)
               
            pfss_in = pfss.Input(pfss_map_cea, nrho, rss)
            pfss_out = pfss.pfss(pfss_in)
            
        if source=='hmi':
            
            # date_str = hdul[0].header['T_OBS']

            # # Parse the original date string
            # dt = datetime.strptime(date_str, "%Y.%m.%d_%H:%M:%S.%f_TAI")

            # # Format the datetime object to the desired format
            # formatted_date = dt.strftime("%Y-%m-%d/%H:%M:%S.%f")
            
            # hdul[0].header['DATE-OBS'] = formatted_date
            
            hmi_fits = fits.open(first_file)
            hmi_header = utils.fix_hmi_meta(hmi_fits.header)
            
            pfss_map = sunpy.map.Map(hmi_fits.data,hmi_header)
            
            pfss_in = pfss.Input(pfss_map, nrho, rss)
            pfss_out = pfss.pfss(pfss_in)
            # breakpoint()

        tracer = pfss.tracing.FortranTracer()
        r = r_tmp * const.R_sun
        
        lat_tmp = carr_lat_psp
        
        lon_tmp = src_lon
        
        lat_tmp, lon_tmp = lat_tmp.ravel() * u.deg, lon_tmp.ravel() * u.deg
        
        seeds = SkyCoord(lon_tmp, lat_tmp, r, frame=pfss_out.coordinate_frame)

        field_lines_tmp = tracer.trace(seeds, pfss_out)
        
        expans_fact = np.array(field_lines_tmp.expansion_factors) #expansion factors
        
        
        tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/footpoints/'
        
        if enc != None:    
            tplot_savename = 'Enc_'+str(enc)+'_footpoint_coords.cdf'
        else:
            t0_name = t0[:10]
            tf_name = tf[:10]
            tplot_savename = t0_name+'_'+tf_name+'_footpoint_coords.cdf'
        
        progress = np.linspace(0,pos_len,21)
        i=0
        ftprnt_lon = []
        ftprnt_lat = []
        polar_tmp = []
        exp_tmp = []
        i_non_breaks = []
        # breakpoint()
        
        
        for field_line in field_lines_tmp:
            # cpkl.dump(field_line, file)
            if i in progress:
                print(int(100*i/pos_len),'%')
            
            coord_check = field_line.coords
            if coord_check.shape != (0,):
                # print('old code')
                coords = field_line.solar_footpoint
                pol = field_line.polarity
                ftprnt_lon.append(float(coords.lon/u.deg))
                ftprnt_lat.append(float(coords.lat/u.deg))
                polar_tmp.append(pol)
                i_non_breaks.append(i)

            i+=1
        
        
    else:
    
        fits_dates = []
        fits_dts = []
        
        #retrieve dates for each file
        
        for file in filenames:
            with fits.open(file) as hdul:
                # Access the primary HDU
                # breakpoint()
                if source=='gong':
                    dates = hdul[0].header["DATE-OBS"]+'/'+hdul[0].header["TIME-OBS"]
                    
                if source=='adapt':
                    dates = hdul[0].header["MAPTIME"][:10]+'/'+hdul[0].header["MAPTIME"][11:]
                    
                if source=='hmi':
                    # breakpoint()
                    
                    date_str = hdul[0].header['T_OBS']
    
                    # Parse the original date string
                    dt = datetime.strptime(date_str, "%Y.%m.%d_%H:%M:%S.%f_TAI")
    
                    # Format the datetime object to the desired format
                    formatted_date = dt.strftime("%Y-%m-%d/%H:%M:%S.%f")
                    
                    hdul[0].header['DATE-OBS'] = formatted_date
                    
                    dates = hdul[0].header["DATE-OBS"]
                
                fits_dates.append(dates)
                fits_dts.append(datetime.fromisoformat(dates))
             
        #------------- select a magnetogram every three days going forward and backward from perihelion --------------# 
        
        fits_dates = np.array(fits_dates)
        fits_dts = np.array(fits_dts)
        

        peri_dt = datetime.fromisoformat(corr_peri_date)
        
        dif_arr = np.abs(fits_dts-peri_dt)
        
        ref_ind = np.where(dif_arr == min(dif_arr))
        ref_ind = ref_ind[0][0] #reference index of perihelion magnetogram
        
        closest_peri_dt = fits_dts[ref_ind]
        
        day_num = 3
        # result,day_indices = find_every_i_days(date_array,ref_ind,day_num)

        # Input reference date (e.g., an element near the middle of the array, including time)
        reference_date = closest_peri_dt
        
        # Find the index of the reference date in fits_dts
        reference_idx = np.argmin(np.abs(fits_dts - reference_date))
        
        # Time delta for 3 days
        delta = timedelta(days=day_num)
        
        # Initialize the result list and the set of visited indices
        result_dates = [fits_dts[reference_idx]]
        visited_indices = set([reference_idx])
        
        # Function to find the closest datetime object in a given direction (forward or backward)
        def closest_date(target_date, direction="forward"):
            if direction == "forward":
                candidates = fits_dts[reference_idx + 1:]  # forward direction
            else:
                candidates = fits_dts[:reference_idx]  # backward direction
            
            closest_date = None
            min_diff = timedelta.max
        
            # Search for the closest match
            for date in candidates:
                diff = abs(date - target_date)
                if diff < min_diff:
                    min_diff = diff
                    closest_date = date
            
            return closest_date
        
        # Iterate forward and backward, looking for closest dates
        for i in range(1,6):  # We can limit how many steps forward/backward we want
            # Search forward direction (e.g., +3 days)
            next_date = closest_date(reference_date + i*delta, "forward")
            if next_date is not None:
                result_dates.append(next_date)
                # reference_date = next_date
                visited_indices.add(np.argmin(np.abs(fits_dts - reference_date)))
            
            # Search backward direction (e.g., -3 days)
            prev_date = closest_date(reference_date - i*delta, "backward")
            if prev_date is not None:
                result_dates.insert(0, prev_date)
                # reference_date = prev_date
                visited_indices.add(np.argmin(np.abs(fits_dts - reference_date)))
            
            # print(i)

        result_dates = np.unique(result_dates)
        
        final_date_set = set(result_dates)

        # Find indices using a list comprehension with set lookup
        indices = [i for i, date in enumerate(fits_dts) if date in final_date_set]
        
        filenames = [filenames[index] for index in indices]
        fits_dates = [fits_dates[index] for index in indices]
        fits_dts = [fits_dts[index] for index in indices]
        
        #----------------------- check which datapoints belong to which magnetogram ------------------------#
        # breakpoint()
        
        fits_floats = pys.time_float(fits_dates)
        
        indices = utils.group_elements(corrected_time.value,fits_floats)
        
        unique_indices, ind_count = np.unique(indices,return_counts=True)
        
        tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/footpoints/'
        # fieldline_savepath = "/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/pfss_files/"
        
        if enc != None:    
            tplot_savename = 'Enc_'+str(enc)+'_footpoint_coords'+save_file_tag+'.cdf'
            # fieldline_savename = 'Enc_'+str(enc)+'_field_lines.pkl'
        else:
            t0_name = t0[:10]
            tf_name = tf[:10]
            
            tplot_savename = t0_name+'_'+tf_name+'_footpoint_coords'+save_file_tag+'.cdf'
            # fieldline_savename = t0_name+'_'+tf_name+'_field_lines.pkl'
        
        progress = np.linspace(0,pos_len,21)
        i=0
        
        ftpnt_lon = []
        ftpnt_lon_err_max = []
        ftpnt_lon_err_min = []
        ftpnt_lat = []
        ftpnt_lat_err_max = []
        ftpnt_lat_err_min = []
        
        polar_tmp = []
        exp_tmp = []
        Br_tmp = np.array([])
        i_non_breaks = []
        # field_lines = []
        # with open(fieldline_savepath+fieldline_savename,'wb') as file:
            
        for ind in unique_indices:
            
            long_where = np.where(indices==ind)
            
            if source=='gong':
                
                pfss_map = sunpy.map.Map(filenames[ind])
                pfss_map.meta['rsun'] = sunpy.sun.constants.radius.value
                pfss_in = pfss.Input(pfss_map, nrho, rss)
                pfss_out = pfss.pfss(pfss_in)
            
            if source=='adapt':

                pfss_map = pfss.utils.load_adapt(filenames[ind])
                pfss_map_0 = pfss_map[0]

                pfss_map_cea = pfss.utils.car_to_cea(pfss_map_0)

                pfss_in = pfss.Input(pfss_map_cea, nrho, rss)
                pfss_out = pfss.pfss(pfss_in)
                   
            if source=='hmi':

                hmi_data, hmi_header = fits.getdata(filenames[ind], header=True)
                
                hmi_data[np.isnan(hmi_data)]=np.nanmean(hmi_data)
                
                hmi_header_new = utils.fix_hmi_meta(hmi_header)
                pfss_map = sunpy.map.Map(hmi_data,hmi_header_new)

                pfss_map.meta['rsun'] = sunpy.sun.constants.radius.value
                pfss_in = pfss.Input(pfss_map, nrho, rss)
                pfss_out = pfss.pfss(pfss_in)

    
            tracer = pfss.tracing.FortranTracer()
            r = r_tmp * const.R_sun
            
            lat_tmp = np.array(carr_lat_psp[long_where])
            lon_tmp = src_lon[long_where]
            r0_Rs_tmp = r0_Rs[long_where]

            lat_tmp, lon_tmp = lat_tmp.ravel(), lon_tmp.ravel()
            seeds = SkyCoord(lon_tmp, lat_tmp, r, frame=pfss_out.coordinate_frame)
            field_lines_tmp = tracer.trace(seeds, pfss_out)
            
            if V_err_check:
            
                lat_tmp = np.array(carr_lat_psp[long_where])
                max_err_lon_tmp = err_max_lon[long_where]
                max_err_lat_tmp, max_err_lon_tmp = lat_tmp.ravel(), max_err_lon_tmp.ravel()
                max_err_seeds = SkyCoord(max_err_lon_tmp, max_err_lat_tmp, r, frame=pfss_out.coordinate_frame)
                
                lat_tmp = np.array(carr_lat_psp[long_where])
                min_err_lon_tmp = err_min_lon[long_where]
                min_err_lat_tmp, min_err_lon_tmp = lat_tmp.ravel(), min_err_lon_tmp.ravel()
                min_err_seeds = SkyCoord(min_err_lat_tmp, min_err_lon_tmp, r, frame=pfss_out.coordinate_frame)

                max_err_field_lines_tmp = tracer.trace(max_err_seeds, pfss_out)
                min_err_field_lines_tmp = tracer.trace(min_err_seeds, pfss_out)
            
            br_guess = utils.PFSS_Br_estimation(pfss_out, seeds, r0_Rs_tmp, rss, A_scale=5)
            
            # Br_tmp.append(br_guess)
            Br_tmp = np.append(Br_tmp,br_guess.value)
            
            """# cpkl.dump(field_lines_tmp, file) No longer outputing the field_line objects to a pickle file. Saving storage space.""" 
            
            for j in range(len(field_lines_tmp)):
                field_line = field_lines_tmp[j]
                
                if V_err_check:
                    field_line_max = max_err_field_lines_tmp[j]
                    field_line_min = min_err_field_lines_tmp[j]
                # cpkl.dump(field_line, file)
                if i in progress:
                    print(int(100*i/pos_len),'%')
                
                coord_check = field_line.coords
                
                if coord_check.shape != (0,):
                    coords = field_line.solar_footpoint
                    pol = field_line.polarity
                    exp = field_line.expansion_factor
                    ftpnt_lon.append(float(coords.lon/u.deg))
                    ftpnt_lat.append(float(coords.lat/u.deg))
                    if V_err_check:
                        err_max_coords = field_line_max.solar_footpoint
                        err_min_coords = field_line_min.solar_footpoint
                        ftpnt_lon_err_max.append(float(err_max_coords.lon/u.deg))
                        ftpnt_lon_err_min.append(float(err_min_coords.lon/u.deg))
                        ftpnt_lat_err_max.append(float(err_max_coords.lat/u.deg))
                        ftpnt_lat_err_min.append(float(err_min_coords.lat/u.deg))
                    polar_tmp.append(pol)
                    exp_tmp.append(exp)
                    
                    i_non_breaks.append(i)
    
                i+=1

    #------------------SAVING THE FOOTPOINT COORDINATES-------------------#
    if save_coords:
        
        sol_long = np.array(ftprnt_lon) #footpoint longitudes

        sol_lat = np.array(ftprnt_lat) #footpoint latitudes

        dates = np.array(pys.time_string(r0_time_arr[i_non_breaks].value))
        date_flts = np.array(r0_time_arr[i_non_breaks].value)
        PSP_lon = np.array(carr_lon_psp[i_non_breaks].value)
        PSP_lat = np.array(carr_lat_psp[i_non_breaks].value)
        PSP_Rs = np.array(r0_Rs[i_non_breaks].value)
        Vsw_Rs_new = np.array(Vsw_Rs[i_non_breaks].value)
        Tp_new = np.array(Tp[i_non_breaks].value)
        polarity = np.array(polar_tmp)
        # Br_PFSS = np.array(Br_tmp[i_non_breaks])
        
        if V_err_check:
            sol_lon_err_max = np.array(ftpnt_lon_err_max)
            sol_lon_err_min = np.array(ftpnt_lon_err_min)
            sol_lat_err_max = np.array(ftpnt_lat_err_max)
            sol_lat_err_min = np.array(ftpnt_lat_err_min)
        else:
            sol_lon_err_max = np.ones(sol_long.shape)*np.nan
            sol_lon_err_min = np.ones(sol_long.shape)*np.nan
            sol_lat_err_max = np.ones(sol_long.shape)*np.nan
            sol_lat_err_min = np.ones(sol_long.shape)*np.nan
        
        
        if len(exp_tmp) == 0:
            expansion = np.array(expans_fact[i_non_breaks])
        else:
            expansion = np.array(exp_tmp)
        c_time = np.array(corrected_time[i_non_breaks])
    
        tplot_time = date_flts
        
        tplot_sol_lon = sol_long
        tplot_sol_lon_err_max =  sol_lon_err_max
        tplot_sol_lon_err_min = sol_lon_err_min
        tplot_sol_lat = sol_lat
        tplot_sol_lat_err_max = sol_lat_err_max
        tplot_sol_lat_err_min = sol_lat_err_min
        tplot_carr_lon = PSP_lon
        tplot_carr_lat = PSP_lat
        tplot_r0_Rs = PSP_Rs
        tplot_rss = np.array([rss])
        tplot_polarity = polarity
        tplot_expansion = expansion
        tplot_c_time = c_time
        tplot_Vsw_Rs = Vsw_Rs_new
        tplot_Tp = Tp_new

        pyt.store_data("solar_lon", data={'x':tplot_time, 'y':tplot_sol_lon})
        pyt.store_data("solar_lon_err_max", data={'x':tplot_time, 'y':tplot_sol_lon_err_max})
        pyt.store_data("solar_lon_err_min", data={'x':tplot_time, 'y':tplot_sol_lon_err_min})
        pyt.store_data("solar_lat", data={'x':tplot_time, 'y':tplot_sol_lat})
        pyt.store_data("solar_lat_err_max", data={'x':tplot_time, 'y':tplot_sol_lat_err_max})
        pyt.store_data("solar_lat_err_min", data={'x':tplot_time, 'y':tplot_sol_lat_err_min})
        pyt.store_data("PSP_lon", data={'x':tplot_time, 'y':tplot_carr_lon})
        pyt.store_data("PSP_lat", data={'x':tplot_time, 'y':tplot_carr_lat})
        pyt.store_data("PSP_Rs", data={'x':tplot_time, 'y':tplot_r0_Rs})
        pyt.store_data("rss",data={'x':tplot_rss, 'y':tplot_rss})
        pyt.store_data("Vsw_Rs",data={'x':tplot_time, 'y':tplot_Vsw_Rs})
        pyt.store_data("Temp_p",data={'x':tplot_time,'y':tplot_Tp})  
        
        pyt.store_data("polarity",data={'x':tplot_time, 'y':tplot_polarity})
        pyt.store_data("expansion_factor",data={'x':tplot_time, 'y':tplot_expansion})
        pyt.store_data("corrected_time",data={'x':tplot_time, 'y':tplot_c_time})
    
        cdf_var_list = ["solar_lon","solar_lon_err_max","solar_lon_err_min","solar_lat","solar_lat_err_max","solar_lat_err_min",
                        "PSP_lon","PSP_lat","PSP_Rs","rss","Vsw_Rs","Temp_p","polarity","expansion_factor"]
        
        pyt.tplot_save(cdf_var_list,tplot_savepath+tplot_savename) #saves the PFSS properties to a .cdf file
        
        if full_run:
            sort_footpoints(enc=enc)
            
    if test_plot:
        fig_br = plt.figure(figsize=(17.5,7))
        ax = fig_br.add_subplot(111)
        
        if source == 'hmi':
            Br_plot = -Br_tmp
        if source == 'gong':
            Br_plot = Br_tmp
        
        ax.plot(r0_time_arr[i_non_breaks],Br_plot[i_non_breaks],color='tab:green',label='PFSS estimated Br') #threw in the minus sign to make the model work. Seems to be backwards.
        ax.plot(r0_time_arr[i_non_breaks],sliding_med_br[i_non_breaks],color='tab:blue',label='FIELDS measured Br')
        
        ax.set_title("Radial Magnetic Field PSP vs PFSS, using "+source,fontsize=26)
        ax.set_ylabel("Br @ PSP in nT",fontsize=20)
        
        # Rotate the labels for readability
        plt.xticks(rotation=45,fontsize=15)
        
        plt.grid(True)
        plt.legend()
        plt.show()
    
def sort_footpoints(t0='2020-01-29',tf=None,enc=None,save=True):
    # w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec

    #------------------------------ Set path to tplot and quiescent region files ----------------------------------#

    if tf==None:
        tf = pys.time_string(pys.time_float(t0)+86400)

    if enc != None:    
        tplot_savename = 'Enc_'+str(enc)+'_footpoint_coords.cdf'
        qregion_savename = 'enc_'+str(enc)+'_regions_raw.csv'
    else:
        tplot_savename = t0+'_'+tf+'_footpoint_coords.cdf'
        qregion_savename = t0+'_'+tf+'_regions_raw.csv'
        
        t0float = pys.time_float(t0)
        enc = 1
        for i in enc_flt:
            # if t0float>i[0] and t0float<i[1]:
            #     encounter = enc
            enc+=1
        if tf==None:
            tffloat = pys.time_float(t0)+86400
        else:
            tffloat=pys.time_float(tf)
        
    tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/footpoints/'
    qregion_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'

    #------------------------------ Read in Tplot Variables ----------------------------------#
    
    pyt.tplot_restore(tplot_savepath+tplot_savename)
    
    rss_data = pyt.get_data('rss')
    rss = rss_data[0]
    rss = rss[0]
    
    carr_lon_data = pyt.get_data('PSP_lon')
    carr_lon_time = carr_lon_data[0]
    carr_lon = carr_lon_data[1]
    
    carr_lat_data = pyt.get_data('PSP_lat')
    carr_lat = carr_lat_data[1]
    
    # sintheta = np.sin((90-carr_lat)*np.pi/180) #sin of the azimuthal angle, which is 90 degrees minus the latitude
    
    solar_lon_data = pyt.get_data('solar_lon')
    solar_lon_time = solar_lon_data[0]
    sol_lon = solar_lon_data[1]
    
    sol_lat_data = pyt.get_data('solar_lat')
    sol_lat = sol_lat_data[1]
    
    r0_Rs_data = pyt.get_data('PSP_Rs')
    r0_Rs = r0_Rs_data[1]
    
    exp_fact_data = pyt.get_data('expansion_factor')
    exp_fact = exp_fact_data[1]
    
    #---------------------------- Read in quiescent regions --------------------------------#
    
    # breakpoint()
    
    qregion_df = pd.read_csv(qregion_savepath+qregion_savename)
    qregion_array = qregion_df.to_numpy()
    
    if enc != None:    
        regions_arr = qregion_array
        
    else:
        
        pre_flt = qregion_array[:,:2]
        
        qregion_lst = []
        for j in pre_flt:
            qregion_lst.append(pys.time_float(j))
        qregion_flt = np.array(qregion_lst) 
        reg_where = np.where((qregion_flt[:,0]>t0float)&(qregion_flt[:,1]<tffloat))
        
        regions_arr = qregion_array[reg_where,:]
        
    # breakpoint()
    #---------------- find footpoints associated with quiescent regions --------------------#
    
    region_time = np.array([])
    region_foot_lon = np.array([])
    region_foot_lat = np.array([])
    region_exp_fact = np.array([])
    
    carr_lat_max = np.array([])
    carr_lat_min = np.array([])
    carr_lon_max = np.array([])
    carr_lon_min = np.array([])
    
    q_index_arr = np.array([])
    
    for k in regions_arr:

        pre_flt = k[:2]
        reg_flt = pys.time_float(pre_flt)
        point_where = np.where((carr_lon_time>reg_flt[0])&(carr_lon_time<reg_flt[1]))
        point_where = point_where[0]
        
        q_index_arr = np.append(q_index_arr,point_where)
        
        region_time_tmp = solar_lon_time[point_where]
        region_foot_lon_tmp =  sol_lon[point_where]
        region_foot_lat_tmp = sol_lat[point_where]
        region_exp_tmp = exp_fact[point_where]

        region_time = np.append(region_time,region_time_tmp)
        region_foot_lon = np.append(region_foot_lon,region_foot_lon_tmp)
        region_foot_lat = np.append(region_foot_lat,region_foot_lat_tmp)
        region_exp_fact = np.append(region_exp_fact,region_exp_tmp)
        
        if len(point_where) != 0:
        
            carr_lat_max = np.append(carr_lat_max,np.max(region_foot_lat_tmp))
            carr_lat_min = np.append(carr_lat_min,np.min(region_foot_lat_tmp))
            
            carr_lon_max = np.append(carr_lon_max,np.max(region_foot_lon_tmp))
            carr_lon_min = np.append(carr_lon_min,np.min(region_foot_lon_tmp))
            
        else:
            
            carr_lat_max = np.append(carr_lat_max,np.nan)
            carr_lat_min = np.append(carr_lat_min,np.nan)
            
            carr_lon_max = np.append(carr_lon_max,np.nan)
            carr_lon_min = np.append(carr_lon_min,np.nan)
            
    
    indices = np.linspace(0,len(solar_lon_time)-1,num=len(solar_lon_time),dtype=int) #indices for whole set of encounter data points.
    non_q_index_arr = np.setdiff1d(indices, q_index_arr) #indices for non-quiescent wind.

    non_q_time = solar_lon_time[non_q_index_arr]
    non_q_foot_lon = sol_lon[non_q_index_arr]
    non_q_foot_lat = sol_lat[non_q_index_arr]
    non_q_exp_fact = exp_fact[non_q_index_arr]

    if save:
    #--------------------------save footpoints for quiescent and non-quiescent wind-----------------# 
        
        tplot_savepath='/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/data_outputs/'
        
        if enc != None:    
            tplot_savename = 'Enc_'+str(enc)+'_sorted_PFSS_data.cdf'
        else:
            tplot_savename = t0+'_'+tf+'_sorted_PFSS_data.cdf'
    
        
        #---totals---#
        
        t_tplot_time = solar_lon_time # total time
        
        t_tplot_sol_lon = sol_lon
        t_tplot_sol_lat = sol_lat
        tplot_carr_lon = carr_lon
        tplot_carr_lat = carr_lat
        tplot_r0_Rs = r0_Rs
        tplot_rss = np.array([rss])
        # tplot_polarity = polarity
        tplot_expansion = exp_fact
        
        #---quiescent---#
    
        q_tplot_time = region_time
        
        q_tplot_sol_lon = region_foot_lon
        q_tplot_sol_lat = region_foot_lat
        q_expansion = region_exp_fact
        
        #---non-quiescent---#
        
        nq_tplot_time = non_q_time # non quiescent time
        
        nq_tplot_sol_lon = non_q_foot_lon
        nq_tplot_sol_lat = non_q_foot_lat
        nq_expansion = non_q_exp_fact
        
        #---------tplot saves---------#
        
        pyt.store_data("solar_lon", data={'x':t_tplot_time, 'y':t_tplot_sol_lon}) #total solar longitude
        pyt.store_data("solar_lat", data={'x':t_tplot_time, 'y':t_tplot_sol_lat})
        pyt.store_data("PSP_lon", data={'x':t_tplot_time, 'y':tplot_carr_lon})
        pyt.store_data("PSP_lat", data={'x':t_tplot_time, 'y':tplot_carr_lat})
        
        pyt.store_data("PSP_Rs", data={'x':t_tplot_time, 'y':tplot_r0_Rs})
        pyt.store_data("rss",data={'x':tplot_rss, 'y':tplot_rss})
        
        # pyt.store_data("polarity",data={'x':t_tplot_time, 'y':tplot_polarity})
        pyt.store_data("expansion_factor",data={'x':t_tplot_time, 'y':tplot_expansion})
        
        #--- quiescent ---#
        
        pyt.store_data("q_solar_lon", data={'x':q_tplot_time, 'y':q_tplot_sol_lon}) #quiescent solar longitude
        pyt.store_data("q_solar_lat", data={'x':q_tplot_time, 'y':q_tplot_sol_lat})
        pyt.store_data("q_expansion_factor",data={'x':q_tplot_time, 'y':q_expansion})
        
        #---non quiescent---#
        
        pyt.store_data("nq_solar_lon", data={'x':nq_tplot_time, 'y':nq_tplot_sol_lon}) #non quiescent solar longitude
        pyt.store_data("nq_solar_lat", data={'x':nq_tplot_time, 'y':nq_tplot_sol_lat})
        pyt.store_data("nq_expansion_factor",data={'x':nq_tplot_time, 'y':nq_expansion})
    
        cdf_var_list = ["solar_lon","solar_lat","PSP_lon","PSP_lat","PSP_Rs","rss","Vsw_Rs","Temp_p","polarity","expansion_factor",
                        "q_solar_lon","q_solar_lat","q_expansion_factor","nq_solar_lon","nq_solar_lat","nq_expansion_factor"]
        
        pyt.tplot_save(cdf_var_list,tplot_savepath+tplot_savename) #saves the PFSS properties to a .cdf file

def select_rss(t0='2020-01-29',tf=None,enc=None,rss=2.5,r_tmp=None,plot=False,save_coords=False,rlim=65,
                  source='hmi',adapt_source='gong',peri=True,low_res=True,full_run=True,test_plot=False):
   
    
    file_tag = '_'+source
    
    Rs_km = 6.957e5*u.km #solar radius in km  
    
    if enc is not None:
        
        t0,tf = utils.encounter_dates(enc,rlim)

    if r_tmp==None:
        r_tmp=rss
    
    if tf==None:
        tf = pys.time_string(pys.time_float(t0)+86400)
    
    def set_axes_lims(ax):
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 180)
        
    if enc == None:
        enc = utils.encounter_check(t0)

    #-------------------------------IMPORT DATA-------------------------------#

    #------position data importing------#    

    psp.fields(trange=[t0,tf], datatype='ephem_spp_hg', level='l1',username=fields_id,password=fields_pass,last_version=True) #going to be used to plot parker position
    pos_data = pyt.get_data('position')
    
    pos_time_arr = pos_data[0]
    pos_data_arr = pos_data[1]
    
    x = pos_data_arr[:,0]*u.km
    y = pos_data_arr[:,1]*u.km
    z = pos_data_arr[:,2]*u.km
    
    #----SPC velocity data importing-----#  
    
    psp.spc(trange=[t0,tf],level='L3',username=sweap_id,password=sweap_pass,last_version=True)
        
    spc_data = pyt.get_data('psp_spc_vp_fit_RTN')
    
    if spc_data == None:
        spc_data = pyt.get_data('spp_spc_vp_fit_RTN')
        
    # breakpoint()
        
    spc_time_arr = spc_data[0]
    spc_data_arr = spc_data[1]

    vr_spc = spc_data_arr[:,0]
    
    vr_spc_clean = utils.sliding_median(vr_spc,275) #about one minute long windows at max cadence

    interpolating_func = interp1d(spc_time_arr, vr_spc_clean, kind='linear', fill_value='extrapolate')
    
    # Create a new array of time values with fixed cadence
    spc_time_down = np.arange(spc_time_arr[0], spc_time_arr[-1], 3)

    # Interpolate arr_two to the new time values
    spc_v_down = interpolating_func(spc_time_down)
    
    spc_time_arr = spc_time_down
    vr_spc = spc_v_down
    
    #------SPI velocity data importing------# 
    
    psp.spi(trange=[t0,tf],level='L3',datatype='spi_sf00',username=sweap_id,password=sweap_pass,last_version=True)

    vel_data = pyt.get_data('psp_spi_VEL_RTN_SUN')
    
    spi_time_arr = vel_data[0]
    spi_data_arr = vel_data[1]
    
    vr_spi = spi_data_arr[:,0]

    #------mag data from SPI product, really from FIELDS-------#
    
    psp.fields(trange=[t0,tf],level='l2',datatype='mag_RTN_1min',username=fields_id,password=fields_id,last_version=True)
    
    mag_data = pyt.get_data('psp_fld_l2_mag_RTN_1min')
    
    mag_time_arr = mag_data[0]
    mag_data_arr = mag_data[1]
    
    br_fields = mag_data_arr[:,0]
    bt_fields = mag_data_arr[:,1]
    bn_fields = mag_data_arr[:,2]
    
    sliding_med_br = utils.sliding_median(br_fields,120) #30 minute sliding window. 120 indices = 120 minutes.
    
    #---------------handling velocity data for Parker Spiral------------------#

    r0_time = pos_time_arr

    r0 = np.sqrt(x**2+y**2+z**2) #parker solar probe position height in kilometers
    r0_Rs = np.sqrt(x**2+y**2+z**2)/Rs_km*u.R_sun #height from center of sun in Solar Radii
    
    hyb_time, hyb_vr = utils.SPC_SPI_Construct(enc,spi_time_arr,vr_spi,spc_time_arr,vr_spc) #hybrid (hyb) velocity arrays
    
    Vsw = hyb_vr*u.km/u.s
    Vsw_Rs = ((hyb_vr*u.km/u.s)/Rs_km)*u.R_sun

    #---interpolate the position data up to velocity data rates---#
    r0_Rs_i = np.interp(hyb_time,r0_time,r0_Rs)
    r0_i = np.interp(hyb_time,r0_time,r0)
    x_i = np.interp(hyb_time,r0_time,x)
    y_i = np.interp(hyb_time,r0_time,y)
    z_i = np.interp(hyb_time,r0_time,z)
    
    r0_time_arr = hyb_time
    r0 = r0_i
    r0_Rs = r0_Rs_i
    x_pos = x_i
    y_pos = y_i
    z_pos = z_i
    
    #-----------------------scale down the pfss model--------------------#
    
    if low_res:
        
        # Original indices (or times)
        # original_indices = np.linspace(r0_time_arr[0], r0_time_arr[-1], num=pos_len)
        # New indices for downsampling
        low_res_times = np.linspace(r0_time_arr[0], r0_time_arr[-1], num=15000)
        # low_res_times = np.linspace(r0_time_arr[0], r0_time_arr[-1], num=5000)
        
        
        r0_Rs_i = np.interp(low_res_times,r0_time_arr,r0_Rs)
        r0_i = np.interp(low_res_times,r0_time_arr,r0)
        
        x_i = np.interp(low_res_times,r0_time_arr,x_pos)
        y_i = np.interp(low_res_times,r0_time_arr,y_pos)
        z_i = np.interp(low_res_times,r0_time_arr,z_pos)
        
        Vsw_i = np.interp(low_res_times,r0_time_arr,Vsw)
        Vsw_Rs_i = np.interp(low_res_times,r0_time_arr,Vsw_Rs)
        
        br_i = np.interp(low_res_times,mag_time_arr,br_fields)
        bt_i = np.interp(low_res_times,mag_time_arr,bt_fields)
        bn_i = np.interp(low_res_times,mag_time_arr,bn_fields)
        
        br_med_i = np.interp(low_res_times,mag_time_arr,sliding_med_br)
        
        r0_time_arr = low_res_times
        r0 = r0_i
        r0_Rs = r0_Rs_i
        x_pos = x_i
        y_pos = y_i
        z_pos = z_i
        Vsw = Vsw_i
        Vsw_Rs = Vsw_Rs_i
        
        br_fields = br_i
        bt_fields = bt_i
        bn_fields = bn_i
        
        sliding_med_br = br_med_i
        
    
    fields_polarity_no_sbs = sliding_med_br/np.abs(sliding_med_br)
    # fields_polarity = br_fields/np.abs(br_fields) #just ±1 for positive/negative Br
    
    # breakpoint()
    
    convert_to_datetime = np.vectorize(lambda epoch: datetime.utcfromtimestamp(epoch))
    datetime_arr = convert_to_datetime(r0_time_arr)
    
    #------------------------Solve for Parker Spiral, I think -------------------#
    # breakpoint()
    # calculate time solar wind measured by PSP launched from the source surface.
    del_time = np.abs(rss*u.R_sun-r0_Rs)/Vsw_Rs #should be in seconds. Time traveled from Rss
    
    corrected_time = r0_time_arr*u.s-del_time #epoch time which the solar wind launched, assuming it was flying at the same speed the whole time. (ASSUMPTION)

    carr_lon_psp = np.arctan2(y_pos, x_pos).to(u.deg)

    carr_lon_psp = np.where(carr_lon_psp < 0 * u.deg, carr_lon_psp + 360 * u.deg, carr_lon_psp)

    carr_lat_psp = np.arcsin(z_pos / r0).to(u.deg)

    w = 360/(25.38*86400)*u.deg/u.s # angular frequency of the sun in degrees/sec
    # w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec

    sintheta = np.sin((90*u.deg-carr_lat_psp).to(u.radian)) #sin of the azimuthal angle, which is 90 degrees minus the latitude

    src_lon = carr_lon_psp - (w*sintheta/Vsw_Rs)*(rss*u.R_sun-r0_Rs) #carrington longitude of the PSP traced by a Parker Spiral on the source surface

    # Create a new array with non-NaN elements
    non_nan_corrected_time = corrected_time[~np.isnan(corrected_time)]

    #----------------------------find date of perihelion----------------------#
    
    per_where = np.where(r0_Rs == np.min(r0_Rs))
    per_where = per_where[0]
    
    peri_time = r0_time_arr[per_where]
    peri_date = pys.time_string(peri_time[0]) #select the date for perihelion
    
    peri_date_t0 = pys.time_string(peri_time[0]-43200) #select the date for perihelion
    peri_date_tf = pys.time_string(peri_time[0]+43200) #select the date for perihelion
    
    #--------------------------PFSS MODEL START-----------------------------#
    
    if source=='hmi':
        
        if peri:
            t_start_hmi = parse_time(peri_date_t0)
            t_end_hmi = parse_time(peri_date_tf)
        # pfss_out = hmi2pfss(dt=t0_del_dt)
        else:
            t_start_hmi = parse_time(t0)
            t_end_hmi = parse_time(tf)
        # results_hmi = Fido.search(a.jsoc.Notify(os.environ["JSOC_EMAIL"]),a.jsoc.Time(t_start_hmi.value,t_end_hmi.value),a.jsoc.Series('hmi.mrdailysynframe_720s'))
        # breakpoint()
        results_hmi = Fido.search(a.jsoc.Notify(os.environ["JSOC_EMAIL"]),a.jsoc.Time(t_start_hmi.value,t_end_hmi.value),a.jsoc.Series('hmi.mrdailysynframe_small_720s'))
        
        # res_atrs = results_hmi['JSOC']
        # result_times = res_atrs['T_REC']
        
        # filenames = []
        # path_check = []
        # for i in result_times:
        #     tmp_str = str(i)
        #     full_path = os.environ['SUNPY_DATA_DIR']+'/hmi/'+'hmi.mrdailysynframe_small_720s.'+tmp_str[:4]+tmp_str[5:7]+tmp_str[8:13]+tmp_str[14:16]+tmp_str[17:23]+'.data.fits'
        #     filenames.append(full_path)
            
        #     if os.path.isfile(full_path):
        #         path_check.append(True)
        #     else:
        #         path_check.append(False)
        
        # if np.sum(path_check)==len(path_check):
        #     pass
        # else:
        #     true_where = np.where(np.array(path_check))
        #     true_where = true_where[0]
        #     filenames = np.array(filenames)[true_where]
        
        filenames = Fido.fetch(results_hmi, path=os.environ['SUNPY_DATA_DIR']+'/hmi')
        
    if source=='gong':
        
        if peri:
            t0 = peri_date_t0
            tf = peri_date_tf
        # pfss_out = hmi2pfss(dt=t0_del_dt)
        else:
            pass
        
        filenames = pys.adapt.gong(trange=[t0,tf])
        
        # pfss_out = gong2pfss(dt=t0_del_dt)
    if source=='adapt':
        
        if peri:
            t0 = peri_date_t0
            tf = peri_date_tf
        # pfss_out = hmi2pfss(dt=t0_del_dt)
        else:
            pass
        
        filenames = pys.adapt.adapt(trange=[t0,tf],adapt_source=adapt_source)
    
    
    first_file = filenames[0]
    
    ###############################################################################
    # The PFSS solution is calculated on a regular 3D grid in (phi, s, rho), where
    # rho = ln(r), and r is the standard spherical radial coordinate. We need to
    # define the number of rho grid points, and the source surface radius.
    nrho = 35
    rss = rss
    
    if peri:
        
        hdul = fits.open(first_file)
        
        if source=='gong':
            dates = hdul[0].header["DATE-OBS"]+'/'+hdul[0].header["TIME-OBS"]
            
            pfss_map = sunpy.map.Map(first_file)
            pfss_map.meta['rsun'] = sunpy.sun.constants.radius.value/pfss_map.meta['cdelt1']
            # pfss_in = pfss.Input(pfss_map, nrho, rss)
            # pfss_out = pfss.pfss(pfss_in)
            
        if source=='adapt':
            dates = hdul[0].header["MAPTIME"][:10]+'/'+hdul[0].header["MAPTIME"][11:]
            
            pfss_map = pfss.utils.load_adapt(first_file)
            pfss_map_0 = pfss_map[0]
            # pfss_map_mean = pfss_map[-1]
            # pfss_map = sunpy.map.Map(br_adapt,adapt_map[0].header)
            pfss_map_cea = pfss.utils.car_to_cea(pfss_map_0)
            # breakpoint()
           	# pfss_in = pfss.Input(pfss_map_tmp, nrho, rss)
            # pfss_in = pfss.Input(pfss_map_cea, nrho, rss)
            # pfss_out = pfss.pfss(pfss_in)
            
        if source=='hmi':
            
            hmi_data, hmi_header = fits.getdata(first_file, header=True)
            
            hmi_data[np.isnan(hmi_data)]=np.nanmean(hmi_data)
            
            hmi_header_new = utils.fix_hmi_meta(hmi_header)
            
            pfss_map = sunpy.map.Map(hmi_data,hmi_header_new)
            

        rss_linspace = np.linspace(1.5,3.5,num=21)
        percent_polarity_accuracy = []
        
        for rss in rss_linspace:
            rss = np.round(rss,1)
            src_lon_i = carr_lon_psp - (w*sintheta/Vsw_Rs)*(rss*u.R_sun-r0_Rs) #iterate the source longitude for different rss
            # print(rss)
            
            pfss_in = pfss.Input(pfss_map, nrho, rss)
            pfss_out = pfss.pfss(pfss_in)
    
            tracer = pfss.tracing.FortranTracer()
            r = rss * const.R_sun
            
            # r2 = (r_tmp-0.2)*const.R_sun
            
            lat_tmp = carr_lat_psp
            lon_tmp = src_lon_i
            
            # lat_tmp, lon_tmp = lat_tmp.ravel() * u.deg, lon_tmp.ravel() * u.deg
            lat_tmp, lon_tmp = lat_tmp.ravel(), lon_tmp.ravel()
            
            seeds = SkyCoord(lon_tmp, lat_tmp, r, frame=pfss_out.coordinate_frame)
    
            field_lines_tmp = tracer.trace(seeds, pfss_out)
            
            # expans_fact = np.array(field_lines_tmp.expansion_factors) #expansion factors
            
            pfss_time_arr = np.array(r0_time_arr)
            pfss_polarity_arr = -field_lines_tmp.polarities
            
            where = np.where(pfss_polarity_arr==fields_polarity_no_sbs)
            where = where[0]
            
            percent_same_pol = len(where)/len(fields_polarity_no_sbs)
            percent_polarity_accuracy.append(percent_same_pol)

        save_name = 'rss_vs_accuracy_enc_'+str(enc)+file_tag+'.csv'
        save_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/Rss vs Accuracy/'
        # Example 1D lists
        list1 = list(rss_linspace)
        list2 = percent_polarity_accuracy
        
        # Write to CSV
        with open(save_path+save_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(zip(list1, list2))  # Write each pair as a row

    
    else:
        ###NEED TO CONTINUE WORKING HERE. IMPORTING EACH GONG MAP BUT STILL NEED TO CALCULATE PFSS SOLUTIONS FOR EACH.###
    
        fits_dates = []
        fits_dts = []
        
        #retrieve dates for each file
        
        for file in filenames:
            with fits.open(file) as hdul:
                # Access the primary HDU
                # breakpoint()
                if source=='gong':
                    dates = hdul[0].header["DATE-OBS"]+'/'+hdul[0].header["TIME-OBS"]
                    
                if source=='adapt':
                    dates = hdul[0].header["MAPTIME"][:10]+'/'+hdul[0].header["MAPTIME"][11:]
                    
                if source=='hmi':
                    # breakpoint()
                    
                    date_str = hdul[0].header['T_OBS']
    
                    # Parse the original date string
                    dt = datetime.strptime(date_str, "%Y.%m.%d_%H:%M:%S.%f_TAI")
    
                    # Format the datetime object to the desired format
                    formatted_date = dt.strftime("%Y-%m-%d/%H:%M:%S.%f")
                    
                    hdul[0].header['DATE-OBS'] = formatted_date
                    
                    dates = hdul[0].header["DATE-OBS"]
                
                fits_dates.append(dates)
                fits_dts.append(datetime.fromisoformat(dates))
        
        #------------- select magnetogram every three days going forward and backward from perihelion --------------# 
        
        fits_dates = np.array(fits_dates)
        fits_dts = np.array(fits_dts)
        
        # date_list = [dt.date() for dt in fits_dts]
        # date_list = np.vectorize(lambda dts: dts.date())(fits_dates)

        peri_dt = datetime.fromisoformat(peri_date)
        
        dif_arr = np.abs(fits_dts-peri_dt)
        
        ref_ind = np.where(dif_arr == min(dif_arr))        
        # ref_ind = np.where(np.array(date_list) == peri_dt)
        ref_ind = ref_ind[0][0] #reference index of perihelion magnetogram
        
        closest_peri_dt = fits_dts[ref_ind]
        
        day_num = 3
        # result,day_indices = find_every_i_days(date_array,ref_ind,day_num)

        # Input reference date (e.g., an element near the middle of the array, including time)
        reference_date = closest_peri_dt
        
        # Find the index of the reference date in fits_dts
        reference_idx = np.argmin(np.abs(fits_dts - reference_date))
        
        # Time delta for 3 days
        delta = timedelta(days=day_num)
        
        # Initialize the result list and the set of visited indices
        result_dates = [fits_dts[reference_idx]]
        visited_indices = set([reference_idx])
        
        # Function to find the closest datetime object in a given direction (forward or backward)
        def closest_date(target_date, direction="forward"):
            if direction == "forward":
                candidates = fits_dts[reference_idx + 1:]  # forward direction
            else:
                candidates = fits_dts[:reference_idx]  # backward direction
            
            closest_date = None
            min_diff = timedelta.max
        
            # Search for the closest match
            for date in candidates:
                diff = abs(date - target_date)
                if diff < min_diff:
                    min_diff = diff
                    closest_date = date
            
            return closest_date
        
        # Iterate forward and backward, looking for closest dates
        for i in range(1,6):  # We can limit how many steps forward/backward we want
            # Search forward direction (e.g., +3 days)
            next_date = closest_date(reference_date + i*delta, "forward")
            if next_date is not None:
                result_dates.append(next_date)
                # reference_date = next_date
                visited_indices.add(np.argmin(np.abs(fits_dts - reference_date)))
            
            # Search backward direction (e.g., -3 days)
            prev_date = closest_date(reference_date - i*delta, "backward")
            if prev_date is not None:
                result_dates.insert(0, prev_date)
                # reference_date = prev_date
                visited_indices.add(np.argmin(np.abs(fits_dts - reference_date)))
            
            # print(i)

        result_dates = np.unique(result_dates)
        
        final_date_set = set(result_dates)

        # Find indices using a list comprehension with set lookup
        indices = [i for i, date in enumerate(fits_dts) if date in final_date_set]
        
        # print(indices)  # Output: [0, 2]
        
                
        # result,day_indices = find_every_i_days(fits_dts,ref_ind,day_num)
        # breakpoint()
        
        filenames = [filenames[index] for index in indices]
        fits_dates = [fits_dates[index] for index in indices]
        fits_dts = [fits_dts[index] for index in indices]
        
        #----------------------- check which datapoints belong to which magnetogram ------------------------#
        # breakpoint()
        
        fits_floats = pys.time_float(fits_dates)
        
        indices = utils.group_elements(non_nan_corrected_time.value,fits_floats)
        
        unique_indices, ind_count = np.unique(indices,return_counts=True)
                
        rss_linspace = np.linspace(1.5,3.5,num=21)
        percent_polarity_accuracy = []
        
        
        # breakpoint()
        for rss in rss_linspace:
            
            # breakpoint()
            rss = np.round(rss,1)
            src_lon_i = carr_lon_psp - (w*sintheta/Vsw_Rs)*(rss*u.R_sun-r0_Rs) #iterate the source longitude for different rss
            # print(rss)
            

            pfss_time_arr = np.array([])
            pfss_polarity_arr = np.array([])

            for ind in unique_indices:
                
                # gong_fname = pfss.sample_data.get_gong_map()
                # gong_map = sunpy.map.Map(gong_fname)
                
                long_where = np.where(indices==ind)
                
                if source=='gong':
                    # breakpoint()
                    pfss_map = sunpy.map.Map(filenames[ind])
                    pfss_map.meta['rsun'] = sunpy.sun.constants.radius.value/pfss_map.meta['cdelt1']
                    pfss_in = pfss.Input(pfss_map, nrho, rss)
                    pfss_out = pfss.pfss(pfss_in)
                    
                                    
                    # ss_br = pfss_out.source_surface_br
                    # Create the figure and axes
                    # fig = plt.figure()
                    # ax = plt.subplot(projection=ss_br)
                    
                    # # Plot the source surface map
                    # ss_br.plot()
                    # # Plot the polarity inversion line
                    # ax.plot_coord(pfss_out.source_surface_pils[0])
                    # # Plot formatting
                    # plt.colorbar()
                    # # ax.set_title('Source surface magnetic field, rss='+str(rss))
                    # ax.set_title('GONG map, '+str(pfss_map.date))
                    # # ax.set_xlim(-180,180)
                    
                    # plt.show()
                    
                
                if source=='adapt':
        
                    pfss_map = pfss.utils.load_adapt(filenames[ind])
                    pfss_map_0 = pfss_map[0]
                    # pfss_map_mean = pfss_map[-1]
                    # pfss_map = sunpy.map.Map(br_adapt,adapt_map[0].header)
                    pfss_map_cea = pfss.utils.car_to_cea(pfss_map_0)
                    # breakpoint()
                   	# pfss_in = pfss.Input(pfss_map_tmp, nrho, rss)
                    pfss_in = pfss.Input(pfss_map_cea, nrho, rss)
                    pfss_out = pfss.pfss(pfss_in)
                       
                if source=='hmi':
                    
                    # if ind == [2]: #just a test
                        
                    hmi_data, hmi_header = fits.getdata(filenames[ind], header=True)
                    
                    hmi_data[np.isnan(hmi_data)]=np.nanmean(hmi_data)
                    
                    # breakpoint()
                    hmi_header_new = utils.fix_hmi_meta(hmi_header)
                    pfss_map = sunpy.map.Map(hmi_data,hmi_header_new)

                    pfss_map.meta['rsun'] = sunpy.sun.constants.radius.value
                    pfss_in = pfss.Input(pfss_map, nrho, rss)
                    pfss_out = pfss.pfss(pfss_in)
                
                
                    # ss_br = pfss_out.source_surface_br
                    # Create the figure and axes
                    # fig = plt.figure()
                    # ax = plt.subplot(projection=ss_br)
                    
                    # # Plot the source surface map
                    # ss_br.plot()
                    # # Plot the polarity inversion line
                    # ax.plot_coord(pfss_out.source_surface_pils[0])
                    # # Plot formatting
                    # plt.colorbar()
                    # # ax.set_title('Source surface magnetic field, rss='+str(rss))
                    # ax.set_title('HMI map, '+str(pfss_map.date))
                    # # ax.set_xlim(-180,180)
                    
                    # plt.show()
                

                tracer = pfss.tracing.FortranTracer()
                r = rss * const.R_sun
                
                lat_tmp = carr_lat_psp[long_where]
                lon_tmp = src_lon_i[long_where]
                # r0_Rs_tmp = r0_Rs[long_where]
                time_tmp = r0_time_arr[long_where]
                
                # plt.plot(lon_tmp,lat_tmp)
                
                # lat_tmp, lon_tmp = lat_tmp.ravel() * u.deg, lon_tmp.ravel() * u.deg
                lat_tmp, lon_tmp = lat_tmp.ravel(), lon_tmp.ravel()
                
                seeds = SkyCoord(lon_tmp, lat_tmp, r, frame=pfss_out.coordinate_frame)
        
                field_lines_tmp = tracer.trace(seeds, pfss_out)

                pfss_time_arr = np.append(pfss_time_arr,time_tmp)
                pfss_polarity_arr = np.append(pfss_polarity_arr, field_lines_tmp.polarities)
            
            pfss_time_arr = pfss_time_arr[np.argsort(pfss_time_arr)] #makes sure all the field lines are in proper order in time.
            pfss_polarity_arr = pfss_polarity_arr[np.argsort(pfss_time_arr)] 
            
            
            fields_pol_non_nan = fields_polarity_no_sbs[~np.isnan(corrected_time)]
            
            where = np.where(pfss_polarity_arr==fields_pol_non_nan)
            where = where[0]
            
            percent_same_pol = len(where)/len(fields_pol_non_nan)
            percent_polarity_accuracy.append(percent_same_pol)
            
            
            
            
            # fig = plt.figure(figsize=(17.5,7))
            # ax1 = fig.add_subplot(211)
            
            # ax1.plot(fields_polarity_no_sbs,color='tab:blue',label='FIELDS polarity')
            # ax1.set_title("FIELDS polarity vs PFSS predicted polarity, rss="+str(rss))
            # plt.legend()
            
            # ax2 = fig.add_subplot(212)
            
            # ax2.plot(pfss_polarity_arr,color='tab:green',label='PFSS polarity')
            
            # plt.legend()
             
            # plt.show()
            
        
        
        save_name = 'rss_vs_accuracy_enc_'+str(enc)+file_tag+'.csv'
        save_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/Rss vs Accuracy/'
        # Example 1D lists
        list1 = list(rss_linspace)
        list2 = percent_polarity_accuracy
        
        # Write to CSV
        with open(save_path+save_name, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(zip(list1, list2))  # Write each pair as a row

        
        
    # breakpoint()
    if test_plot:
        
        #----------------# PFSS FIELD MODEL FIGURE #----------------#
        pfss_in = pfss.Input(pfss_map, nrho, rss)
        pfss_out = pfss.pfss(pfss_in)

        tracer = pfss.tracing.FortranTracer()
        r = r_tmp * const.R_sun
        
        
        # r2 = (r_tmp-0.2)*const.R_sun
        
        lat_tmp = carr_lat_psp
        lon_tmp = src_lon
        
        lat_tmp, lon_tmp = lat_tmp.ravel(), lon_tmp.ravel()
        
        seeds = SkyCoord(lon_tmp, lat_tmp, r, frame=pfss_out.coordinate_frame)

        field_lines_tmp = tracer.trace(seeds, pfss_out)
        
        pfss_polarity_arr = field_lines_tmp.polarities
        
        br_pfss = utils.PFSS_Br_estimation(pfss_out,seeds,r0_Rs,rss) #calculates the Br estimated by the PFSS Model.
        
        
        fig_br = plt.figure(figsize=(17.5,7))
        ax = fig_br.add_subplot(111)
        
        ax.plot(datetime_arr,sliding_med_br,color='tab:blue',label='FIELDS measured Br')
        ax.plot(datetime_arr,br_pfss,color='tab:orange',label='PFSS estimated Br') #threw in the minus sign to make the model work. Seems to be backwards.
        
        ax.set_title("Radial Magnetic Field PSP vs PFSS",fontsize=26)
        ax.set_ylabel("Br @ PSP in nT",fontsize=20)
        
        # ax.text("Br for PFSS flipped sign and scaled as in Badman 2020.")
        ax.text(datetime_arr[int(4.4*len(datetime_arr)/8)], -65, "Br for PFSS sign flipped here, scaled as in Badman 2020.", fontsize=15, color='green', ha='left')
        
        # Set the locator to show a maximum of 5 ticks on the x-axis
        # ax.xaxis.set_major_locator(mdates.MaxNLocator(nbins=5))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        # Rotate the labels for readability
        plt.xticks(rotation=45,fontsize=15)
        
        plt.grid(True)
        plt.legend()
        plt.show()
        
        
        
        fig = plt.figure(figsize=(17.5,7))
        ax1 = fig.add_subplot(121, projection='3d')
        
        n_ind = list(range(len(carr_lon_psp)))
        # n_lines = 250
        n_lines = 25
        
        idx = np.round(np.linspace(0, max(n_ind) - 1, n_lines)).astype(int) # number of lines
        
        j, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x1 = np.cos(j)*np.sin(v)
        y1 = np.sin(j)*np.sin(v)
        z1 = np.cos(v)
        ax1.plot_wireframe(x1, y1, z1, color="w", edgecolor="gray")
    
        ax2 = fig.add_subplot(122)
    
        j = 0
    
        # for i in field_lines_tmp:
        field_lines_2_plot = field_lines_tmp[idx]
        
            
        # breakpoint()
        # field_lines_2_plot = i
        # field_lines_2_plot = np.append(field_lines_2_plot,field_lines2)
        
        for field_line in field_lines_2_plot:
                
            # if sub_ind in idx:
            color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
            coords = field_line.coords
            coords.representation_type = 'cartesian'
            ax1.plot(coords.x / const.R_sun,
                    coords.y / const.R_sun,
                    coords.z / const.R_sun,
                    color=color, linewidth=1)

        
            # breakpoint()

    
            #-----------------# PARKER SPIRAL FIGURE #------------------#
            
            # fig = plt.figure(figsize=(10,10))
            # ax2 = fig.add_subplot(122)

            sub_ind = idx[j]
    
            # ib = idx[j]
            # breakpoint()
            r = np.linspace(rss*u.R_sun,r0_Rs[sub_ind],100)
            spiral_plot_lon = carr_lon_psp[sub_ind] - (w*sintheta[sub_ind]/Vsw_Rs[sub_ind])*(r-r0_Rs[sub_ind])
            # spiral_plot_lon = carr_lon_psp[ib] - (w*sintheta[ib]/Vsw_Rs)*(r-r0_Rs[ib])
            
            # spiral_x = r*np.cos(spiral_plot_lon*np.pi/180)
            # spiral_y = r*np.sin(spiral_plot_lon*np.pi/180)
            spiral_x = r*np.cos(spiral_plot_lon.to(u.rad))
            spiral_y = r*np.sin(spiral_plot_lon.to(u.rad))
    
            # if field_lines_2_plot[sub_ind].polarity == 1:
            if field_line.polarity == 1:
                color = 'tab:red'
            else:
                color = 'tab:blue'
    
            ax2.plot(spiral_x,spiral_y,color=color)
            
            j+=1
                
        
        
        # Define circle parameters
        center_x, center_y = 0, 0  # Center of the circle
        radius = rss             # Radius of the circle
        
        # Generate points for the circle
        theta = np.linspace(0, 2 * np.pi, 500)  # Angle from 0 to 2π
        circ_x = center_x + radius * np.cos(theta)  # x-coordinates
        circ_y = center_y + radius * np.sin(theta)  # y-coordinates
        
        ax2.plot(circ_x, circ_y, label="PFSS Model Outer Boundary",color='black')
        
        # ax1.view_init(10, 40) #VIEW THAT SAM USED FOR ENC 1
        # ax1.view_init(10, 60)
        ax1.view_init(40, 280)
        # ax1.set_title('PFSS solution with Source Surface at '+str(rss)+' Rs',fontsize=20)
        ax1.set_title(r"PFSS solution with Source Surface at "+str(rss)+" Rs", fontsize=20)
        
        ax1.text2D(0.5,-0.08,"using HMI Synoptic Map",fontsize=20,transform=ax1.transAxes,horizontalalignment='center')
        # text2D(0.05, 0.95, "2D Text", transform=ax.transAxes
        
    
        utils.set_axes_equal(ax1)
        
        # Make panes transparent
        ax1.xaxis.pane.fill = False # Left pane
        ax1.yaxis.pane.fill = False # Right pane
        
        # Remove grid lines
        ax1.grid(False)
        
        # Remove tick labels
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        
        # Transparent spines
        ax1.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Transparent panes
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # No ticks
        ax1.set_xticks([]) 
        ax1.set_yticks([]) 
        ax1.set_zticks([])
        
        ax1.set_xlim3d(-1.75, 1.75)
        ax1.set_ylim3d(-1.75, 1.75)
        ax1.set_zlim3d(-1.4, 1.4)
            
        ax2.set_title('Parker Spiral Propagation to PSP Position',fontsize=20)
        ax2.plot(x/Rs_km,y/Rs_km,color='black',label='PSP Position')
        ax2.set_ylim(-250,250)
        ax2.set_xlim(-250,250)
        ax2.set_xlabel('Carrington X in Rs',fontsize=16)
        ax2.set_ylabel('Carrington Y in Rs',fontsize=16)
        
        # We change the fontsize of minor ticks label 
        ax2.tick_params(axis='both', which='major', labelsize=16)
        # ax2.tick_params(axis='both', which='minor', labelsize=8)
        # ax.set_aspect("equal")
        plt.show()
               
def footpoint_plot(t0='2020-01-29',tf=None,enc=None,save_coords=False,plot=True,wavelen=171,instr='AIA',
                   region_labels=False,plot_shape='circ',zoom=False,save=True):

    euv_chn_opts = [94, 131, 171, 193, 211, 304, 335]
    uv_chn_opts = [1600, 1700]
    vis_chn_opts = [4500]

    euv_aia = 'aia.lev1_euv_12s'
    uv_aia = 'aia.lev1_uv_24s'
    vis_aia = 'aia.lev1_vis_1h'
    
    if wavelen in euv_chn_opts:
        aia_lab = euv_aia
        sec = 12.
    elif wavelen in uv_chn_opts:
        aia_lab = uv_aia
        sec = 24.
    elif wavelen in vis_chn_opts:
        aia_lab = vis_aia
        sec = 3600.
    else:
        
        print("Channel option not valid, setting wavelength to 171A.")
        aia_lab = euv_aia
        wavelen=171
        sec = 12.


    #------------------------------ Set path to tplot and quiescent region files ----------------------------------#

    if tf==None:
        tf = pys.time_string(pys.time_float(t0)+86400)

    if enc != None:    
        tplot_savename = 'Enc_'+str(enc)+'_footpoint_coords.cdf'
        qregion_savename = 'enc_'+str(enc)+'_regions_raw.csv'
    else:
        tplot_savename = t0+'_'+tf+'_footpoint_coords.cdf'
        qregion_savename = t0+'_'+tf+'_regions_raw.csv'
        
        t0float = pys.time_float(t0)
        enc = 1
        for i in enc_flt:
            # if t0float>i[0] and t0float<i[1]:
            #     encounter = enc
            enc+=1
        if tf==None:
            tffloat = pys.time_float(t0)+86400
        else:
            tffloat=pys.time_float(tf)
        
    tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/footpoints/'
    qregion_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/psp_regions/region_data/'
    
    lanes_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/fits/'
    lanes_savename = 'lanes_gaussian_fwhm7_run_avg.fits'
    # fits_savename = 'CR2232.fits'

    # breakpoint()
    #------------------------------ Read in Tplot Variables ----------------------------------#
    
    pyt.tplot_restore(tplot_savepath+tplot_savename)
    
    rss_data = pyt.get_data('rss')
    rss = rss_data[0]
    rss = rss[0]
    
    carr_lon_data = pyt.get_data('PSP_lon')
    carr_lon_time = carr_lon_data[0]
    carr_lon = carr_lon_data[1]
    
    carr_lat_data = pyt.get_data('PSP_lat')
    carr_lat = carr_lat_data[1]
    
    solar_lon_data = pyt.get_data('solar_lon')
    solar_lon_time = solar_lon_data[0]
    sol_lon = solar_lon_data[1]
    
    sol_lat_data = pyt.get_data('solar_lat')
    # sol_lat_time = sol_lat_data[0]
    sol_lat = sol_lat_data[1]
    
    r0_Rs_data = pyt.get_data('PSP_Rs')
    r0_Rs_time = r0_Rs_data[0]
    r0_Rs = r0_Rs_data[1]
    
    
    exp_fact_data = pyt.get_data('expansion_factor')
    exp_fact = exp_fact_data[1]
    
    rs_where = np.where(r0_Rs==min(r0_Rs))
    
    date = pys.time_string(r0_Rs_time[rs_where])
    t0d = date[0]
    tfd = pys.time_string(pys.time_float(t0d)+sec)
    
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
        qregion_flt = np.array(qregion_lst) 
        reg_where = np.where((qregion_flt[:,0]>t0float)&(qregion_flt[:,1]<tffloat))
        
        regions_arr = qregion_array[reg_where,:]
        
    #---------------- find footpoints associated with quiescent regions --------------------#
    
    region_time = np.array([])
    region_foot_lon = np.array([])
    region_foot_lat = np.array([])
    region_exp_fact = np.array([])
    
    carr_lat_max = np.array([])
    carr_lat_min = np.array([])
    carr_lon_max = np.array([])
    carr_lon_min = np.array([])
    
    long_region_time = np.array([])
    long_region_foot_lon = np.array([])
    long_region_foot_lat = np.array([])
    long_exp_fact = np.array([])
    
    q_index_arr = np.array([])
    
    for k in regions_arr:

        pre_flt = k[:2]
        reg_flt = pys.time_float(pre_flt)
        point_where = np.where((carr_lon_time>reg_flt[0])&(carr_lon_time<reg_flt[1]))
        point_where = point_where[0]
        
        q_index_arr = np.append(q_index_arr,point_where)
        
        region_time_tmp = solar_lon_time[point_where]
        region_foot_lon_tmp =  sol_lon[point_where]
        region_foot_lat_tmp = sol_lat[point_where]
        region_exp_tmp = exp_fact[point_where]

        region_time = np.append(region_time,region_time_tmp)
        region_foot_lon = np.append(region_foot_lon,region_foot_lon_tmp)
        region_foot_lat = np.append(region_foot_lat,region_foot_lat_tmp)
        region_exp_fact = np.append(region_exp_fact,region_exp_tmp)
        
        if len(point_where) != 0:
        
            carr_lat_max = np.append(carr_lat_max,np.max(region_foot_lat_tmp))
            carr_lat_min = np.append(carr_lat_min,np.min(region_foot_lat_tmp))
            
            carr_lon_max = np.append(carr_lon_max,np.max(region_foot_lon_tmp))
            carr_lon_min = np.append(carr_lon_min,np.min(region_foot_lon_tmp))
            
        else:
            
            carr_lat_max = np.append(carr_lat_max,np.nan)
            carr_lat_min = np.append(carr_lat_min,np.nan)
            
            carr_lon_max = np.append(carr_lon_max,np.nan)
            carr_lon_min = np.append(carr_lon_min,np.nan)
            
    
    indices = np.linspace(0,len(solar_lon_time)-1,num=len(solar_lon_time),dtype=int) #indices for whole set of encounter data points.
    non_q_index_arr = np.setdiff1d(indices, q_index_arr) #indices for non-quiescent wind.

    non_q_time = solar_lon_time[non_q_index_arr]
    non_q_foot_lon = sol_lon[non_q_index_arr]
    non_q_foot_lat = sol_lat[non_q_index_arr]
    non_q_exp_fact = exp_fact[non_q_index_arr]
    
    # breakpoint()
    
    if save:
    #--------------------------save footpoints for quiescent and non-quiescent wind-----------------# 
        
        tplot_savepath='/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/data_outputs/'
        
        if enc != None:    
            tplot_savename = 'Enc_'+str(enc)+'_sorted_PFSS_data.cdf'
        else:
            tplot_savename = t0+'_'+tf+'_sorted_PFSS_data.cdf'
    
        #---totals---#
        
        t_tplot_time = solar_lon_time # total time
        
        t_tplot_sol_lon = sol_lon
        t_tplot_sol_lat = sol_lat
        tplot_carr_lon = carr_lon
        tplot_carr_lat = carr_lat
        tplot_r0_Rs = r0_Rs
        tplot_rss = np.array([rss])
        tplot_expansion = exp_fact
        
        #---quiescent---#
        
        region_time = region_time # total time
        
        region_foot_lon = region_foot_lon
        region_foot_lat = region_foot_lat
        region_exp_fact = region_exp_fact
        
        #---non-quiescent---#
        
        nq_tplot_time = non_q_time # non quiescent time
        
        nq_tplot_sol_lon = non_q_foot_lon
        nq_tplot_sol_lat = non_q_foot_lat
        nq_expansion = non_q_exp_fact
        
        #----long quiescent----#
        
        long_tplot_time = long_region_time # non quiescent time
        
        long_tplot_sol_lon = long_region_foot_lon
        long_tplot_sol_lat = long_region_foot_lat
        long_expansion = long_exp_fact
        
        
        #---------tplot saves---------#
        
        pyt.store_data("solar_lon", data={'x':t_tplot_time, 'y':t_tplot_sol_lon}) #total solar longitude
        pyt.store_data("solar_lat", data={'x':t_tplot_time, 'y':t_tplot_sol_lat})
        pyt.store_data("PSP_lon", data={'x':t_tplot_time, 'y':tplot_carr_lon})
        pyt.store_data("PSP_lat", data={'x':t_tplot_time, 'y':tplot_carr_lat})
        
        pyt.store_data("PSP_Rs", data={'x':t_tplot_time, 'y':tplot_r0_Rs})
        pyt.store_data("rss",data={'x':tplot_rss, 'y':tplot_rss})
        
        # pyt.store_data("polarity",data={'x':t_tplot_time, 'y':tplot_polarity})
        pyt.store_data("expansion_factor",data={'x':t_tplot_time, 'y':tplot_expansion})
        
        #--- quiescent ---#
        
        pyt.store_data("q_solar_lon", data={'x':region_time, 'y':region_foot_lon}) #quiescent solar longitude
        pyt.store_data("q_solar_lat", data={'x':region_time, 'y':region_foot_lat})
        pyt.store_data("q_expansion_factor",data={'x':region_time, 'y':region_exp_fact})
        
        #---non quiescent---#
        
        pyt.store_data("nq_solar_lon", data={'x':nq_tplot_time, 'y':nq_tplot_sol_lon}) #non quiescent solar longitude
        pyt.store_data("nq_solar_lat", data={'x':nq_tplot_time, 'y':nq_tplot_sol_lat})
        pyt.store_data("nq_expansion_factor",data={'x':nq_tplot_time, 'y':nq_expansion})
        
        #---long quiescent---#
        
        pyt.store_data("long_solar_lon", data={'x':long_tplot_time, 'y':long_tplot_sol_lon}) #non quiescent solar longitude
        pyt.store_data("long_solar_lat", data={'x':long_tplot_time, 'y':long_tplot_sol_lat})
        pyt.store_data("long_expansion_factor",data={'x':long_tplot_time, 'y':long_expansion})
        
        
    
        cdf_var_list = ["solar_lon","solar_lat","PSP_lon","PSP_lat","PSP_Rs","rss","Vsw_Rs","polarity","expansion_factor",
                        "q_solar_lon","q_solar_lat","q_expansion_factor","nq_solar_lon","nq_solar_lat","nq_expansion_factor",
                        "long_solar_lon","long_solar_lat","long_expansion_factor"]
        
        pyt.tplot_save(cdf_var_list,tplot_savepath+tplot_savename) #saves the PFSS properties to a .cdf file
    
    
    
    if plot:
        
        #------------------------------ Read in AIA Images ----------------------------------#
        
        """ Check to see if AIA file already exists. They are large files and the program runs slow if sunpy gets carried away."""
        
        date_form = pys.time_string(r0_Rs_time[rs_where],fmt='%Y-%m-%dT%H')
        
        file_path = os.environ.get('SUNPY_DATA_DIR')+'/AIA/'+str(wavelen)+'/'+t0d[:4]+'/'+t0d[5:7]+'/'+t0d[8:10]+'/'
        file_name = aia_lab+'.'+date_form[0]+'*Z'+'.'+str(wavelen)+'.image_lev1.fits'
        
        pattern = file_path+file_name
    
        if any(os.path.isfile(file) for file in glob.glob(pattern)):
            print("File exists!")
            files = glob.glob(pattern)
            """If it exists locally already, use it."""        
        else:
            print("File does not exist!")
            res = Fido.search(a.Time(t0d, tfd),a.jsoc.Series(aia_lab), a.Wavelength(wavelen*u.AA),a.jsoc.Notify(jsoc_email)) 
            files = Fido.fetch(res[0,0], path=file_path)
            # breakpoint()
            files = glob.glob(pattern)
            """If it doesn't exist, find it and download it.""" 
        
        #------------------------------ Generate in Footpoint Plot ----------------------------------#
        # files = Fido.fetch(res[0,0], path=file_path)
        aia_map = sunpy.map.Map(files[0])
        lanes_map = sunpy.map.Map(lanes_savepath+lanes_savename)
    
        fig = plt.figure(figsize=(25,20))
        shape = (720, 1440)
        
        carr_header = make_heliographic_header(aia_map.date, aia_map.observer_coordinate, shape, frame='carrington')
        outmap = aia_map.reproject_to(carr_header)
        
        if plot_shape == 'flat':
            axs = plt.subplot(projection=outmap) 
            im = outmap.plot(clip_interval=(1, 99.99)*u.percent)
            
        elif plot_shape == 'circ':
        
            axs = plt.subplot(projection=aia_map)
            im = aia_map.plot(clip_interval=(1, 99.99)*u.percent)
            # aia_map.draw_grid(axes=axs)
            # pass
    
    
        if zoom:
            xlims_world = [-750, 0]*u.arcsec
            ylims_world = [0, 600]*u.arcsec
    
            world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=aia_map.coordinate_frame)
            pixel_coords = aia_map.world_to_pixel(world_coords)
        
            # we can then pull out the x and y values of these limits.
            xlims_pixel = pixel_coords.x.value
            ylims_pixel = pixel_coords.y.value
        else:
            xlims_world = [-1000, 1000]*u.arcsec
            ylims_world = [-700, 700]*u.arcsec
        
    
        sol_coords = SkyCoord(sol_lon*u.deg, sol_lat*u.deg, frame=outmap.coordinate_frame)
        # carr_coords = SkyCoord(carr_lon*u.deg, carr_lat*u.deg, frame=aia_map.coordinate_frame)
        q_coords = SkyCoord(region_foot_lon*u.deg, region_foot_lat*u.deg, frame=outmap.coordinate_frame)
        
        
        sol_p = axs.plot_coord(sol_coords, '.', color="lime",markersize=10,label='PFSS Footpoints')
        q_p = axs.plot_coord(q_coords, 'v', color="magenta",markersize=10,label='Quiescent Region Footpoints')
        
        if region_labels:
            for i in range(len(carr_lat_max)):
                # axs.annotate('memes', (carr_lat_max[i],carr_lon_max[i]))
                # print(i+2)
                # breakpoint()
                if regions_arr[i,2] > 1800:
                # if i == 7:
                    axs.text(carr_lon_min[i],carr_lat_max[i],str(i),color="black",fontsize=14,transform=axs.get_transform('world'))
                # else:
                    # axs.text(carr_lon_min[i],carr_lat_max[i],str(i),color="black",fontsize=14,transform=axs.get_transform('world'))
        
        if zoom:
            axs.set_title('Zoomed Plot of Active Region',fontsize=80)
            axs.set_ylim(ylims_pixel)
            axs.set_xlim(xlims_pixel)
        else:
            axs.set_title('PSP Footpoints on '+instr+' '+str(wavelen)+'Å Images, Encounter '+str(enc),fontsize=52)
        
        axs.set_ylabel("Helioprojective Lattitude (Solar-Y)",fontsize=60)
        axs.set_xlabel("Helioprojective Longitude (Solar-X)",fontsize=60)
    
        # axs.tick_params(axis='both', which='major', labelsize=22)
        axs.tick_params(axis='both', which='major', labelsize=60)
        # axs.tick_params(axis='y', labelrotation=90)
        
        handles, labels = axs.get_legend_handles_labels()
        handler_map = {type(sol_p): MarkerSizeHandler() for sol_p in handles}
        leg = axs.legend(handles, labels, handler_map=handler_map,fontsize=40,loc='lower right')
        
    
        leg.legend_handles[0].set_color('lime')
        # leg.legend_handles[1].set_color('blue')
        leg.legend_handles[1].set_color('magenta')
        # leg.legend_handles[2].set_color('magenta')
        # leg.legend_handles[3].set_color('magenta')
    
        plt.show()
        
        #-------------------------create supergranule overplot----------------------#
        
        zoom=True
        
        plt.figure(figsize=(25,20))
        shape = (720, 1440)
        
        lanes_map = sunpy.map.Map(lanes_savepath+lanes_savename)
        
        reprojected_map = lanes_map
        
        if plot_shape == 'flat':
            axs = plt.subplot(projection=lanes_map) 
            lanes_map.plot(clip_interval=(1, 99.99)*u.percent)
            
        elif plot_shape == 'circ':
        
            axs = plt.subplot(projection=reprojected_map)
            reprojected_map.plot(clip_interval=(1, 99.99)*u.percent,cmap='binary') #cmap='binary',
        
        
        # if zoom:
            # # xlims_world = [-92, 34]*u.arcsec
            # # ylims_world = [216, 340]*u.arcsec
            
            # xlims_world = [123, 129]*u.deg
            # ylims_world = [5, 13]*u.deg
    
            # world_coords = SkyCoord(Tx=xlims_world, Ty=ylims_world, frame=reprojected_map.coordinate_frame)
            # pixel_coords = reprojected_map.world_to_pixel(world_coords)
        
            # # we can then pull out the x and y values of these limits.
            # xlims_pixel = pixel_coords.x.value
            # ylims_pixel = pixel_coords.y.value

        sol_coords = SkyCoord(sol_lon*u.deg, sol_lat*u.deg, frame="heliographic_carrington",
                                obstime=reprojected_map.date,  # Use the observation time of the map
                                observer="earth")  # Match the observer of the map)

        q_coords = SkyCoord(region_foot_lon*u.deg, region_foot_lat*u.deg, frame="heliographic_carrington",
                                obstime=reprojected_map.date,  # Use the observation time of the map
                                observer="earth")  # Match the observer of the map)
        
        # sol_coords_hp = sol_coords.transform_to("helioprojective")
        # q_coords_hp = q_coords.transform_to("helioprojective")
        
        
        # sol_p = axs.plot_coord(sol_coords_hp, 'o',mew=2, mfc='none', color="lime",markersize=20,label='Non-Quiescent Footpoints')
        # q_p = axs.plot_coord(q_coords_hp, 'o',mew=4, mfc='none', color="magenta",markersize=20,label='Quiescent Region Footpoints')
        


        axs.set_title('Supergranulation Lanes',fontsize=65)
        
        
        # Get the map's spatial extent in world coordinates
        lon_min, lon_max = reprojected_map.meta['crval1'] - reprojected_map.meta['cdelt1'] * reprojected_map.data.shape[1] / 2, \
                            reprojected_map.meta['crval1'] + reprojected_map.meta['cdelt1'] * reprojected_map.data.shape[1] / 2
        lat_min, lat_max = reprojected_map.meta['crval2'] - reprojected_map.meta['cdelt2'] * reprojected_map.data.shape[0] / 2, \
                            reprojected_map.meta['crval2'] + reprojected_map.meta['cdelt2'] * reprojected_map.data.shape[0] / 2

        # Define the bounds (already in decimal degrees)
        lon_min = lon_min * u.deg
        lon_max = lon_max * u.deg
        lat_min = lat_min * u.deg
        lat_max = lat_max * u.deg
        
        # Wrap longitudes around 0–360 degrees if necessary
        data_coords_lon = sol_coords.lon.wrap_at(360 * u.deg)
        q_data_coords_lon = q_coords.lon.wrap_at(360 * u.deg)
        
        # Create masks for the range
        lon_mask = (data_coords_lon >= lon_min) & (data_coords_lon <= lon_max)
        lat_mask = (sol_coords.lat >= lat_min) & (sol_coords.lat <= lat_max)
        
        q_lon_mask = (q_data_coords_lon >= lon_min) & (q_data_coords_lon <= lon_max)
        q_lat_mask = (q_coords.lat >= lat_min) & (q_coords.lat <= lat_max)
        
        # Combine the masks
        mask = lon_mask & lat_mask
        q_mask = q_lon_mask & q_lat_mask
        
        # Filter the SkyCoord object
        sol_filtered_coords = sol_coords[mask]
        q_filtered_coords = q_coords[q_mask]
        
        sol_p = axs.plot_coord(sol_filtered_coords, 'o',mew=2, mfc='none', color="lime",markersize=20,label='Non-Quiescent Footpoints')
        q_p = axs.plot_coord(q_filtered_coords, 'o',mew=4, mfc='none', color="magenta",markersize=20,label='Quiescent Region Footpoints')

        # # Set limits using WCSAxes world coordinates
        # axs.set_xlim([lon_min, lon_max])
        # axs.set_ylim([lat_min, lat_max])
            
        # axs.set_ylabel("Helioprojective Lattitude (Solar-Y)",fontsize=60)
        # axs.set_xlabel("Helioprojective Longitude (Solar-X)",fontsize=60)
        
        axs.set_ylabel("Carrington Latitude",fontsize=60)
        axs.set_xlabel("Carrington Longitude",fontsize=60)

        axs.tick_params(axis='both', which='major', labelsize=60)
        
        handles, labels = axs.get_legend_handles_labels()
        handler_map = {type(sol_p): MarkerSizeHandler() for sol_p in handles}
        leg = axs.legend(handles, labels, handler_map=handler_map,fontsize=40,loc='lower right')
        
    
        leg.legend_handles[0].set_color('lime')
        leg.legend_handles[1].set_color('magenta')
        
        plt.show()
       
def expansion_analysis(t0='2020-01-29',tf=None,enc=None,save_coords=False,plot=True,save=True,normal=False,log=False):
    
    #----------------------Choosing which Encounters to use-------------------#
    
    if enc == 'all':
        # enc = list(range(1,17))
        enc_list = list(range(1,19))
        
    if enc == 'no 13':
        enc_list = list(range(2,13))
        enc_list.extend(range(14,17))
        
    if enc == 'no 1':
        enc_list = list(range(2,17)) #shortcut to exlude encounter 1, we use SPC and it gets weird.
        
    if type(enc) is int:
        enc_list = [enc]
    
    if type(enc) is list:
        enc_list = enc
    
    if tf==None:
        tf = pys.time_string(pys.time_float(t0)+86400)
    
    exp_fact_fin_time = np.array([])
    exp_fact_fin = np.array([])
    
    q_exp_fact_fin_time = np.array([])
    q_exp_fact_fin = np.array([])
    
    nq_exp_fact_fin_time = np.array([])
    nq_exp_fact_fin = np.array([])
        
    tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/data_outputs/'
    
    
    if enc != None:    
        
    
        for enc in enc_list:
            
            #-------------------------- import tplot files ---------------------------#
            tplot_savename = 'Enc_'+str(enc)+'_sorted_PFSS_data.cdf'
            
            pyt.tplot_restore(tplot_savepath+tplot_savename)
            
            # breakpoint()
            
            rss_data = pyt.get_data('rss')
            rss = rss_data[0]
            rss = rss[0]
            
            exp_data = pyt.get_data('expansion_factor') #full data for expansion factors
            exp_time = exp_data[0]
            expansion_factor = exp_data[1]
            expansion_factor_fin_time = exp_time[np.isfinite(expansion_factor)]
            expansion_factor_finite = expansion_factor[np.isfinite(expansion_factor)]
            
            
            
            q_exp_data = pyt.get_data('q_expansion_factor') #quiescent expansion factors
            q_exp_time = q_exp_data[0]
            q_expansion_factor = q_exp_data[1]
            q_expansion_factor_fin_time = q_exp_time[np.isfinite(q_expansion_factor)]
            q_expansion_factor_finite = q_expansion_factor[np.isfinite(q_expansion_factor)]
            
            
            nq_exp_data = pyt.get_data('nq_expansion_factor') #non quiescent expansion factors
            nq_exp_time = nq_exp_data[0]
            nq_expansion_factor = nq_exp_data[1]
            
            nq_expansion_factor_fin_time = nq_exp_time[np.isfinite(nq_expansion_factor)]
            nq_expansion_factor_finite = nq_expansion_factor[np.isfinite(nq_expansion_factor)]
            
            #-----------------write to larger array----------------------#
            
            
            exp_fact_fin_time = np.append(exp_fact_fin_time,expansion_factor_fin_time)
            exp_fact_fin = np.append(exp_fact_fin,expansion_factor_finite)
            
            q_exp_fact_fin_time = np.append(q_exp_fact_fin_time,q_expansion_factor_fin_time)
            q_exp_fact_fin = np.append(q_exp_fact_fin,q_expansion_factor_finite)
            
            nq_exp_fact_fin_time = np.append(nq_exp_fact_fin_time,nq_expansion_factor_fin_time)
            nq_exp_fact_fin = np.append(nq_exp_fact_fin,nq_expansion_factor_finite)
            
            print(q_exp_fact_fin.shape)
            
            
    else:
        tplot_savename = t0+'_'+tf+'_sorted_PFSS_data.cdf'
        print('yeppers')
        #-------------------------- import tplot files ---------------------------#
        
        pyt.tplot_restore(tplot_savepath+tplot_savename)
        
        rss_data = pyt.get_data('rss')
        rss = rss_data[0]
        rss = rss[0]
        
        exp_data = pyt.get_data('expansion_factor') #full data for expansion factors
        exp_time = exp_data[0]
        expansion_factor = exp_data[1]
        exp_fact_fin_time = exp_time[np.isfinite(expansion_factor)]
        exp_fact_fin = expansion_factor[np.isfinite(expansion_factor)]
        
        q_exp_data = pyt.get_data('q_expansion_factor') #quiescent expansion factors
        q_exp_time = q_exp_data[0]
        q_expansion_factor = q_exp_data[1]
        q_exp_fact_fin_time = q_exp_time[np.isfinite(expansion_factor)]
        q_exp_fact_fin = q_expansion_factor[np.isfinite(q_expansion_factor)]
        
        
        nq_exp_data = pyt.get_data('nq_expansion_factor') #non quiescent expansion factors
        nq_exp_time = nq_exp_data[0]
        nq_expansion_factor = nq_exp_data[1]
        
        nq_exp_fact_fin_time = q_exp_time[np.isfinite(expansion_factor)]
        nq_exp_fact_fin = nq_expansion_factor[np.isfinite(nq_expansion_factor)]
    
    #--------------------- create histogram -------------------------#
    
    fig = plt.figure(figsize=(12,7))
    axs = fig.add_subplot(111)
    bin_num=30
    # drange = (0,3000)
    drange = (0,300)

    
    bins = np.linspace(drange[0],drange[1], bin_num)
    hist1, _ = np.histogram(exp_fact_fin, bins=bins)
    hist2, _ = np.histogram(q_exp_fact_fin, bins=bins)


    if normal:    
        hist1_norm = hist1 / np.sum(hist1)
        hist2_norm = hist2 / np.sum(hist2)
        
        # Calculate error bars for each bin (normalized)
        error1 = np.sqrt(hist1) / np.sum(hist1)
        error2 = np.sqrt(hist2) / np.sum(hist2)
        y_lab = 'Probability Density'
    else:
        hist1_norm = hist1
        hist2_norm = hist2
        # Calculate error bars for each bin (normalized)
        error1 = np.sqrt(hist1)
        error2 = np.sqrt(hist2)
        y_lab = 'Field Line Counts'
    
    if log:
        axs.set_yscale("log")
        y_lab = y_lab+' (Log Scale)'
    # Plot histograms with error bars
    axs.bar(bins[:-1], hist1_norm, width=np.diff(bins), align='center', alpha=0.5, label='Non-Quiescent Wind',edgecolor='black')
    axs.bar(bins[:-1], hist2_norm, width=np.diff(bins), align='center', alpha=0.5,label='Quiescent Region Wind',edgecolor='black')
    axs.errorbar(bins[:-1], hist1_norm, yerr=error1, fmt='none', color='k', capsize=3)
    axs.errorbar(bins[:-1], hist2_norm, yerr=error2, fmt='none', color='k', capsize=3)
    
    axs.set_ylabel(y_lab,fontsize=15)
    axs.set_xlabel("PFSS Expansion Factor, $f_{ss}$",fontsize=15)
    axs.set_title("PFSS Expansion Factor Histograms, Encounter 11",fontsize=20)
    

    
    plt.legend(loc='upper right')
    
    plt.show()

def find_data_for_supergranule(t0='2020-01-29',tf=None,enc=None,save_coords=False,plot=False,save=False,
                               rlim=60,good_angle=60,wavelen=171,plot_shape='circ',inst='aia',time_corr='parker',V_err=0.04):
    
    if enc is not None:
        
        t0,tf = utils.encounter_dates(enc,rlim)
    
    
    if tf==None:
        tf = pys.time_string(pys.time_float(t0)+86400)
    
    def set_axes_lims(ax):
        ax.set_xlim(0, 360)
        ax.set_ylim(0, 180)
        
    if enc == None:
        enc = utils.encounter_check(t0)
        
    #------------------ find encounter footpoint locations and convert them to stonyhurst ------------#
    # breakpoint()
    
    tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/data_outputs/'
    
    tplot_savename = 'Enc_'+str(enc)+'_sorted_PFSS_data.cdf'
    pyt.tplot_restore(tplot_savepath+tplot_savename)
    
    # breakpoint()
    
    PSP_pos_data = pyt.get_data('PSP_Rs')
    PSP_Rs = PSP_pos_data[1]*u.solRad
    
    rss_data = pyt.get_data('rss')
    rss = rss_data[0][0]
    
    Vsw_Rs_data = pyt.get_data('Vsw_Rs')
    Vsw_Rs = Vsw_Rs_data[1]*u.solRad/u.s
    
    Tp_data = pyt.get_data('Temp_p')
    Tp_time = Tp_data[0]
    Tp = Tp_data*u.eV
    
    lon_data = pyt.get_data('solar_lon')
    lon_time = lon_data[0]
    carrington_foot_lon = lon_data[1]
    
    lat_data = pyt.get_data('solar_lat')
    lat_time = lat_data[0]
    carrington_foot_lat = lat_data[1]
    
    nq_lon_data = pyt.get_data('nq_solar_lon') #non_quiescent data for footpoint longitude
    nq_carrington_foot_lon = nq_lon_data[1]
    
    nq_lat_data = pyt.get_data('nq_solar_lat') #non_quiescent data for footpoint latittude
    nq_lat_time = nq_lat_data[0]
    nq_carrington_foot_lat = nq_lat_data[1]
    
    q_lon_data = pyt.get_data('q_solar_lon') #non_quiescent data for footpoint longitude
    q_carrington_foot_lon = q_lon_data[1]
    
    q_lat_data = pyt.get_data('q_solar_lat') #non_quiescent data for footpoint latittude
    q_lat_time = q_lat_data[0]
    q_carrington_foot_lat = q_lat_data[1]
    
    dates = pys.time_string(lat_time)
    times = Time(dates)
    
    nq_dates = pys.time_string(nq_lat_time)
    nq_times = Time(nq_dates)
    
    q_dates = pys.time_string(q_lat_time)
    q_times = Time(q_dates)
    # breakpoint()
    # Define observer (using Earth's location for simplicity)
    earth_position = get_body_heliographic_stonyhurst('earth', times)
    nq_earth_position = get_body_heliographic_stonyhurst('earth', nq_times)
    q_earth_position = get_body_heliographic_stonyhurst('earth', q_times)
    
    # Create SkyCoord in Carrington frame
    carrington_coords = SkyCoord(lon=carrington_foot_lon*u.deg, lat=carrington_foot_lat*u.deg, 
                                 radius=1*u.Rsun,  # Assuming 1 AU for simplicity
                                 frame=frames.HeliographicCarrington(obstime=times, observer=earth_position))
    
    nq_carrington_coords = SkyCoord(lon=nq_carrington_foot_lon*u.deg, lat=nq_carrington_foot_lat*u.deg, 
                                 radius=1*u.Rsun,  # Assuming 1 AU for simplicity
                                 frame=frames.HeliographicCarrington(obstime=nq_times, observer=nq_earth_position))
    
    # breakpoint()
    
    q_carrington_coords = SkyCoord(lon=q_carrington_foot_lon*u.deg, lat=q_carrington_foot_lat*u.deg, 
                                 radius=1*u.Rsun,  # Assuming 1 AU for simplicity
                                 frame=frames.HeliographicCarrington(obstime=q_times, observer=q_earth_position))
    
    # Transform to Heliographic Stonyhurst
    stonyhurst_coords = carrington_coords.transform_to(frames.HeliographicStonyhurst)
    nq_stonyhurst_coords = nq_carrington_coords.transform_to(frames.HeliographicStonyhurst)
    q_stonyhurst_coords = q_carrington_coords.transform_to(frames.HeliographicStonyhurst)
    
    # Extract the new coordinates
    stonyhurst_lon = stonyhurst_coords.lon
    stonyhurst_lat = stonyhurst_coords.lat
    
    # Extract the new coordinates
    nq_stonyhurst_lon = nq_stonyhurst_coords.lon
    nq_stonyhurst_lat = nq_stonyhurst_coords.lat
    
    # Extract the new coordinates
    q_stonyhurst_lon = q_stonyhurst_coords.lon
    q_stonyhurst_lat = q_stonyhurst_coords.lat
    
    # Convert to SkyCoord in Stonyhurst frame
    coords = SkyCoord(lon=stonyhurst_lon, lat=stonyhurst_lat,radius=1*u.Rsun, frame='heliographic_stonyhurst')
    nq_coords = SkyCoord(lon=nq_stonyhurst_lon, lat=nq_stonyhurst_lat,radius=1*u.Rsun, frame='heliographic_stonyhurst')
    q_coords = SkyCoord(lon=q_stonyhurst_lon, lat=q_stonyhurst_lat,radius=1*u.Rsun, frame='heliographic_stonyhurst')
    
    # Define the origin (0°, 0°) in the Stonyhurst frame
    origin = SkyCoord(lon=0*u.deg, lat=0*u.deg, frame='heliographic_stonyhurst')
    
    # Define your cone half-angle
    cone_half_angle = good_angle * u.deg  # Example: 60 degrees
    
    # Calculate angular separation from the origin for each coordinate
    separations = coords.separation(origin)
    nq_separations = nq_coords.separation(origin)
    q_separations = q_coords.separation(origin)
    
    # Check which points lie within the cone
    in_cone = separations <= cone_half_angle
    nq_in_cone = nq_separations <= cone_half_angle
    q_in_cone = q_separations <= cone_half_angle
    
    good_where = np.where(in_cone)
    good_where = good_where[0]
    
    good_times = lat_time[good_where]
    good_lat = carrington_foot_lat[good_where]
    good_lon = carrington_foot_lon[good_where]
    
    nq_good_where = np.where(nq_in_cone)
    nq_good_where = nq_good_where[0]
    
    nq_good_times = nq_lat_time[nq_good_where]
    nq_good_lat = nq_carrington_foot_lat[nq_good_where]
    nq_good_lon = nq_carrington_foot_lon[nq_good_where]
    
    q_good_where = np.where(q_in_cone)
    q_good_where = q_good_where[0]
    
    q_good_times = q_lat_time[q_good_where]
    q_good_lat = q_carrington_foot_lat[q_good_where]
    q_good_lon = q_carrington_foot_lon[q_good_where]
    
    # breakpoint()
    
    if time_corr == 'const':
        
        Vsw_Rs[np.isnan(Vsw_Rs)] = np.nanmean(Vsw_Rs)
        
        time_of_flight = (PSP_Rs.value-rss)/Vsw_Rs.value
        sdo_obs_time = lat_time - time_of_flight
        
    if time_corr == 'parker':
        
        """Need to select T0 by velocity, but for now just pick one"""
        
        Rgrid = np.linspace(1,215,2000)*u.R_sun
        T0_arr = np.linspace(1*10**6,5*10**6,5)*u.K
        sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_isothermal(Rgrid,T0_arr[0])

        Vsw_kms = Vsw_Rs.to(u.km/u.s)
        
        r_obs = PSP_Rs
        v_obs = Vsw_kms
        v_max_err = Vsw_kms*(1+V_err)
        v_min_err = Vsw_kms*(1-V_err)
        # Fit a T0 for each observation
        sys.path.append('/Users/besh2109/GitHub/psp_python')
        # T0_fitted_slow = fit_T0(r_obs, v_obs)
        
        fit_T0_savename = 'Enc_'+str(enc)+'_fit_T0_data.cdf'
        fit_T0_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/fit_T0/'
        
        if os.path.exists(fit_T0_savepath+fit_T0_savename):
            print(f"The file {fit_T0_savename} exists.")
            pyt.tplot_restore(fit_T0_savepath+fit_T0_savename)
            
            T0_data = pyt.get_data('T0_coronal_temp_fit')
            T0_fitted = T0_data[1]
            
            T0_max_data = pyt.get_data('T0_coronal_temp_fit_max')
            T0_max_fitted = T0_max_data[1]
            
            T0_min_data = pyt.get_data('T0_coronal_temp_fit_min')
            T0_min_fitted = T0_min_data[1]
            
        else:
            print(f"The file {fit_T0_savename} does not exist.")
            
            T0_fitted = utils.fit_T0_parallel(r_obs, v_obs)
            pyt.store_data("T0_coronal_temp_fit", data={'x':lat_time, 'y':T0_fitted})
            
            T0_max_fitted = utils.fit_T0_parallel(r_obs, v_max_err)
            pyt.store_data("T0_coronal_temp_fit_max", data={'x':lat_time, 'y':T0_max_fitted})
            
            T0_min_fitted = utils.fit_T0_parallel(r_obs, v_min_err)
            pyt.store_data("T0_coronal_temp_fit_min", data={'x':lat_time, 'y':T0_min_fitted})
            
            cdf_var = ["T0_coronal_temp_fit","T0_coronal_temp_fit_max","T0_coronal_temp_fit_min"]
            
            pyt.tplot_save(cdf_var,fit_T0_savepath+fit_T0_savename)
            
        
        #------------Assign a launch velocity to each temperature---------------#
        
        T0_min, T0_max = np.min(T0_fitted), 2     # Input range (original T0_fitted temperatures in MK)
        out_min, out_max = 20, 100  # Output velocities (in km/s, based on jet launch velocities from Kumar 2022)
        
        # Apply linear mapping
        launch_vel_values = out_min + (T0_fitted - T0_min) * (out_max - out_min) / (T0_max - T0_min)
        
        # Cap values at 100 (20) if they exceed the max (are below the minimum)
        launch_vel_values = np.clip(launch_vel_values, out_min, out_max)
        
        #------max error------#
        
        T0_min, T0_max = np.min(T0_max_fitted), 2     # Input range (original T0_fitted temperatures in MK)
        out_min, out_max = 20, 100  # Output velocities (in km/s, based on jet launch velocities from Kumar 2022)
        
        # Apply linear mapping
        launch_vel_values_max = out_min + (T0_max_fitted - T0_min) * (out_max - out_min) / (T0_max - T0_min)
        
        # Cap values at 100 (20) if they exceed the max (are below the minimum)
        launch_vel_values_max = np.clip(launch_vel_values_max, out_min, out_max)
        
        #------min error------#
        
        T0_min, T0_max = np.min(T0_min_fitted), 2     # Input range (original T0_fitted temperatures in MK)
        out_min, out_max = 20, 100  # Output velocities (in km/s, based on jet launch velocities from Kumar 2022)
        
        # Apply linear mapping
        launch_vel_values_min = out_min + (T0_min_fitted - T0_min) * (out_max - out_min) / (T0_max - T0_min)
        
        # Cap values at 100 (20) if they exceed the max (are below the minimum)
        launch_vel_values_min = np.clip(launch_vel_values_min, out_min, out_max)
        
        
        #-------------------- Velocities from jetlet study Kumar 2022 ------------------#
        
        time_of_flight = []
        time_of_flight_max = []
        time_of_flight_min = []
        for i in range(len(PSP_Rs)):

            Rgrid = np.linspace(1,70,2000)*u.R_sun
            T0 = T0_fitted[i]*u.MK
            sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_isothermal(Rgrid,T0)

            vel_limit = launch_vel_values[i]
            mod_sol_vel = np.array(sol_vel.value)
            
            vel_where = np.where(mod_sol_vel<vel_limit)
            mod_sol_vel[vel_where] = vel_limit

            sol_a = sol_pos.value>1
            sol_b = sol_pos.value<PSP_Rs[i].value
            sol_spread = np.logical_and(sol_a,sol_b)
            diff_rad_where = np.where(sol_spread)
            
            tof_pos = (sol_pos[diff_rad_where]).to(u.km).value
            tof_vel = mod_sol_vel[diff_rad_where]
            
            tof_pos_dif = np.diff(tof_pos)
            tof_vel_dif = tof_vel[:-1]

            tof = np.sum(tof_pos_dif/tof_vel_dif)
            time_of_flight.append(tof)
            
            #------max error------#
            
            Rgrid = np.linspace(1,70,2000)*u.R_sun
            T0 = T0_max_fitted[i]*u.MK
            sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_isothermal(Rgrid,T0)

            vel_limit = launch_vel_values_max[i]
            mod_sol_vel = np.array(sol_vel.value)
            
            vel_where = np.where(mod_sol_vel<vel_limit)
            mod_sol_vel[vel_where] = vel_limit

            sol_a = sol_pos.value>1
            sol_b = sol_pos.value<PSP_Rs[i].value
            sol_spread = np.logical_and(sol_a,sol_b)
            diff_rad_where = np.where(sol_spread)
            
            tof_pos = (sol_pos[diff_rad_where]).to(u.km).value
            tof_vel = mod_sol_vel[diff_rad_where]
            
            tof_pos_dif = np.diff(tof_pos)
            tof_vel_dif = tof_vel[:-1]

            tof_max = np.sum(tof_pos_dif/tof_vel_dif)
            time_of_flight_max.append(tof_max)
            
            #------min error------#
            
            Rgrid = np.linspace(1,70,2000)*u.R_sun
            T0 = T0_min_fitted[i]*u.MK
            sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_isothermal(Rgrid,T0)

            vel_limit = launch_vel_values_min[i]
            mod_sol_vel = np.array(sol_vel.value)
            
            vel_where = np.where(mod_sol_vel<vel_limit)
            mod_sol_vel[vel_where] = vel_limit

            sol_a = sol_pos.value>1
            sol_b = sol_pos.value<PSP_Rs[i].value
            sol_spread = np.logical_and(sol_a,sol_b)
            diff_rad_where = np.where(sol_spread)
            
            tof_pos = (sol_pos[diff_rad_where]).to(u.km).value
            tof_vel = mod_sol_vel[diff_rad_where]
            
            tof_pos_dif = np.diff(tof_pos)
            tof_vel_dif = tof_vel[:-1]

            tof_min = np.sum(tof_pos_dif/tof_vel_dif)
            time_of_flight_min.append(tof_min)
            
            
        time_of_flight = np.array(time_of_flight)
        sdo_obs_time = lat_time - time_of_flight
        
        time_of_flight_max_err = np.array(time_of_flight_max)
        sdo_obs_time_max_err = lat_time - time_of_flight_max_err #probably going to be a sooner arrival, because max V err means faster wind and less travel time
        
        time_of_flight_min_err = np.array(time_of_flight_min)
        sdo_obs_time_min_err = lat_time - time_of_flight_min_err #probably going to be a later arrival, because min V err means slower wind and more travel time
        
        # breakpoint()
        
        # rangey = np.linspace(0,len(PSP_Rs)-1,20,dtype=int)

        # rangex = random.sample(range(0,len(PSP_Rs)-1), 20)
        
        # fig = plt.figure(figsize=(15,10))
        # ax = fig.add_subplot(111)

        # accel_curves = []
        # for i in rangex:
        #     Rgrid = np.linspace(1,70,100)*u.R_sun
        #     T0 = T0_fitted[i]*u.MK
        #     sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_isothermal(Rgrid,T0)
        #     ax.plot(sol_pos,sol_vel)
        #     accel_curves.append(sol_vel)
        #     ax.plot(PSP_Rs[i],Vsw_kms[i],color='red',label='PSP In-situ Velocity', marker='o', markersize=10)
            
        # accel_curves = np.vstack(accel_curves)
        
        # # ax.plot(PSP_Rs,Vsw_kms,color='tab:blue',label='PSP In-situ Velocity')
        # ax.set_xlim(0,70)
        # ax.set_ylabel('Solar Wind Velocity (km/s)',fontsize=15)
        # ax.set_xlabel('Radial Position (Rs)',fontsize=15)
        # ax.set_title('Parker Solution, Coronal Temperature fit to each dot',fontsize=18)
        # # plt.legend()
        # plt.show()
    
    sdo_good_time = sdo_obs_time[good_where]
    sdo_good_time_max_err = sdo_obs_time_max_err[good_where]
    sdo_good_time_min_err = sdo_obs_time_min_err[good_where]
    
    nq_where = np.where(np.isin(lat_time,nq_lat_time))
    nq_sdo_obs_time = sdo_obs_time[nq_where]
    nq_sdo_obs_time_max_err = sdo_obs_time_max_err[nq_where]
    nq_sdo_obs_time_min_err = sdo_obs_time_min_err[nq_where]
    
    nq_sdo_obs_good_time = nq_sdo_obs_time[nq_good_where]
    nq_sdo_obs_good_time_max_err = nq_sdo_obs_time_max_err[nq_good_where]
    nq_sdo_obs_good_time_min_err = nq_sdo_obs_time_min_err[nq_good_where]
    
    q_where = np.where(np.isin(lat_time,q_lat_time))
    q_sdo_obs_time = sdo_obs_time[q_where]
    q_sdo_obs_time_max_err = sdo_obs_time_max_err[q_where]
    q_sdo_obs_time_min_err = sdo_obs_time_min_err[q_where]
    
    q_sdo_obs_good_time = q_sdo_obs_time[q_good_where]
    q_sdo_obs_good_time_max_err = q_sdo_obs_time_max_err[q_good_where]
    q_sdo_obs_good_time_min_err = q_sdo_obs_time_min_err[q_good_where]
    
    # breakpoint()
        
    if save:
        
        pyt.del_data()
        
        save_name = 'Enc_'+str(enc)+'_good_observations.cdf'
        save_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/raphael_times/'
        
        #---------tplot saves---------#
        t_tplot_sol_lon = carrington_foot_lon
        t_tplot_sol_lat = carrington_foot_lat
        t_tplot_time = lon_time
        t_tplot_obs_time = sdo_obs_time
        t_tplot_obs_time_max_err = sdo_obs_time_max_err
        t_tplot_obs_time_min_err = sdo_obs_time_min_err
        tplot_r0_Rs = PSP_Rs
        tplot_rss = np.array([rss])
        
        pyt.store_data("solar_lon", data={'x':t_tplot_time, 'y':t_tplot_sol_lon}) #total solar longitude
        pyt.store_data("solar_lat", data={'x':t_tplot_time, 'y':t_tplot_sol_lat})
        pyt.store_data("sdo_obs_time",data={'psp_time':t_tplot_time,'y':t_tplot_obs_time})
        pyt.store_data("sdo_obs_time_max_err",data={'psp_time':t_tplot_time,'y':t_tplot_obs_time_max_err})
        pyt.store_data("sdo_obs_time_min_err",data={'psp_time':t_tplot_time,'y':t_tplot_obs_time_min_err})
        
        pyt.store_data("PSP_Rs", data={'x':t_tplot_time, 'y':tplot_r0_Rs})
        pyt.store_data("rss",data={'x':tplot_rss, 'y':tplot_rss})
        
        #--- quiescent ---#
        
        q_tplot_time = q_lat_time
        q_tplot_sol_lon = q_carrington_foot_lon
        q_tplot_sol_lat = q_carrington_foot_lat
        q_tplot_sdo_obs_time = q_sdo_obs_time
        q_tplot_sdo_obs_time_max_err = q_sdo_obs_time_max_err
        q_tplot_sdo_obs_time_min_err = q_sdo_obs_time_min_err
        
        pyt.store_data("q_solar_lon", data={'x':q_tplot_time, 'y':q_tplot_sol_lon}) #quiescent solar longitude
        pyt.store_data("q_solar_lat", data={'x':q_tplot_time, 'y':q_tplot_sol_lat})
        pyt.store_data("q_sdo_obs_time", data={'psp_time':q_tplot_time,'y':q_tplot_sdo_obs_time})
        pyt.store_data("q_sdo_obs_time_max_err", data={'psp_time':q_tplot_time,'y':q_tplot_sdo_obs_time_max_err})
        pyt.store_data("q_sdo_obs_time_min_err", data={'psp_time':q_tplot_time,'y':q_tplot_sdo_obs_time_min_err})
        
        #---non quiescent---#
        
        nq_tplot_time = nq_lat_time
        nq_tplot_sol_lon = nq_carrington_foot_lon
        nq_tplot_sol_lat = nq_carrington_foot_lat
        nq_tplot_sdo_obs_time = nq_sdo_obs_time
        nq_tplot_sdo_obs_time_max_err = nq_sdo_obs_time_max_err
        nq_tplot_sdo_obs_time_min_err = nq_sdo_obs_time_min_err
        
        pyt.store_data("nq_solar_lon", data={'x':nq_tplot_time, 'y':nq_tplot_sol_lon}) #non quiescent solar longitude
        pyt.store_data("nq_solar_lat", data={'x':nq_tplot_time, 'y':nq_tplot_sol_lat})
        pyt.store_data("nq_sdo_obs_time", data={'psp_time':nq_tplot_time,'y':nq_tplot_sdo_obs_time})
        pyt.store_data("nq_sdo_obs_time_max_err", data={'psp_time':nq_tplot_time,'y':nq_tplot_sdo_obs_time_max_err})
        pyt.store_data("nq_sdo_obs_time_min_err", data={'psp_time':nq_tplot_time,'y':nq_tplot_sdo_obs_time_min_err})
        
        #------good times------#
        
        t_tplot_good_sol_lon = good_lon
        t_tplot_good_sol_lat = good_lat
        t_tplot_good_time = good_times
        t_tplot_good_obs_time = sdo_good_time
        t_tplot_good_obs_time_max_err = sdo_good_time_max_err
        t_tplot_good_obs_time_min_err = sdo_good_time_min_err
        
        pyt.store_data("good_solar_lon", data={'x':t_tplot_good_time, 'y':t_tplot_good_sol_lon}) #total solar longitude
        pyt.store_data("good_solar_lat", data={'x':t_tplot_good_time, 'y':t_tplot_good_sol_lat})
        pyt.store_data("good_sdo_obs_time",data={'psp_time':t_tplot_good_time,'y':t_tplot_good_obs_time})
        pyt.store_data("good_sdo_obs_time_max_err",data={'psp_time':t_tplot_good_time,'y':t_tplot_good_obs_time_max_err})
        pyt.store_data("good_sdo_obs_time_min_err",data={'psp_time':t_tplot_good_time,'y':t_tplot_good_obs_time_min_err})
        
        #------quiescent good times------#
        
        q_tplot_good_time = q_good_times
        q_tplot_good_sol_lon = q_good_lon
        q_tplot_good_sol_lat = q_good_lat
        q_tplot_sdo_obs_good_time = q_sdo_obs_good_time
        q_tplot_sdo_obs_good_time_max_err = q_sdo_obs_good_time_max_err
        q_tplot_sdo_obs_good_time_min_err = q_sdo_obs_good_time_min_err
        
        pyt.store_data("q_good_solar_lon", data={'x':q_tplot_good_time, 'y':q_tplot_good_sol_lon}) #quiescent solar longitude
        pyt.store_data("q_good_solar_lat", data={'x':q_tplot_good_time, 'y':q_tplot_good_sol_lat})
        pyt.store_data("q_good_sdo_obs_time", data={'psp_time':q_tplot_good_time,'y':q_tplot_sdo_obs_good_time})
        pyt.store_data("q_good_sdo_obs_time_max_err", data={'psp_time':q_tplot_good_time,'y':q_tplot_sdo_obs_good_time_max_err})
        pyt.store_data("q_good_sdo_obs_time_min_err", data={'psp_time':q_tplot_good_time,'y':q_tplot_sdo_obs_good_time_min_err})
        
        #------nonquiescent good times-------#
        
        nq_tplot_good_time = nq_good_times
        nq_tplot_good_sol_lon = nq_good_lon
        nq_tplot_good_sol_lat = nq_good_lat
        nq_tplot_sdo_obs_good_time = nq_sdo_obs_good_time
        nq_tplot_sdo_obs_good_time_max_err = nq_sdo_obs_good_time_max_err
        nq_tplot_sdo_obs_good_time_min_err = nq_sdo_obs_good_time_min_err
        
        pyt.store_data("nq_good_solar_lon", data={'x':nq_tplot_good_time, 'y':nq_tplot_good_sol_lon}) #non quiescent solar longitude
        pyt.store_data("nq_good_solar_lat", data={'x':nq_tplot_good_time, 'y':nq_tplot_good_sol_lat})
        pyt.store_data("nq_good_sdo_obs_time", data={'psp_time':nq_tplot_good_time,'y':nq_tplot_sdo_obs_good_time})
        pyt.store_data("nq_good_sdo_obs_time_max_err", data={'psp_time':nq_tplot_good_time,'y':nq_tplot_sdo_obs_good_time_max_err})
        pyt.store_data("nq_good_sdo_obs_time_min_err", data={'psp_time':nq_tplot_good_time,'y':nq_tplot_sdo_obs_good_time_min_err})
        
        #-----coronal temperature-------#
        
        if time_corr == 'parker':
            tplot_T0 = np.array(T0_fitted)
            tplot_T0_max = np.array(T0_max_fitted)
            tplot_T0_min = np.array(T0_min_fitted)
        else:
            tplot_T0 = np.zeros(lat_time.shape)*np.nan
            tplot_T0_max = np.zeros(lat_time.shape)*np.nan
            tplot_T0_min = np.zeros(lat_time.shape)*np.nan
            
            
        attr_dict = {'units': 'MK'}
        
        pyt.store_data("T0_coronal_temp_fit",data={'x':t_tplot_time,'y':tplot_T0}, attr_dict=attr_dict)
        
        pyt.store_data("T0_coronal_temp_fit_max", data={'x':t_tplot_time, 'y':tplot_T0_max}, attr_dict=attr_dict)
        
        pyt.store_data("T0_coronal_temp_fit_min", data={'x':t_tplot_time, 'y':tplot_T0_min}, attr_dict=attr_dict)
        
        #----------------------#
    
        cdf_var_list = ["sdo_obs_time","sdo_obs_time_max_err","sdo_obs_time_min_err","solar_lon","solar_lat","PSP_Rs","rss","T0_coronal_temp_fit","T0_coronal_temp_fit_max","T0_coronal_temp_fit_min",
                        "q_solar_lon","q_solar_lat","q_sdo_obs_time","q_sdo_obs_time_max_err","q_sdo_obs_time_min_err","nq_solar_lon","nq_solar_lat","nq_sdo_obs_time","nq_sdo_obs_time_max_err",
                        "nq_sdo_obs_time_min_err","good_solar_lon","good_solar_lat","good_sdo_obs_time","good_sdo_obs_time_max_err","good_sdo_obs_time_min_err",
                        "q_good_solar_lon","q_good_solar_lat","q_good_sdo_obs_time","q_good_sdo_obs_time_max_err","q_good_sdo_obs_time_min_err",
                        "nq_good_solar_lon","nq_good_solar_lat","nq_good_sdo_obs_time","nq_good_sdo_obs_time_max_err","nq_good_sdo_obs_time_min_err"]
        
        pyt.tplot_save(cdf_var_list,save_path+save_name)

        #------------------- group good quiescent points into continuous regions --------------------------#
        
        if len(q_good_where>0):
    
            csv_save_name = 'Enc_'+str(enc)+'_good_regions.csv'
            csv_save_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/raphael_times/'
            
            q_bool = np.isin(good_times,q_good_times) #sorts quiescent and non-quiescent solar wind into 1s and 0s.
            
            #flip the bits in previous bool array, 1s are good, but next function groups zeros
            q_bool = 1-q_bool
            
            #now finds where quiescent times are good
            date_groups = utils.group_zeros(q_bool)
            
            q_starts = ['Region PSP Start']
            q_ends = ['PSP End']
            q_lon_start = ['Longitude Start']
            q_lon_end = ['Longitude End']
            q_lat_start = ['Latitude Start']
            q_lat_end = ['Latitude End']
            q_sdo_start = ['SDO Start']
            q_sdo_end = ['SDO End']
            q_sdo_start_max_err = ['SDO Start (max err)']
            q_sdo_end_max_err = ['SDO End (max err)']
            q_sdo_start_min_err = ['SDO Start (min err)']
            q_sdo_end_min_err = ['SDO End (min err)']
            
            for i in date_groups:
                # print(sdo.obstime[i[0]],sdo.obstime[i[1]])
                q_starts.append(pys.time_string(good_times[i[0]]))
                q_ends.append(pys.time_string(good_times[i[1]]))
                
                q_lon_start.append(good_lon[i[0]])
                q_lon_end.append(good_lon[i[1]])
                
                q_lat_start.append(good_lat[i[0]])
                q_lat_end.append(good_lat[i[1]])
                
                q_sdo_start.append(pys.time_string(sdo_good_time[i[0]]))
                q_sdo_end.append(pys.time_string(sdo_good_time[i[1]]))
                
                q_sdo_start_max_err.append(pys.time_string(sdo_good_time_max_err[i[0]]))
                q_sdo_end_max_err.append(pys.time_string(sdo_good_time_max_err[i[1]]))
                
                q_sdo_start_min_err.append(pys.time_string(sdo_good_time_min_err[i[0]]))
                q_sdo_end_min_err.append(pys.time_string(sdo_good_time_min_err[i[1]]))
                
            with open(csv_save_path+csv_save_name, mode='w', newline='') as file:
                writer = csv.writer(file)
                writer.writerows(zip(q_starts, q_ends,q_sdo_start,q_sdo_start_max_err,q_sdo_start_min_err,q_sdo_end,q_sdo_end_max_err,q_sdo_end_min_err,q_lat_start,q_lat_end,q_lon_start,q_lon_end))  # Write each pair as a row
        
        

    if plot and (len(nq_good_where)>0 or len(q_good_where)>0):
        
        check_ind = round(len(nq_good_times)/2)
        
        euv_chn_opts = [94, 131, 171, 193, 211, 304, 335]
        uv_chn_opts = [1600, 1700]
        vis_chn_opts = [4500]

        euv_aia = 'aia.lev1_euv_12s'
        uv_aia = 'aia.lev1_uv_24s'
        vis_aia = 'aia.lev1_vis_1h'
        
        if wavelen in euv_chn_opts:
            aia_lab = euv_aia
            sec = 12.
            myid = inst+'_0'+str(wavelen)
        elif wavelen in uv_chn_opts:
            aia_lab = uv_aia
            sec = 24.
            myid = inst+'_'+str(wavelen)
        elif wavelen in vis_chn_opts:
            aia_lab = vis_aia
            sec = 3600.
            myid = inst+'_'+str(wavelen)
        else:
            
            print("Channel option not valid, setting wavelength to 171A.")
            aia_lab = euv_aia
            wavelen=171
            sec = 12.
            myid = inst+'_0'+str(wavelen)
            
        t0d = pys.time_string(nq_good_times[check_ind])
        # t0d = date[0]
        # t0d = date[0][:10]
        tfd = pys.time_string(pys.time_float(t0d)+sec)
        
        # breakpoint()
        #------------------------------ Read in AIA Images ----------------------------------#
        
        """ Check to see if AIA file already exists. They are large files and the program runs slow if sunpy gets carried away."""
        
        date_form = pys.time_string(nq_good_times[check_ind],fmt='%Y-%m-%dT%H')
        
        file_path = os.environ.get('SUNPY_DATA_DIR')+'/AIA/'+str(wavelen)+'/'+t0d[:4]+'/'+t0d[5:7]+'/'+t0d[8:10]+'/'
        file_name = aia_lab+'.'+date_form[0]+'*Z'+'.'+str(wavelen)+'.image_lev1.fits'
        
        pattern = file_path+file_name
    
        if any(os.path.isfile(file) for file in glob.glob(pattern)):
            print("File exists!")
            files = glob.glob(pattern)
            """If it exists locally already, use it."""        
        else:
            print("File does not exist!")
            res = Fido.search(a.Time(t0d, tfd),a.jsoc.Series(aia_lab), a.Wavelength(wavelen*u.AA),a.jsoc.Notify(jsoc_email)) 
            files = Fido.fetch(res[0,0], path=file_path)
            """If it doesn't exist, find it and download it.""" 
        
        # breakpoint()
        
        #------------------------------ Generate in Footpoint Plot ----------------------------------#
                
        aia_map = sunpy.map.Map(files[0])
    
        fig = plt.figure(figsize=(25,20))
        shape = (720, 1440)
        
        carr_header = make_heliographic_header(aia_map.date, aia_map.observer_coordinate, shape, frame='carrington')
        outmap = aia_map.reproject_to(carr_header)
        
        if plot_shape == 'flat':
            axs = plt.subplot(projection=outmap) 
            im = outmap.plot(clip_interval=(1, 99.99)*u.percent)
            
        elif plot_shape == 'circ':
        
            axs = plt.subplot(projection=aia_map)
            im = aia_map.plot(clip_interval=(1, 99.99)*u.percent)
            # aia_map.draw_grid(axes=axs)
            # pass
    
        xlims_world = [-1000, 1000]*u.arcsec
        ylims_world = [-700, 700]*u.arcsec

        sol_coords = SkyCoord(nq_good_lon*u.deg, nq_good_lat*u.deg, frame=outmap.coordinate_frame)
        q_sol_coords = SkyCoord(q_good_lon*u.deg, q_good_lat*u.deg, frame=outmap.coordinate_frame)
        
        sol_p = axs.plot_coord(sol_coords, '.', color="lime",markersize=10,label='PFSS Footpoints')
        q_sol_p = axs.plot_coord(q_sol_coords, '.', color="magenta",markersize=10,label='PFSS Quiescent Footpoints')


        # Generate points on the circle where the cone intersects the solar surface
        # Here, we treat the Sun's surface as having a radius of 1 solar radius for simplicity
        num_points = 500  # Number of points to plot the circle
        angles = np.linspace(0, 2*np.pi, num_points)
        
        # Convert cone_half_angle to radians for consistency
        cone_half_angle_rad = cone_half_angle.to(u.rad)
        
        lon_circle_rad = cone_half_angle_rad * np.cos(angles)
        lat_circle_rad = cone_half_angle_rad * np.sin(angles)
        
        # Convert back to degrees for SkyCoo
        lon_circle_deg = lon_circle_rad.to(u.deg)
        lat_circle_deg = lat_circle_rad.to(u.deg)
        
        # Create SkyCoord for these points in HeliographicStonyhurst
        circle_points_hgs = SkyCoord(lon=lon_circle_deg, lat=lat_circle_deg, 
                                     radius=1*u.Rsun,  # Assuming 1 solar radius for the surface
                                     frame=frames.HeliographicStonyhurst)

        circ_p = axs.plot_coord(circle_points_hgs, '.', color="blue",markersize=10,label='Supergranule code window')
        
        # breakpoint()
        
        # # n = [58, 651, 393, 203, 123]
        
        # if region_labels:
        #     for i in range(len(carr_lat_max)):
        #         # axs.annotate('memes', (carr_lat_max[i],carr_lon_max[i]))
        #         # print(i+2)
        #         # breakpoint()
        #         if regions_arr[i,2] > 1800:
        #         # if i == 7:
        #             axs.text(carr_lon_min[i],carr_lat_max[i],str(i),color="black",fontsize=14,transform=axs.get_transform('world'))
        #         # else:
        #             # axs.text(carr_lon_min[i],carr_lat_max[i],str(i),color="black",fontsize=14,transform=axs.get_transform('world'))
        

        axs.set_title('PSP Footpoints as seen by AIA '+str(wavelen)+'Å, Encounter '+str(enc),fontsize=52)
        
        axs.set_ylabel("Helioprojective Lattitude (Solar-Y)",fontsize=60)
        axs.set_xlabel("Helioprojective Longitude (Solar-X)",fontsize=60)
    
        axs.tick_params(axis='both', which='major', labelsize=60)
        # axs.tick_params(axis='y', labelrotation=90)
        
        handles, labels = axs.get_legend_handles_labels()
        handler_map = {type(sol_p): MarkerSizeHandler() for sol_p in handles}
        leg = axs.legend(handles, labels, handler_map=handler_map,fontsize=40,loc='lower right')
        
    
        leg.legend_handles[0].set_color('lime')
        # leg.legend_handles[1].set_color('blue')
        leg.legend_handles[1].set_color('magenta')
        # leg.legend_handles[2].set_color('magenta')
        # leg.legend_handles[3].set_color('magenta')
    
        plt.show()

def quiescent_plots(t0='2020-01-29',tf=None,enc=None,enc_radius=65,save_coords=False,plot=True,overview=False,save=False):
    
    Rs_km = 6.957e5 #solar radius in km  
    w = 360/(25.38*86400) # angular frequency of the sun in degrees/sec
    # w = 2*np.pi/(25.38*86400) # angular frequency of the sun in radians/sec

    if tf==None:
        tf = pys.time_string(pys.time_float(t0)+86400)

    
    fieldline_savepath = "/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/"
    if enc != None:    
        fieldline_savename = 'Enc_'+str(enc)+'_field_lines.pkl'
    else:
        fieldline_savename = t0+'_'+tf+'_field_lines.pkl'

    field_lines = utils.loadall(fieldline_savepath+fieldline_savename)
    with open(fieldline_savepath+fieldline_savename, 'rb') as file:
        # field_lines = pkl.load(file)
        field_lines = cpkl.load(file)

    field_lines = list(field_lines)

    if enc != None:    
        tplot_savename = 'Enc_'+str(enc)+'_footpoint_coords.cdf'
    else:
        tplot_savename = t0+'_'+tf+'_footpoint_coords.cdf'
        
    tplot_savepath = '/Users/besh2109/Desktop/Quiescent Region Connectivity/pfss_outs/'

    pyt.tplot_restore(tplot_savepath+tplot_savename)

    rss_data = pyt.get_data('rss')
    rss = rss_data[0]
    rss = rss[0]
    
    carr_lon_data = pyt.get_data('PSP_lon')
    carr_lon_time = carr_lon_data[0]
    carr_lon = carr_lon_data[1]
    
    carr_lat_data = pyt.get_data('PSP_lat')
    carr_lat = carr_lat_data[1]
    
    sintheta = np.sin((90-carr_lat)*np.pi/180) #sin of the azimuthal angle, which is 90 degrees minus the latitude
    
    r0_Rs_data = pyt.get_data('PSP_Rs')
    r0_Rs = r0_Rs_data[1]
    
    Vsw_Rs_data = pyt.get_data('Vsw_Rs')
    Vsw_Rs = Vsw_Rs_data[1]
    
    
    t0p = pys.time_string(carr_lon_time[0])
    tfp = pys.time_string(carr_lon_time[-1])

    
    psp.fields(trange=[t0p,tfp], datatype='ephem_spp_hg', level='l1',username=fields_id,password=fields_pass,last_version=True) #going to be used to plot parker position
    pos_data = pyt.get_data('position')
    
    pos_data_arr = pos_data[1]
    
    x = pos_data_arr[:,0]
    y = pos_data_arr[:,1]

    #-----------------------------PLOTTING ROUTINES---------------------------#
        
    if plot:
    
        #----------------# PFSS FIELD MODEL FIGURE #----------------#
        
        fig = plt.figure(figsize=(17.5,7))
        ax1 = fig.add_subplot(121, projection='3d')
        
        n_ind = list(range(len(carr_lon)))
        n_lines = 25
        
        idx = np.round(np.linspace(0, max(n_ind) - 1, n_lines)).astype(int) # number of lines
        
        j, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
        x1 = np.cos(j)*np.sin(v)
        y1 = np.sin(j)*np.sin(v)
        z1 = np.cos(v)
        ax1.plot_wireframe(x1, y1, z1, color="w", edgecolor="gray")
    
        ax2 = fig.add_subplot(122)
    
        sub_ind = 0
    
        for i in field_lines:

            field_lines_2_plot = i
            
            for field_line in field_lines_2_plot:
                    
                if sub_ind in idx:
                    color = {0: 'black', -1: 'tab:blue', 1: 'tab:red'}.get(field_line.polarity)
                    coords = field_line.coords
                    coords.representation_type = 'cartesian'
                    ax1.plot(coords.x / const.R_sun,
                            coords.y / const.R_sun,
                            coords.z / const.R_sun,
                            color=color, linewidth=1)

                    #-----------------# PARKER SPIRAL FIGURE #------------------#
                    
                    # ib = idx[j]
                    r = np.linspace(rss,r0_Rs[sub_ind],100)
                    spiral_plot_lon = carr_lon[sub_ind] - (w*sintheta[sub_ind]/Vsw_Rs[sub_ind])*(r-r0_Rs[sub_ind])
                    # spiral_plot_lon = carr_lon_psp[ib] - (w*sintheta[ib]/Vsw_Rs)*(r-r0_Rs[ib])
                    
                    spiral_x = r*np.cos(spiral_plot_lon*np.pi/180)
                    spiral_y = r*np.sin(spiral_plot_lon*np.pi/180)
            
                    # if field_lines_2_plot[sub_ind].polarity == 1:
                    if field_line.polarity == 1:
                        color = 'tab:red'
                    else:
                        color = 'tab:blue'
            
                    ax2.plot(spiral_x,spiral_y,color=color)
                
                sub_ind+=1
                
        ax1.view_init(40, 280)
        ax1.set_title(r"PFSS solution with Source Surface at "+str(rss)+" Rs", fontsize=20)
        
        ax1.text2D(0.5,-0.08,"using GONG Synoptic Map",fontsize=20,transform=ax1.transAxes,horizontalalignment='center')
        

        utils.set_axes_equal(ax1)
        
        # Make panes transparent
        ax1.xaxis.pane.fill = False # Left pane
        ax1.yaxis.pane.fill = False # Right pane
        
        # Remove grid lines
        ax1.grid(False)
        
        # Remove tick labels
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        ax1.set_zticklabels([])
        
        # # Transparent spines
        # ax1.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # ax1.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        # ax1.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # Transparent spines
        ax1.xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
        
        # # Transparent panes
        # ax1.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax1.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        # ax1.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Transparent panes
        ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # No ticks
        ax1.set_xticks([]) 
        ax1.set_yticks([]) 
        ax1.set_zticks([])
        
        #scale up axes
        
        # x_scale=4
        # y_scale=4
        # z_scale=4
        
        # scale=np.diag([x_scale, y_scale, z_scale, 1.0])
        # scale=scale*(1.0/scale.max())
        # scale[3,3]=1.0
        
        # def short_proj():
        #   return np.dot(Axes3D.get_proj(ax1), scale)
        
        # ax1.get_proj=short_proj
        
        ax1.set_xlim3d(-1.75, 1.75)
        ax1.set_ylim3d(-1.75, 1.75)
        ax1.set_zlim3d(-1.4, 1.4)
            
        ax2.set_title('Parker Spiral Propagation to PSP Position',fontsize=20)
        ax2.plot(x/Rs_km,y/Rs_km,color='black')
        ax2.set_ylim(-150,100)
        ax2.set_xlim(-150,150)
        ax2.set_xlabel('Carrington X in Rs',fontsize=16)
        ax2.set_ylabel('Carrington Y in Rs',fontsize=16)
        
        # We change the fontsize of minor ticks label 
        ax2.tick_params(axis='both', which='major', labelsize=16)
        # ax2.tick_params(axis='both', which='minor', labelsize=8)
        # ax.set_aspect("equal")
        plt.show()
    
    solo_launch_date_str = '2020-02-11 00:00'
    solo_launch_date = parse_time(solo_launch_date_str)
    
    today = datetime.today()
    #choose last week to make sure data is available,
    #fresher data may not have come down yet.
    last_week = today - timedelta(days=7) 
    
    end_date_str = last_week.strftime('%Y-%m-%d')
    end_date = parse_time(end_date_str)
    
    psp_coords = get_horizons_coord('Parker Solar Probe',
                              {'start': solo_launch_date,
                              'stop': end_date,
                              'step': '180m'},include_velocity=True)
    
    solo = get_horizons_coord('Solar Orbiter',
                              {'start': solo_launch_date,
                              'stop': end_date,
                              'step': '180m'})
    
    sdo = get_horizons_coord('Solar Dynamics Observatory',
                              {'start': solo_launch_date,
                              'stop': end_date,
                              'step': '180m'})

    yes_no_array = []
    for i in range(len(solo.obstime)):

        solo_coord = utils.coord_to_polar(solo[i])
        
        # set the bounds of when to use Solar Orbiter magnetograms
        # if (solo_coord[0] > np.pi/2) or (solo_coord[0] < -np.pi/2): #180 degree 'cone'
        if (solo_coord[0] > np.pi/3) or (solo_coord[0] < -np.pi/3): #checking an 120 degree cone
            yes_no_array.append(1)
        else:
            yes_no_array.append(0)
            
    yes_no_array = np.array(yes_no_array)
    
    #flip the bits in previous bool array, 1s are good, but next function groups zeros
    yes_no_array = 1-yes_no_array
    
    #now finds where solo times are good
    date_groups = utils.group_zeros(yes_no_array)
    
    far_orbit_where = np.where(1-yes_no_array)
    
    farside_orbit_starts = []
    farside_orbit_ends = []
    for i in date_groups:
        # print(sdo.obstime[i[0]],sdo.obstime[i[1]])
        farside_orbit_starts.append(sdo.obstime[i[0]])
        farside_orbit_ends.append(sdo.obstime[i[1]])
    
    fig = plt.figure(figsize=(8,8))
    ax = fig.add_subplot(projection='polar')
    ax.plot(0, 0, 'o', label='Sun', color='orange')
    ax.plot(*utils.coord_to_polar(sdo[0]), 'o', label='SDO', color='blue')
    # ax.plot(*coord_to_polar(solo),
    #         label='Solar Orbiter (as seen from SDO)', color='purple')
    ax.plot(*utils.coord_to_polar(solo[far_orbit_where]),
            label='Solar Orbiter (as seen from SDO)', color='green')

    ax.set_title('Relative Position of Solar Orbiter/PSP to SDO')
    ax.legend(loc='upper right')
    
    plt.show()
    
    for i in range(len(farside_orbit_starts)):
        
        if i == 1:
            data_dir ='~/sunpy/solo_data/'        
    
            # t_start_fdt = farside_orbit_starts[i]
            # t_end_fdt = farside_orbit_ends[i]
            t_start_fdt = parse_time('2022-07-25')
            t_end_fdt = parse_time('2022-07-26')
            results_phi_fdt = Fido.search(a.Instrument('PHI'), a.Time(t_start_fdt.value, t_end_fdt.value), a.soar.Product('phi-fdt-blos'),a.Level(2))
            files_phi_fdt = Fido.fetch(results_phi_fdt[0,5], path=data_dir)
        
        
            t_start_hmi = parse_time('2022-07-25')
            t_end_hmi = parse_time('2022-07-26')
            results_hmi = Fido.search(a.jsoc.Notify(os.environ["JSOC_EMAIL"]),a.jsoc.Time(t_start_hmi.value,t_end_hmi.value),a.jsoc.Series('hmi.mrdailysynframe_720s'))
            files_hmi = Fido.fetch(results_hmi, path=data_dir)
        
        # print(results_phi_fdt)
        
    # print(results_phi_fdt[0,350])
    # print(results_hmi)
    
    # breakpoint()
    
    # fdt_blos_map = pfss.utils.car_to_cea(fdt_blos_map)
    
    hmi_data, hmi_header = fits.getdata(files_hmi[0], header=True)
    hmi_header_new = utils.fix_hmi_meta(hmi_header)
    hmi_br_map = sunpy.map.Map(hmi_data,hmi_header_new)
    
    hmi_br_map.plot_settings['norm'].vmin = -100
    hmi_br_map.plot_settings['norm'].vmax = 100
    # hmi_br_map = hmi_br_map.rotate(recenter = True)
    hmi_br_map.peek()
    
    fdt_data, fdt_header = fits.getdata(files_phi_fdt[0], header=True)
    # fdt_header_new = fix_syn_map_units(fdt_header)
    
    # def build_checkerboard(w, h) :
    #       re = np.r_[ w*[0,1] ]              # even-numbered rows
    #       ro = np.r_[ w*[1,0] ]              # odd-numbered rows
    #       return np.row_stack(h*(re, ro))
    
    # checkerboard = build_checkerboard(384, 384)
    
    # fdt_data = fdt_data-1000*checkerboard
    
    fdt_blos_map = sunpy.map.Map(fdt_data,fdt_header)
    
    
    fdt_blos_map.plot_settings['norm'].vmin = -100
    fdt_blos_map.plot_settings['norm'].vmax = 100
    fdt_blos_map = fdt_blos_map.rotate(recenter = True)
    fdt_blos_map.peek()
    

    
    # shape = (720, 1440)
    shape = (1440, 3600)
    
    carr_header = make_heliographic_header(fdt_blos_map.date, fdt_blos_map.observer_coordinate, shape, frame='carrington')
    fdt_blos_map_repro = fdt_blos_map.reproject_to(carr_header)
    fdt_blos_map_repro.peek()
    
    # carr_header = make_heliographic_header(hmi_br_map.date, hmi_br_map.observer_coordinate, shape, frame='carrington')
    # hmi_br_map_repro = hmi_br_map.reproject_to(carr_header)
    # hmi_br_map_repro.peek()
    
    # maps = sunpy.map.Map([hmi_br_map,fdt_blos_map_repro])
    maps = sunpy.map.Map([fdt_blos_map_repro,hmi_br_map])
    # maps = sunpy.map.Map([fdt_blos_map,hmi_br_map])
    
    header = sunpy.map.make_fitswcs_header(shape,
                                           SkyCoord(0, 0, unit=u.deg,
                                                    frame="heliographic_stonyhurst",
                                                    obstime=maps[0].date),
                                           scale=[360 / shape[1],
                                                  180 / shape[0]] * u.deg / u.pix,
                                           projection_code="CAR")
    # out_wcs = WCS(header)
    out_wcs = maps[0].wcs
    
    # fdt_blos_map_repro = fdt_blos_map.reproject_to(header)
    # fdt_blos_map_repro.peek()
    
    array, footprint = reproject_and_coadd(maps, out_wcs, shape,
                                        reproject_function=reproject_interp)
    
    zero_where = np.where(array==0)
    # zero_where = zero_where[0]
    
    array[zero_where] = np.nan
    
    outmap = sunpy.map.Map((array, header))
    # outmap.peek()
    # outmap.plot_settings = maps[0].plot_settings
    
    fig = plt.figure(figsize=(20,10))
    ax = fig.add_subplot(projection=maps[0])
    ax.set_facecolor('m')
    
    outmap.plot_settings['norm'].vmin = -100
    outmap.plot_settings['norm'].vmax = 100
    outmap.plot(axes=ax,zorder=1)
    
    # maps[1].plot_settings['norm'].vmin = -100
    # maps[1].plot_settings['norm'].vmax = 100
    maps[1].plot(axes=ax,zorder=0)
    
    # fdt_blos_map_repro.plot(axes=ax)
    
    plt.show()
    
    
    coordinates = tuple(map(sunpy.map.all_coordinates_from_map, maps))
    weights = [coord.transform_to("heliocentric").z.value for coord in coordinates]
    
    weights = [(w / np.nanmax(w)) ** 3 for w in weights]
    for w in weights:
        w[np.isnan(w)] = 0
    
    fig, ax = plt.subplots()
    im = ax.imshow(weights[0])
    fig.colorbar(im)
    
    plt.show()
    
    
    
    array, _ = reproject_and_coadd(maps, out_wcs, shape,
                               input_weights=weights,
                               reproject_function=reproject_interp,
                               match_background=True,
                               background_reference=0)
    
    outmap = sunpy.map.Map((array, header))
    # outmap.plot_settings['norm'].vmin = -100
    # outmap.plot_settings['norm'].vmax = 100
    outmap.nickname = 'HMI+PHI Magnetogram'
    
    fig = plt.figure(figsize=(20, 10))
    ax = fig.add_subplot(projection=outmap)
    im = outmap.plot(axes=ax)
    
    lon, lat = ax.coords
    lon.set_coord_type("longitude")
    lon.coord_wrap = 180 * u.deg
    lon.set_format_unit(u.deg)
    lat.set_coord_type("latitude")
    lat.set_format_unit(u.deg)
    
    lon.set_axislabel('Heliographic Longitude', minpad=0.8)
    lat.set_axislabel('Heliographic Latitude', minpad=0.9)
    # lon.set_ticks(spacing=25*u.deg, color='k')
    # lat.set_ticks(spacing=15*u.deg, color='k')
    
    # plt.colorbar(im, ax=ax)
    
    # Reset the view to pixel centers
    _ = ax.axis((0, shape[1], 0, shape[0]))
    
    plt.show()
    
    if overview:
        
        # Start date
        start_date = datetime(2020, 5, 1)
        # breakpoint()
        # Number of months to loop through
        num_months = 12*4
        
        # Loop through the months
        dates = []
        for i in range(num_months):
            current_date = start_date + relativedelta(months=i)
            dates.append(current_date.strftime('%Y-%m-%d'))
            
        # Print the dates
        for date in dates:
            print(date)
        
            check_date = parse_time(date)
            
            solo_i_date = str(check_date - 20 * u.day)
            solo_f_date = str(check_date + 20 * u.day)
            psp_coords = get_horizons_coord('Parker Solar Probe',
                                      {'start': check_date - 20 * u.day,
                                      'stop': check_date + 20 * u.day,
                                      'step': '180m'})
            
            solo = get_horizons_coord('Solar Orbiter',
                                     {'start': check_date - 20 * u.day,
                                      'stop': check_date + 20 * u.day,
                                      'step': '180m'})
            
            sdo = get_horizons_coord('Solar Dynamics Observatory', check_date)
            
            # earth = get_body_heliographic_stonyhurst('Earth', check_date)
            
            data_dir ='~/sunpy/solo_data/'
            
            # t_start_hrt = Time('2022-03-07T00:00', format='isot', scale='utc')
            # t_end_hrt = Time('2022-03-07T00:01', format='isot', scale='utc')
        
            # results_phi_hrt = Fido.search(a.Instrument('PHI'), a.Time(t_start_hrt.value, t_end_hrt.value), a.soar.Product('phi-hrt-blos'))
            # files_phi_hrt = Fido.fetch(results_phi_hrt, path=data_dir)
            
            # hrt_blos_map = sunpy.map.Map(files_phi_hrt[0])
        
            # hrt_blos_map.plot_settings['norm'].vmin = -100
            # hrt_blos_map.plot_settings['norm'].vmax = 100
            # hrt_blos_map.peek()
            
            # breakpoint()
            
            # t_start_fdt = Time(solo_i_date, format='isot', scale='utc')
            # t_end_fdt = Time(solo_f_date, format='isot', scale='utc')
        
            
            # results_phi_fdt = Fido.search(a.Instrument('PHI'), a.Time(t_start_fdt.value, t_end_fdt.value), a.soar.Product('phi-fdt-blos'))
            # files_phi_fdt = Fido.fetch(results_phi_fdt, path=data_dir)
            
            # fdt_blos_map = sunpy.map.Map(files_phi_fdt[0])
        
            # fdt_blos_map.plot_settings['norm'].vmin = -100
            # fdt_blos_map.plot_settings['norm'].vmax = 100
            # fdt_blos_map = fdt_blos_map.rotate(recenter = True)
            # fdt_blos_map.peek()
            
            
            #-------------------try to plot orbits----------------#
            
            if plot:
            
                # results_hmi = Fido.search(a.Instrument('HMI'), a.Time(t_start_hrt.value, t_end_hrt.value)) #a.soar.Product('phi-hrt-blos')
                # files_hmi = Fido.fetch(results_hmi, path=data_dir)
                
    
                # maps = sunpy.map.Map(files_phi_fdt[0],files_hmi[1])
                
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(projection='polar')
                ax.plot(0, 0, 'o', label='Sun', color='orange')
                ax.plot(*utils.coord_to_polar(sdo), 'o', label='SDO', color='blue')
                ax.plot(*utils.coord_to_polar(solo),
                        label='Solar Orbiter (as seen from SDO)', color='purple')
                ax.plot(*utils.coord_to_polar(psp_coords),
                        label='PSP (as seen from SDO)', color='green')
                # ax.plot(*coord_to_polar(solo.transform_to(earth)),
                #         label='Solar Orbiter (non-rotating frame)', color='purple', linestyle='dashed')
                ax.set_title('Relative Position of Solar Orbiter/PSP to SDO '+date)
                ax.legend(loc='upper right')
                
                plt.show()
            
            #---------------------------------------------------#
            
            if save:
                
                save_path = '/Users/besh2109/Desktop/Quiescent Region Connectivity/Plots/Solo Orbits/'
                save_name = 'Solo_Orbit_'+date+'.png'
                
                fig = plt.figure(figsize=(8,8))
                ax = fig.add_subplot(projection='polar')
                ax.plot(0, 0, 'o', label='Sun', color='orange')
                ax.plot(*utils.coord_to_polar(sdo), 'o', label='SDO', color='blue')
                ax.plot(*utils.coord_to_polar(solo),
                        label='Solar Orbiter (as seen from SDO)', color='purple')
                ax.plot(*utils.coord_to_polar(psp_coords),
                        label='PSP (as seen from SDO)', color='green')
                # ax.plot(*coord_to_polar(solo.transform_to(earth)),
                #         label='Solar Orbiter (non-rotating frame)', color='purple', linestyle='dashed')
                ax.set_title('Relative Position of Solar Orbiter/PSP to SDO '+date)
                ax.legend(loc='upper right')
                
                plt.savefig(save_path+save_name)
                plt.clf()
                plt.cla()
                plt.close('all')
                plt.close(fig)
                
