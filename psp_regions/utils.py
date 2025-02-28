#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 28 14:47:09 2025

@author: besh2109
"""

from .config import CONFIG
from .config import enc_flt

import numpy as np

import os
import pyspedas as pys
import pytplot as pyt

import astropy.units as u

import sunpy.map
from sunpy.net import Fido, attrs as a
from sunpy.time import parse_time

from datetime import datetime, timedelta

import parkersolarwind as psw
from multiprocessing import Pool

from scipy.optimize import minimize
import sys

import _pickle as cpkl


def find_every_i_days(dates, reference_index,day_num=3):
    reference_date = dates[reference_index]
    # three_days = timedelta(days=day_num)

    # Find dates going forward
    forward_dates = [(i, date) for i, date in enumerate(dates[reference_index:], start=reference_index)
                     if (date - reference_date).days % day_num == 0]

    # Find dates going backward
    backward_dates = [(i, date) for i, date in enumerate(dates[:reference_index])
                      if (reference_date - date).days % day_num == 0]
    
    result = backward_dates + forward_dates
    # Combine results, maintaining original order
    dates_result = [dt[1] for dt in result]
    index_result = [i[0] for i in result]
    
    return dates_result,index_result

def checkKey(dic, key):
    if key in dic.keys():
        return True
    else:
        return False

def _observer_coord_meta(observer_coord):
    """
    Convert an observer coordinate into FITS metadata.
    """
    new_obs_frame = sunpy.coordinates.HeliographicStonyhurst(obstime=observer_coord.obstime)
    observer_coord = observer_coord.transform_to(new_obs_frame)

    new_meta = {}
    new_meta['HGLT_OBS'] = observer_coord.lat.to_value(u.deg)
    new_meta['HGLN_OBS'] = observer_coord.lon.to_value(u.deg)
    new_meta['DSUN_OBS'] = observer_coord.radius.to_value(u.km)
    # new_meta['HGLT_OBS'] = f"{observer_coord.lat.to_value(u.deg):.6f}"
    # new_meta['HGLN_OBS'] = f"{observer_coord.lon.to_value(u.deg):.6f}"
    # new_meta['DSUN_OBS'] = f"{observer_coord.radius.to_value(u.m):.6f}"

    return new_meta

def earth_obs_coord_meta(obstime):
    """
    Return metadata for an Earth obeserver coordinate.
    """
    return _observer_coord_meta(sunpy.coordinates.get_earth(obstime))

def fix_hmi_meta(header):

    if header['cunit1'] == 'Degree':
        header['cunit1'] = 'deg'

    if header['cunit2'] == 'Sine Latitude' or header['cunit2'] == 'sin(deg)':
        header['cunit2'] = 'deg'   
        header['cdelt2'] = 180 / np.pi * header['cdelt2']
        header['cdelt1'] = np.abs(header['cdelt1'])
    
    if header['bunit'] == 'Mx/cm^2':
        header['bunit'] = 'G'
    
    if checkKey(header,'DATE-OBS') is False:
        date_str = header['T_OBS']
        # Parse the original date string
        dt = datetime.strptime(date_str, "%Y.%m.%d_%H:%M:%S.%f_TAI")
        # Format the datetime object to the desired format
        formatted_date = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        header['DATE-OBS'] = formatted_date
    
    if checkKey(header,'HGLT_OBS') is False:
        # hmi_map.meta.update(pfss.map._earth_obs_coord_meta(hmi_map.meta['date-obs']))
        meta_update = earth_obs_coord_meta(header['DATE-OBS'])
        header['HGLT_OBS'] = meta_update['HGLT_OBS']
        header['HGLN_OBS'] = meta_update['HGLN_OBS']
        header['DSUN_OBS'] = meta_update['DSUN_OBS']
        
    header['DSUN_OBS'] = header['DSUN_OBS']*10**3 #convert this to meters, as written its in km.
    
    header['CRVAL1'] = 120 + sunpy.coordinates.sun.L0(time=header['DATE-OBS']).value
    header['CUNIT1'] = 'deg'
    
    return header

def cost_function(T0, r_obs, v_obs):
    """Computes the squared difference between observed and model velocities for a single r."""
    # parker_model = ParkerSolution(T0[0])  # Ensure T0 is treated as a scalar
    # print('gg')
    r_in = np.linspace(1,70,2000)*u.R_sun
    T0_in = T0[0]*u.MK
    # print(T0_in)
    # sys.stdout.flush()
    sol_pos,sol_dens,sol_vel,sol_T0,num = psw.solve_parker_isothermal(r_in,T0_in)
    # print('one')
    # breakpoint()
    # Find the index of the closest element
    closest_index = np.abs(sol_pos.value - r_obs.value).argmin()
    v_model = sol_vel[closest_index]
    # print('two')
    return (v_model - v_obs) ** 2  # Squared error

def fit_single_T0(args):
    r, v = args
    # result = minimize(cost_function, x0=[1.0], args=(r, v), method="Nelder-Mead", options={"maxiter": 20})
    result = minimize(cost_function, x0=[1.0], args=(r, v), method="L-BFGS-B", bounds=[(0.4, 4)], options={'maxiter': 100, 'gtol': 1e-5})
    return result.x[0]

def fit_T0_parallel(r_obs, v_obs):
    n = len(r_obs)
    
    with Pool() as pool:
        # Use imap for efficiency with a generator
        T0_fitted = []
        
        # Iterate over imap with the index to print progress
        for i, T0 in enumerate(pool.imap(fit_single_T0, zip(r_obs, v_obs))):
            T0_fitted.append(T0)
            
            # Print the percentage progress every 1% complete
            if (i + 1) % (n // 100) == 0:  # Print every 1% of the tasks
                percent_complete = (i + 1) / n * 100
                print(f"Progress: {percent_complete:.1f}% complete")
                sys.stdout.flush()
    return np.array(T0_fitted)

def coord_to_polar(coord):
    return coord.lon.to_value('rad'), coord.radius.to_value('AU')

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

def loadall(filename):
    with open(filename, "rb") as f:
        while True:
            try:
                yield cpkl.load(f)
            except EOFError:
                break

def group_elements(A, B):
    
    """
    Group elements in Array A to elements in Array B
    by which element in Array B is closest to the element in Array A.
    Returns an array with the indices of the closest elements in Array B
    for each element in Array A.
    """
    
    indices = np.zeros_like(A, dtype=np.int32)
    for i, a in enumerate(A):
        indices[i] = np.argmin(np.abs(B - a))
    return indices

def set_axes_equal(ax):
    
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])

def sliding_median(array, window_size):
    
    # Calculate padding amounts
    pad_left = window_size // 2
    pad_right = window_size - pad_left - 1  # Ensures total padding equals window_size - 1

    # Pad the array asymmetrically if window_size is even
    padded_array = np.pad(array, (pad_left, pad_right), mode='edge')

    # Create a 2D array of sliding windows
    rolling_window = np.lib.stride_tricks.sliding_window_view(padded_array, window_shape=window_size)

    # Compute the median along the window axis
    medians = np.nanmedian(rolling_window, axis=1)
    
    # Determine if the number of NaNs exceeds half the window size
    num_nans = np.sum(np.isnan(rolling_window), axis=1)
    # too_many_nans = num_nans > 3*window_size / 4
    too_many_nans = num_nans > 1*window_size / 4
    
    # Set medians to np.nan where there are too many NaNs
    medians[too_many_nans] = np.nan
    
    return medians

def sort_list(list1, list2):
 
    zipped_pairs = zip(list2, list1)
 
    z = [x for _, x in sorted(zipped_pairs)]
 
    return z

def SPC_SPI_Construct(enc,spi_time,spi_data,spc_time,spc_data):
    
    span_check_savename = 'SPAN_SPC_QTN_flags_enc_'+str(enc)+'.cdf'
    span_check_savepath = '/Users/besh2109/Documents/SPAN Checks/'
    
    pyt.tplot_restore(span_check_savepath+span_check_savename)

    SPAN_QTN_flag = pyt.get_data('SPAN_qual_flag')
    SPAN_flag_time = SPAN_QTN_flag[0]
    SPAN_flag = SPAN_QTN_flag[1]
    
    # SPC_QTN_flag = pyt.get_data('SPC_qual_flag')
    # SPC_flag_time = SPC_QTN_flag[0]
    # SPC_flag = SPC_QTN_flag[1]
    
    flag_zero_group = group_zeros(SPAN_flag)
    
    # bad_span_times = []
    vel_time_construct = np.array([],dtype=float)
    vel_mag_data_construct = np.array([],dtype=float)
    
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
            
            spc_where = np.where((spc_time>t0i)&(spc_time<tfi))
            spc_where = spc_where[0]
            
            vel_time_construct = np.append(vel_time_construct,spc_time[spc_where])
            vel_mag_data_construct = np.append(vel_mag_data_construct,spc_data[spc_where])
            
            good_bad = 1 #alternate between good and bad times. Should start with bad.
            # breakpoint()
        elif good_bad ==1:
            
            spi_where = np.where((spi_time>t0i)&(spi_time<tfi))
            spi_where = spi_where[0]
            
            vel_time_construct = np.append(vel_time_construct,spi_time[spi_where])
            vel_mag_data_construct = np.append(vel_mag_data_construct,spi_data[spi_where])
            
            good_bad = 0
            
    # breakpoint()
    
    #--------------check for Nans, and fill in with SPI if possible--------------#
    
    a = np.where(np.isnan(vel_mag_data_construct))
    
    one_arr = np.ones(vel_mag_data_construct.shape)
    one_arr[a] = 0
    
    groups = group_zeros(one_arr)
    index_list = []
    
    for group in groups:
        
        index_list.append(list(range(group[0],group[1])))
        # print(pys.time_string(t0_group),pys.time_string(tf_group))
        # check = np.delete(check,np.s_[group[0]:group[1]])
    
    indices_to_delete = [index for chunk in index_list for index in chunk]
    
    vel_time_temp = np.delete(vel_time_construct, indices_to_delete)
    vel_temp = np.delete(vel_mag_data_construct, indices_to_delete)
    
    for group in groups:
        t0 = vel_time_construct[group[0]]
        tf = vel_time_construct[group[1]]
        
        spi_where = np.where((spi_time>t0)&(spi_time<tf))
        spi_where = spi_where[0]
        
        vel_time_temp = np.append(vel_time_temp,spi_time[spi_where])
        vel_temp = np.append(vel_temp,spi_data[spi_where])
        
    
    # Get the sorted indices based on the time array
    sorted_indices = np.argsort(vel_time_temp)
    
    # Sort both arrays using the sorted indices
    sorted_time = vel_time_temp[sorted_indices]
    sorted_data = vel_temp[sorted_indices]
    
    #--------------output the result---------------------#
    
    time_arr = sorted_time
    data_arr = sorted_data
    
    return time_arr, data_arr

def encounter_check(day):
    
    i_day = pys.time_float(day)
    enci = 0
    for enc in enc_flt:          
        if (i_day>enc[0]) and (i_day<enc[1]): 
            encounter = enci+1 #determine which encounter input day is part of
        enci+=1
    
    return encounter

def encounter_dates(enc,rlim):
    
    Rs_km = 6.957e5*u.km #solar radius in km  
    
    hpos_path = CONFIG['local_data_dir']+'/fields/l1/ephem_eclipj2000/full_mission/' #historical position
    pyt.cdf_to_tplot(hpos_path+'spp_fld_l1_ephem_eclipj2000_20180812_090000_20250831_090000_v02.cdf')
    hpos = pyt.get_data('position')
    
    hpos_time_arr = hpos[0]
    hpos_data_arr = hpos[1]
    
    enc_ind = (enc-1)
    
    enc_sel = enc_flt[enc_ind]
    
    
    hpos_where = np.where((hpos_time_arr>enc_sel[0])&(hpos_time_arr<enc_sel[1]))
    hpos_where = hpos_where[0]
    
    hpos_time = hpos_time_arr[hpos_where]
    
    hposx_data = hpos_data_arr[hpos_where,0]*u.km
    hposy_data = hpos_data_arr[hpos_where,1]*u.km
    hposz_data = hpos_data_arr[hpos_where,2]*u.km
    
    R = ((np.sqrt(hposx_data**2+hposy_data**2+hposz_data**2)-Rs_km)/Rs_km)
    
    R_where = np.where(R<rlim)
    R_where = R_where[0]
    
    time_select = hpos_time[R_where]
    
    t0 = pys.time_string(time_select[0])
    tf = pys.time_string(time_select[-1])
    
    return t0, tf


def jsoc_check(enc_start=1,enc_end=22,rlim=70,enc_list=None):
    
        if enc_list == None:
            for i in range(enc_start,enc_end+1):
        
                t0,tf = encounter_dates(i,rlim)
                t_start_hmi = parse_time(t0)
                t_end_hmi = parse_time(tf)
                
                results_hmi = Fido.search(a.jsoc.Notify(os.environ["JSOC_EMAIL"]),a.jsoc.Time(t_start_hmi.value,t_end_hmi.value),a.jsoc.Series('hmi.mrdailysynframe_small_720s'))
                
                print("Encounter ",str(i))
                print(results_hmi)
                
                try:
                    filenames = Fido.fetch(results_hmi, path=os.environ['SUNPY_DATA_DIR']+'/hmi')
                    print(f"Download successful for Encounter {i}")
                except:
                    # Skip this iteration if an exception occurs
                    print(f"Skipping {i} due to error.")
        else:
             for i in enc_list:
         
                 t0,tf = encounter_dates(i,60)
                 t_start_hmi = parse_time(t0)
                 t_end_hmi = parse_time(tf)
                 
                 results_hmi = Fido.search(a.jsoc.Notify(os.environ["JSOC_EMAIL"]),a.jsoc.Time(t_start_hmi.value,t_end_hmi.value),a.jsoc.Series('hmi.mrdailysynframe_small_720s'))
                 
                 print("Encounter ",str(i))
                 print(results_hmi)
                 # breakpoint()
                 
                 try:
                    filenames = Fido.fetch(results_hmi, path=os.environ['SUNPY_DATA_DIR']+'/hmi')
                    print(f"Download successful for Encounter {i}")
                 except:
                    # Skip this iteration if an exception occurs
                    print(f"Skipping {i} due to error.")
                    res_atrs = results_hmi['JSOC']
                    result_times = res_atrs['T_REC']
                    
                    filenames = []
                    path_check = []
                    for i in result_times:
                        tmp_str = str(i)
                        full_path = 'hmi.mrdailysynframe_small_720s.'+tmp_str[:4]+tmp_str[5:7]+tmp_str[8:13]+tmp_str[14:16]+tmp_str[17:23]+'.data.fits'
                        filenames.append(full_path)
                    
                    print(filenames)
                    
                    with open('/Users/besh2109/Desktop/output.txt', 'a') as file:
                        file.writelines(string + '\n' for string in filenames)
                        
def PFSS_Br_estimation(pfss_out,seeds,r0_Rs,rss,A_scale=6.90):
        
    ss_br = pfss_out.source_surface_br
    # Convert Carrington coordinates to pixel coordinates
    pixel_coords = ss_br.world_to_pixel(seeds)

    # Retrieve data values at the pixel locations
    # Note: Pixel coordinates are in floating point; round or floor for integer indices
    
    x_pixels = np.round(pixel_coords.x)
    y_pixels = np.round(pixel_coords.y)
    
    
    nan_where = np.where(np.isnan(x_pixels))
    nan_where = nan_where[0]
    
    
    x_pixels[nan_where] = 0 #set nans equal to zero to make the next step possible.
    y_pixels[nan_where] = 0 #we'll throw these data points away afterward
    
    x_pixels = np.array(x_pixels,dtype=int) #convert np array to integers, i.e. which can be used as indices
    y_pixels = np.array(y_pixels,dtype=int)
    
    pfss_ss_br = ss_br.data[y_pixels,x_pixels] # retrieve values using pixel indices
    
    A_scale = A_scale #scaling factor for PFSS model B to match PSP data.
    br_pfss = A_scale*np.array(pfss_ss_br)*(rss*u.R_sun/(r0_Rs))**2 * u.G # in Gauss, propagate out to PSP distances.
    br_pfss[nan_where] = np.nan # put the nans back in
    br_pfss = br_pfss.to(u.nT) #convert to nT
    # breakpoint()

    return br_pfss