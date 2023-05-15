#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:34:29 2022

@author: malaspina
"""


###############
# Meta import
###############

import sys
import gc
import numpy as np
gc.collect()  # deactivate to cause memory leak

def david_rot_mat(B): #function takes in a magnetic field vector and constructs a rotation matrix.

    ####################
    # Rotate (E, B, S) from SC coordinates into FAC
    ####################
    mag_sc_x_smooth = np.array([B[0], B[0]])
    mag_sc_y_smooth = np.array([B[1], B[1]])
    mag_sc_z_smooth = np.array([B[2], B[2]]) 
    
    # Determine angles for rotation matricies
    phi   = np.arctan(np.abs(mag_sc_y_smooth)/np.abs(mag_sc_x_smooth))
    theta = np.arctan(mag_sc_z_smooth / np.sqrt(mag_sc_x_smooth**2 + mag_sc_y_smooth**2))
        
    # Add in quantrant specific changes to get signs right (don't do anything for the first case)
    #here1 = pm.np.where( pm.np.logical_and(mag_sc_x_smooth > 0 , mag_sc_y_smooth > 0) )[0]
    here2 = np.where(np.logical_and(
        mag_sc_x_smooth > 0, mag_sc_y_smooth < 0))[0]
    here3 = np.where(np.logical_and(
        mag_sc_x_smooth < 0, mag_sc_y_smooth > 0))[0]
    here4 = np.where(np.logical_and(
            mag_sc_x_smooth < 0, mag_sc_y_smooth < 0))[0]
    #here5 = pm.np.where( mag_sc_z_smooth > 0 )[0]
    #here6 = pm.np.where( mag_sc_z_smooth < 0 )[0]
    
    # if len(here1) != 0: phi[here] = phi[here]
    if len(here2) != 0:
        phi[here2] = 2. * np.pi - phi[here2]
    if len(here3) != 0:
            phi[here3] = np.pi - phi[here3]
    if len(here4) != 0:
            phi[here4] = phi[here4] + np.pi
        #if len(here5) != 0: theta[here5] = theta[here5]
        #if len(here6) != 0: theta[here6] = pm.np.pi - theta[here6]
    
        # Build Phi rotation arrays (rotation about z)
    sobig = len(phi)
    rot_matrix_phi = np.zeros([sobig, 3, 3])
    top_row = np.zeros([sobig, 3])
    mid_row = np.zeros([sobig, 3])
    low_row = np.zeros([sobig, 3])
    
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    top_row[:, 0] = cos_phi
    top_row[:, 1] = sin_phi
    mid_row[:, 0] = -sin_phi
    mid_row[:, 1] = cos_phi
    low_row[:, 2] = 1.
    
    rot_matrix_phi[:, 0, :] = top_row
    rot_matrix_phi[:, 1, :] = mid_row
    rot_matrix_phi[:, 2, :] = low_row
    
    # Build Theta rotation arrays(rotation about x)
    rot_matrix_theta = np.zeros([sobig, 3, 3])
    top_row = np.zeros([sobig, 3])
    mid_row = np.zeros([sobig, 3])
    low_row = np.zeros([sobig, 3])
    
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    
    top_row[:, 0] = cos_theta
    top_row[:, 2] = sin_theta
    mid_row[:, 1] = 1.
    low_row[:, 0] = -sin_theta
    low_row[:, 2] = cos_theta
    
    rot_matrix_theta[:, 0, :] = top_row
    rot_matrix_theta[:, 1, :] = mid_row
    rot_matrix_theta[:, 2, :] = low_row
    
    ## Combine the two rotations 
    ind       = 0
    rot_total = np.matmul(rot_matrix_theta[ind, :, :], rot_matrix_phi[ind, :, :])
    
    return rot_total
    
def david_anis(B,tensor):
    
    
    ####################
    # Import the Temperature tensor 
    ####################
    
    tensor = np.array(tensor)
    
    ####################
    # Following Pashman et al. 1998
    # Moments of Plasma Velocity Distributions
    # Equation 6.17
    ####################
    
    ## Magnitude of B 
    Mag_B = np.sqrt( B[0]**2 + B[1]**2 + B[2]**2   )
    
    ## Form the array B B^T
    BBT_array      = np.zeros([3,3])
    BBT_array[:,0] = [B[0]*B[0], B[0] * B[1], B[0] * B[2]]
    BBT_array[:,1] = [B[1]*B[0], B[1] * B[1], B[1] * B[2]]
    BBT_array[:,2] = [B[2]*B[0], B[2] * B[1], B[2] * B[2]]
    
    ## Define the identity tensor 
    I_tensor      = np.zeros([3,3])
    I_tensor[:,0] = [1,0,0]
    I_tensor[:,1] = [0,1,0]
    I_tensor[:,2] = [0,0,1]
    
    ## Use two equations (P_00 and P_22) from the Equation 6.17 matrix 
    ## Solve for P_perp and P_parallel 
    
    Bx = B[0]
    By = B[1]
    Bz = B[2]
    
    ## P_perp expression 
    left   = tensor[0,0] - tensor[2,2] * Bx**2 / Bz**2
    right  = (1 - Bx**2 / Bz**2)
    p_perp = left / right 
    
    ## P_parallel expression 
    p_par = ( tensor[2,2,] - p_perp*(1 - Bz**2 / Mag_B**2 ) ) * (Mag_B**2 / Bz**2)
    
    t_anis = p_perp/p_par
    
    return t_anis
    
def steven_anis(B,tensor):
    
    tensor = np.array(tensor)
    
    Mag_B = np.sqrt( B[0]**2 + B[1]**2 + B[2]**2   )
    
    Bx = B[0]
    By = B[1]
    Bz = B[2]
    
    b_unit = np.array([Bx, By, Bz])/Mag_B
    
    atempt  = np.matmul(tensor, b_unit)            
    T_par   = np.matmul(b_unit, atempt)  
    trace  = np.trace(tensor)
    
    T_per = (trace - T_par)/2
    
    t_anis = T_per/T_par
    
    return t_anis
    
