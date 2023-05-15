#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 16:34:29 2022

@author: malaspina
"""


###############
# Meta import
###############

import numpy as np 


####################
# Define the magnetic field 
####################

B     = [129.2818, 41.371666, 27.124466]

####################
# Define the Temperature tensor 
####################

tensor      = np.zeros([3,3])
tensor[:,0] = [29.17563   ,  0.21432723, -1.4489131 ]
tensor[:,1] = [0.21432723, 41.68735   , -0.76404285]
tensor[:,2] = [-1.4489131 , -0.76404285, 40.607883]

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

print('')
print('T parallel ', p_par)
print('T perp     ', p_perp)
print('T par / T prp ', p_par / p_perp)
print('')

## check by reproducing equation 6.17 with the derived p_perp and p_par values.  
## should recover the entire input temperature tensor. 
check = p_perp * I_tensor + (p_par / Mag_B**2) * BBT_array - (p_perp / Mag_B**2) * BBT_array


print('')
print('Check tensor')
print(check)
print('')
print('Input tensor')
print(tensor)
print('')

## Differences imply breaks from cylindrical symmetry with respect to B in the measurement
