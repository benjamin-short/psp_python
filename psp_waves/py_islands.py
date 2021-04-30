#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 11:51:14 2021

@author: benshort

py_islands finds all islands of four-connected elements in a matrix.
py_islands returns a matrix the same size as the input matrix, with all of 
the four-connected elements assigned an arbitrary island number. A second
return argument is an nx3 matrix.  Each row of this matrix has 
information about a particular island:  the first column is the island    
number which corresponds to the numbers in the first return argument, the     
second column is the number of elements in the island, the third column   
is the value of the elements in that island (the elements of the input
matrix).  py_islands will also return a binary matrix with ones in the 
positions of the elements of the largest four-connected island.  The 
largest four-connected island is determined first by value; for example  
if there are 2 islands each with 5 elements, one of which is made up of   
6's and the other of which is made up of 4's, the island with the 6's  
will be returned.  In the case where there are 2 islands with the same    
number of elements and the same value, an arbitrary choice will be 
returned.

Translated from ISLANDS matlab routine by Matt Fig.

"""

import numpy as np
import math

def islands(a,vmin=None): #requires array input to be a 2 dimensional numpy array, optional argument vmin applies a quick filter.
    
    if len(a.shape) != 2:
        print("Input array not the correct shape; input must be a 2D array.")
        print("This input array has dimension:",len(a.shape))
        AG = [] 
        G = []
        NSV = []
        return AG,G,NSV
    
    bool_check = a.max()
    
    if bool_check != True or bool_check != 1:
        if vmin == None:
            vmin = np.nanmedian(a)
        
        a = np.where(a <= vmin,np.nan,a) #converts array to boolean array.
        a = np.where(a > vmin,1,a)            
    
    nrows = a.shape[0]
    ncols = a.shape[1]
    
    AG = np.zeros(a.shape,dtype=np.int64)
    
    arr_size = 1
    for dim in a.shape: arr_size*=dim #counts the number of elements in input array
    
    V = np.zeros(math.ceil(arr_size/2),dtype=np.int64) #index array, index of each found island
    L = np.array(V) #array to hold number of elements in each island
    cntr = 1
    #vct = np.zeros(ncols)
    """
    Notes on comments in the loops:  CRNT is the element of the matrix we are
    currently looking at.  RGHT is the element to the immediate right of 
    CRNT.  RUD is the element to the right and up from the CRNT.
    The RUD element is directly above RGHT.
    """
    
    for gg in range(0,ncols-1): #Look along the first row.
        if a[0,gg] == a[0,gg+1]: #CRNT matches RGHT.
            if not AG[0,gg]: #CRNT does not have an island number.
                AG[0,gg] = cntr #Assign an island number to CRNT.
                AG[0,gg+1] = cntr #Assign an island number to RGHT.
                V[cntr-1] = a[0,gg] #Assign the value of the island.
                L[cntr-1] = 2 #Add to the island count.
                cntr+=1 #Increment the counter.
            else: #CRNT does have an island number.
                AG[0,gg+1] = AG[0,gg] #Assign the island number to RGHT.
                L[AG[0,gg]-1] += 1 #Add to the island count.
    
    for hh in range(0,nrows-1): # Look down the first column.
        if a[hh,0]==a[hh+1,0]: #  CRNT matches 'RGHT'.
            if not AG[hh,0]: # CRNT does not have an island number.
                AG[hh,0] = cntr # Assign an island number to CRNT.
                AG[hh+1,0] = cntr # Assign an island number to 'RGHT'.
                V[cntr-1] = a[hh,0] #Assign the value of the island.
                L[cntr-1] = 2 #Add to the island count.
                cntr += 1 #
            else: #CRNT does have an island number.
                AG[hh+1,0] = AG[hh,0] # Assign an island number to 'RGHT'.
                L[AG[hh,0]-1] += 1 #Add to the island count.
                
    #Now we can look at the rest of the matrix.
    for ii in range(1,nrows): # Start on the second row.
        for jj in range(0,ncols-1): #Start on the first column.
            if a[ii,jj]==a[ii,jj+1]: #CRNT matches RGHT.
                if a[ii,jj]==a[ii-1,jj+1]: # CRNT matches RUD too.
                    if not AG[ii,jj] and not AG[ii-1,jj+1]: #Both aren't yet grouped.
                        AG[ii,jj]=cntr #Give CRNT a new island number.
                        AG[ii-1,jj+1]=cntr #Give RUD the new island num.
                        AG[ii,jj+1]=cntr #Give RGHT the new island number.
                        V[cntr-1] = a[ii,jj] #Store the value of the island.
                        L[cntr-1] = 3 #Number of members in the new island.
                        cntr+=1 #Increment the counter.
                    elif not AG[ii-1,jj+1]: #RUD not yet been grouped, CRNT is.
                        AG[ii-1,jj+1] = AG[ii,jj] #Give RUD CRNTs isl. num.
                        AG[ii,jj+1]=AG[ii,jj] #And RGHT as well.
                        L[AG[ii,jj]-1] += 2 # Add to island size.
                    elif not AG[ii,jj]: #CRNT not yet grouped, RUD is grouped.
                        AG[ii,jj] = AG[ii-1,jj+1] #Give CRNT RUDs isl. num.
                        AG[ii,jj+1] = AG[ii-1,jj+1] #And RGHT as well.
                        L[AG[ii-1,jj+1]-1]+=2 #Add to cnt.
                    else: #Both CRNT and RUD have been grouped:  merge islands.
                          #First decide which island has the least members.
                        if L[AG[ii-1,jj+1]-1]<L[AG[ii,jj]-1]: #
                            idx = AG[ii-1,jj+1] #Save RUDs island number.
                            IDX = AG[ii,jj] # Used below: new islands.
                        else: #
                            idx = AG[ii,jj] #Save CRNTs island number.
                            IDX = AG[ii-1,jj+1] #Used below: new islands.
                        
                        cnt=1 #The counter.
                        if idx != IDX: #They could already match!
                            for pp in range(ii+1,nrows): #Must search first column too.
                                if AG[pp,0]==idx: #
                                    AG[pp,0]=IDX #Assign island number.
                                    cnt+=1 #Increment member counter.
                                else: #
                                    break #Stop search if one mismatch found.
                            for mm in reversed(range(0,ii+1)): #Start at current row, work up.
                                for kk in range(0,ncols): #
                                    if AG[mm,kk]==idx: #
                                        AG[mm,kk]=IDX #Assign new isl. num.
                                        cnt+=1 #Increment count.
                                        if cnt>L[idx-1]: #
                                            break #Stop search for old islnds.
                                if cnt>L[idx-1]: #
                                    break #Stop search for old islnds.
                        AG[ii,jj+1] = IDX #Give RGHT the island number.
                        L[IDX-1] += cnt #Add to count.
                        L[idx-1] = L[idx-1] - cnt + 1 # subtract from old island.
                        
                elif not AG[ii,jj]: #RGHT matches CRNT, not RUD. Need new isl.
                    AG[ii,jj] = cntr # Give CRNT a new island number.
                    AG[ii,jj+1] = cntr #Give RGHT the new island number.
                    V[cntr-1]=a[ii,jj] #Store the new island number.
                    L[cntr-1]=2 #The number of members in the new island.
                    cntr+=1 #Increment the counter.
                else: # RGHT matches CRNT, not RUD.  No new island needed.
                    AG[ii,jj+1] = AG[ii,jj] #Give RGHT CRNTs island number.
                    L[AG[ii,jj]-1]+=1 #Add to island count.
            elif a[ii,jj+1] == a[ii-1,jj+1]: # RUD & RGHT match, not CRNT.
                if not AG[ii-1,jj+1]: #RUD has not yet been grouped.
                    AG[ii,jj+1] = cntr #Give RGHT new island number.
                    AG[ii-1,jj+1] = cntr #Give RUD new island number.
                    V[cntr-1] = a[ii,jj+1] # Store the value of the island.
                    L[cntr-1] = 2 #Add to island count.
                    cntr+=1 #
                else: # RUD is already part of a island.
                    AG[ii,jj+1] = AG[ii-1,jj+1] #Add RGHT to RUD's island.
                    L[AG[ii-1,jj+1]-1] += 1 #Add island cnt.
    del a
    

    
    
    N = np.array(list(range(1,len(L)+1)))
    
    
    Ntmp = np.delete(N,L==0) #island number
    Ltmp = np.delete(L,L==0) #island area
    Vtmp = np.delete(V,L==0) #
    
    Xstart = []
    Ystart = []
    
    Xend = []
    Yend = []
    
    Xrange = []
    Yrange = []
    for i in Ntmp:
        tmp = np.where(AG == i)
        
        y_vals = tmp[0]
        x_vals = tmp[1]
        
        y_start = y_vals.min()
        x_start = x_vals.min()
        
        y_end = y_vals.max()
        x_end = x_vals.max()
        
        y_range = abs(y_end-y_start) + 1 #add 1 to define range as one at least
        x_range = abs(x_end-x_start) + 1 #
        
        Ystart.append(y_start)
        Xstart.append(x_start)
        
        Yend.append(y_end)
        Xend.append(x_end)
        
        Yrange.append(y_range)
        Xrange.append(x_range)
        #print(x_start,x_end,x_range)
    
    Ystart = np.array(Ystart) #array containing the starting y index
    Xstart = np.array(Xstart) #array containing the starting x index
    Yend = np.array(Yend) #ending y index
    Xend = np.array(Xend) #ending x index
    Yrange = np.array(Yrange) #length of island by y coordinate
    Xrange = np.array(Xrange) #length of island by x coordinate
    
    NSV = np.vstack((Ntmp,Ltmp,Ystart,Yend,Yrange,Xstart,Xend,Xrange))
    
    if len(Ltmp) != 0:
        
        tmp = np.where(Ltmp==Ltmp.max())
        tmp = tmp[0][0]  #i dont know why np.where outputs this ugly format.

        G = np.where(AG!=Ntmp[tmp],0,AG)
        
    else: 
        G = AG

    return AG,G,NSV