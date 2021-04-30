#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  5 13:43:47 2021

@author: benshort
"""

import os
import cv2
from pyspedas import time_string
from pyspedas import time_float
from datetime import date
from glob import glob
import pandas as pd
import numpy as np
import csv

from .config import enc_flt
from .config import per_flt
from .config import per_dist_lst

def mylistdir(directory):
    """A specialized version of os.listdir() that ignores files that
    start with a leading period. To kill .DS_store files."""
    filelist = os.listdir(directory)
    return [x for x in filelist
            if not (x.startswith('.'))]

def event_find(t0='2018-10-03',view_type=0): #used to find interesting waves
    
    i_day = time_float(t0)
    min_day = time_float('2018-10-02')
    max_day = time_float(date.today().strftime("%Y-%m-%d"))
    
    if view_type == 0:
        view_fold = 'psp_pictures'
        file_suffix = '_plot.png'
    else:
        view_fold = 'psp_islands'
        file_suffix = '_isl.png'
    
    viewpath = '/Users/besh2109/Desktop/'+view_fold
    savepath = '/Users/besh2109/Desktop/interesting_waves'
    i_min = 0
    index = 1
    while i_day > min_day and i_day < max_day:
        
        
        while i_min <= 85200.0 and i_min >=0:
            
            t_win = i_day + i_min
            
            file = time_string(t_win,fmt='/%Y/%m/%d/psp_%Y%m%d%H%M'+file_suffix)
            
            isfile = os.path.isfile(viewpath+file)
            
            if isfile:
                
                img = cv2.imread(viewpath+file)
                cv2.imshow('image',img)
                
                k = cv2.waitKey(0)
                
                if k == ord('p'): #skips forward 20 minutes
                    i_min+=1200.
                    index = 1
                if k == ord('o'): #skips backward 20 minutes
                    i_min-=1200.
                    index = 0

                if k == ord('P'): #skips forward a day
                    i_day+=86400.
                    index = 1
                if k == ord('O'): #skips backward a day
                    i_day-=86400.
                    index=0
                
                if k == ord('s'): #saves the image to interesting waves folder
                    isdir = os.path.isdir(savepath+time_string(t_win,fmt='/%Y/%m/'))
                    if isdir == False:
                        os.makedirs(savepath+time_string(t_win,fmt='/%Y/%m/'))
                    cv2.imwrite(savepath+time_string(t_win,fmt='/%Y/%m/psp_%Y%m%d%H%M'+file_suffix),img)
                    print(time_string(t_win,fmt='psp_%Y%m%d%H%M'+file_suffix)+' has been saved!')
                if k == ord('q'): #quits the program
                    cv2.destroyAllWindows()
                    i_day = 0
            else:
                if index == 1:
                    i_min+=1200.
                elif index == 0:
                    i_min-=1200.
        
        if index == 1:
            i_day+=86400.
            i_min = 0
        elif index == 0:
            i_day-=86400.
            i_min = 85200.
    cv2.destroyAllWindows()
    
    
def event_sort():   #used to sort interesting wave events by what is interesting about them
    
    fold_path = '/Users/besh2109/Desktop/interesting_waves/' #folder path
    tmp_path_list = glob(fold_path+'*/*/')
    tmp_path_list.sort()
    neglect = len(glob(fold_path+'sorted/*'))+1 # specifies only folders with unsorted images.
    path_list = tmp_path_list[0:len(tmp_path_list)-neglect]
    save_tmp = fold_path+'sorted/'
    file_list = []
    file_name = []
    for i in tmp_path_list:
        tmp_list = mylistdir(i) #calls function defined above, kills Mac .DS_store files.
        for j in tmp_list:
            file_list.append(i+j)
            file_name.append(j)
    
    file_list.sort()
    file_name.sort()

    i_file = 0
    
    print('')
    print('Press "p","o" to scroll forwards,backwards.')
    print('')
    print('Press "a" to sort as misc,')
    print('Press "s" to sort as shock,')
    print('Press "d" to sort as david_bernstein,')
    print('Press "f" to sort as freq_ramp,')
    print('Press "g" to sort as ghost,')
    print('Press "b" to sort as oscillating b field,')
    print('Press "v" to sort as very low freq,')
    print('Press "c" to sort as loud.')
    print('')
    print('Press "q" to quit.')
    print('')
    
    while i_file >= 0 and i_file <= len(file_list)-1:
        
        img = cv2.imread(file_list[i_file])
        cv2.imshow('image',img)
                
        k = cv2.waitKey(0)
                
        if k == ord('p'): #moves forward one file
            i_file+=1

        if k == ord('o'): #moves backward one file
            i_file-=1
                
        if k == ord('s'): #saves the image to shock folder, if wave power associated with shocks
            
            cv2.imwrite(save_tmp+'shock/'+file_name[i_file],img)
            print(file_name[i_file]+' has been sorted as shock!')
            print('')
        
        if k == ord('a'): #saves to the "I dont know what that is" folder
            
            cv2.imwrite(save_tmp+'misc/'+file_name[i_file],img)
            print(file_name[i_file]+' has been sorted as miscellaneous!')
            print('')

        if k == ord('d'):
            
            cv2.imwrite(save_tmp+'david_bernstein/'+file_name[i_file],img)
            print(file_name[i_file]+' has been sorted as david_bernstein!')
            print('')

        if k == ord('f'):
        
            cv2.imwrite(save_tmp+'freq_ramp/'+file_name[i_file],img)
            print(file_name[i_file]+' has been sorted as freq_ramp!')
            print('')

        if k == ord('g'):
        
            cv2.imwrite(save_tmp+'ghost/'+file_name[i_file],img)
            print(file_name[i_file]+' has been sorted as ghost!')
            print('')
        
        if k == ord('b'):
        
            cv2.imwrite(save_tmp+'wiggling_b/'+file_name[i_file],img)
            print(file_name[i_file]+' has been sorted as wiggling_b!')
            print('')

        if k == ord('v'):
        
            cv2.imwrite(save_tmp+'vlf_whistler/'+file_name[i_file],img)
            print(file_name[i_file]+' has been sorted as vlf_whistler!')
            print('')
        
        if k == ord('c'):
        
            cv2.imwrite(save_tmp+'loud/'+file_name[i_file],img)
            print(file_name[i_file]+' has been sorted as loud!')
            print('')
        
        if i_file < 0:
            i_file = len(file_list)-1
            
        if i_file > len(file_list)-1:
            i_file = 0
        
        if k == ord('q'): #quits the program
            cv2.destroyAllWindows()
            i_file = -100
        
def wave_sort(arg=''):   #used to sort interesting wave events by what is interesting about them
    
    fold_path = '/Users/besh2109/Desktop/psp_islands/waveforms/' #folder path
    tmp_path_list = glob(fold_path+'*/*/*/')
    
    csv_filepath = '/Users/besh2109/Desktop/psp_islands/wave_data/'
    csv_filename =  []
    
    csv_unsortpath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
    unsort_csv = 'unsorted_waves.csv' #prepare to keep a list of waves we didnt get to.
    
    master_csv = np.zeros((11,))
    enc_csv = []
    enc_csv_names = []
    
    for i in range(1,len(per_flt)+1):
        csv_name = 'enc_'+str(i)+'_harmwave_raw.csv'
        csv_filename.append(csv_name)
        df0 = pd.read_csv(csv_filepath+csv_name)
        bf0 = df0.to_numpy()
        master_csv = np.vstack((master_csv,bf0))
        enc_csv.append(bf0)
        enc_csv_names.append('enc_'+str(i)+'_harmwave_sorted.csv')   
        
    master_csv = np.delete(master_csv, 0, 0) #snip off the zeros used to create master_csv initially
   
    tmp_arr = np.zeros((11,)) #sort out duplicates
    i = 0
    while i < len(master_csv[:,0])-1:
        if master_csv[i,0] == master_csv[i+1,0]:
            tmp_list = [master_csv[i,1],master_csv[i+1,1]]
            
            if master_csv[i,1]>=master_csv[i+1,1]:
                tmp_arr = np.vstack((tmp_arr,master_csv[i,:]))
            else:
                tmp_arr = np.vstack((tmp_arr,master_csv[i+1,:]))

            i+=1
        else:
            tmp_arr = np.vstack((tmp_arr,master_csv[i,:]))
        
        i+=1
    tmp_arr = np.vstack((tmp_arr,master_csv[i,:]))
    master_csv = np.delete(tmp_arr,0,0)
    
    enco = []
    
    for i in range(len(master_csv[:,0])):
        date = time_float(master_csv[i,0])
        enci=0
        for enc in enc_flt:
            if (date>enc[0]) and (date<enc[1]): 
                enco.append(enci+1) #determine which encounter today is in
            enci+=1
    
    tmp_path_list.sort()
    neglect = len(glob(fold_path+'sorted/*'))+1 # specifies only folders with unsorted images.
    path_list = tmp_path_list[0:len(tmp_path_list)-neglect]
    save_tmp = csv_unsortpath
    file_list = []
    file_name = []
    for i in tmp_path_list:
        tmp_list = mylistdir(i) #calls function defined above, kills Mac .DS_store files.
        for j in tmp_list:
            file_list.append(i+j)
            file_name.append(j)
    
    file_list.sort()
    file_name.sort()
    
    if arg == 'unsorted': #only consider waves that have yet to be sorted
        df1 = pd.read_csv(csv_unsortpath+unsort_csv)
        bf1 = df1.to_numpy()
        master_csv = np.array(bf1)
        wave_names = np.array(bf1[:,0])
        pic_names = []
        #print(wave_names)
        for i in range(len(wave_names)):
            name = wave_names[i]
            pic_form = name[0:4]+name[5:7]+name[8:10]+'_'+name[11:13]+name[14:16]+name[17:19]
            pic_names.append('wave_'+pic_form+'.png')
        
        pic_names.sort()
        
        filepath_list = []
        filename_list = []
        for i in range(len(file_name)):
            #print(file_name[i]==pic_names[i])
        
            if file_name[i] in pic_names:
                filename_list.append(file_name[i])
                filepath_list.append(file_list[i])
                
        file_list = list(filepath_list)
        file_name = list(filename_list)
    
    i_file = 0
    
    ind_list = []
    
    print('')
    print('Press "p","o" to scroll forwards,backwards.')
    print('')
    print('Press "y" to sort as yes,')
    print('Press "n" to sort as no,')
    print('')
    print('Press "q" to quit.')
    print('')
    
    #"""
    while i_file >= 0 and i_file <= len(file_list)-1:
        
        img = cv2.imread(file_list[i_file])
        cv2.imshow('waveform',img)
                
        k = cv2.waitKey(0)
                
        if k == ord('p'): #moves forward one file
            i_file+=1

        if k == ord('o'): #moves backward one file
            i_file-=1
                
        if k == ord('y'): #saves the wave as 'yes' and writes its info to its respective encounter CSV and the Master CSV
            
            row = [str(master_csv[i_file,0]),str(master_csv[i_file,1]),str(master_csv[i_file,2]),
                   str(master_csv[i_file,3]),str(master_csv[i_file,4]),str(master_csv[i_file,5]),
                   str(master_csv[i_file,6]),str(master_csv[i_file,7]),str(master_csv[i_file,8]),
                   str(master_csv[i_file,9]),str(master_csv[i_file,10])]
            
            csv_arr = np.transpose(master_csv[i_file,:])
            
            csv_savepath = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/Enc'+str(enco[i_file])+'/'
            csv_savename = 'enc_'+str(enco[i_file])+'_harmwave_sorted.csv'
            
            csv_isdir = os.path.isdir(csv_savepath)
            csv_isfile = os.path.isfile(csv_savepath+csv_savename)
            
            if not csv_isdir:
                os.makedirs(csv_savepath)    
            
            if not csv_isfile:

                
                header=['wave date','length of wave','freq@max pow','f/fce@max pow','median freq',
                                'median f/fce', 'mean freq','mean f/fce','fce', 'dist in Rs','dist to peri']
                d = open(csv_savepath+csv_savename,"w")
                writer = csv.writer(d)
                writer.writerow(header)
                d.close()
                
                            
            d = open(csv_savepath+csv_savename,"a") #as unfile:
            writer = csv.writer(d)
            writer.writerow(row)
            d.close()

            
            csv_mastersave = '/Users/besh2109/Desktop/psp_islands/wave_data/sorted_csvs/'
            csv_master = 'harmwave_sorted_master.csv'
                      
            master_isdir = os.path.isdir(csv_mastersave)
            master_isfile = os.path.isfile(csv_mastersave+csv_master)
            
            if not master_isdir:
                os.makedirs(csv_mastersave)
                            
            if not master_isfile:
                
                header=['wave date','length of wave','freq@max pow','f/fce@max pow','median freq',
                                'median f/fce', 'mean freq','mean f/fce','fce', 'dist in Rs','dist to peri']
                e = open(csv_mastersave+csv_master,"w")
                writer = csv.writer(e)
                writer.writerow(header)
                e.close()
                
            e = open(csv_mastersave+csv_master,"a") #as unfile:
            writer = csv.writer(e)
            writer.writerow(row)
            e.close()
            
            ind_list.append(i_file) #use this index list to mark indices that I've sorted
            i_file+=1
            
        if k == ord('n'): #sort a wave as 'no', basically just ignores it and marks the wave down as having been sorted
            
            ind_list.append(i_file)
            i_file+=1
        
        if k == ord('q'): #quits the program
            cv2.destroyAllWindows()
            ind_list = list(np.unique(ind_list)) #kill all duplicates
            
            unsort_isdir = os.path.isdir(csv_unsortpath)
            unsort_isfile = os.path.isfile(csv_unsortpath+unsort_csv)
            
            for i in range(len(file_list)):
                if i not in ind_list:
                    csv_arr = np.transpose(master_csv[i,:])
                    if not unsort_isdir:
                        os.makedirs(csv_unsortpath)
                    
                    if not unsort_isfile:
                        header = np.array(['wave date','length of wave','freq@max pow','f/fce@max pow','median freq','median f/fce', 'mean freq',
                                                   'mean f/fce','fce', 'dist in Rs','dist to peri'])
                        header = np.transpose(header)
                        np.savetxt(csv_unsortpath+unsort_csv,np.array([]),
                                   header='wave date,length of wave,freq@max pow,f/fce@max pow,median freq,median f/fce, mean freq,'+
                                   'mean f/fce,fce, dist in Rs,dist to peri',
                                   fmt='%s',delimiter=',')
                    """
                    row = [str(master_csv[i,0])+','+str(master_csv[i,1])+','+str(master_csv[i,2])+','+\
                        str(master_csv[i,3])+','+str(master_csv[i,4])+','+str(master_csv[i,5])+','+\
                        str(master_csv[i,6])+','+str(master_csv[i,7])+','+str(master_csv[i,8])+','+\
                        str(master_csv[i,9])+','+str(master_csv[i,10])]
                    """
                        
                    row = [str(master_csv[i,0]),str(master_csv[i,1]),str(master_csv[i,2]),
                        str(master_csv[i,3]),str(master_csv[i,4]),str(master_csv[i,5]),
                        str(master_csv[i,6]),str(master_csv[i,7]),str(master_csv[i,8]),
                        str(master_csv[i,9]),str(master_csv[i,10])]
                    
                    #print(row)
                    if i == 0:
                        header=['wave date','length of wave','freq@max pow','f/fce@max pow','median freq',
                                'median f/fce', 'mean freq','mean f/fce','fce', 'dist in Rs','dist to peri']
                        f = open(csv_unsortpath+unsort_csv,"w")
                        writer = csv.writer(f)
                        writer.writerow(header)
                        f.close()
                    else:
                        
                        f = open(csv_unsortpath+unsort_csv,"a") #as unfile:
                        writer = csv.writer(f)
                        writer.writerow(row)
                        f.close()
                        #np.savetxt(unfile,csv_arr,delimiter=',',fmt='%s')
                        
            i_file = -100
    #"""