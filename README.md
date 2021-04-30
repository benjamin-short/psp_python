# psp_python
psp python code folder

This repository contains a few pieces of code used to download, view, and analyze data from the
PSP mission.

# Requirements:

Python 3.8
pyspedas package
numpy package
matplotlib package

These sets of programs rely on the use of Python Environment variables for things like 
data storage pathing, server data username and passwords.

Using the programming language IDL, you would do this by modifying your bash file and setting
the environment variables manually. With Python there is a better way.

# Better way: 

Install Python 3.8 using Anaconda.

Anaconda Download: https://www.anaconda.com/products/individual

After downloading Anaconda for your machine, your Terminal or Command Prompt will have a
new label at the far left of the command line.

For example, before the anaconda install, my command line looked like this:

username@users-MacBook-Pro ~ % 

After anaconda install:

(base) username@users-MacBook-Pro ~ % 

This new (base) specifies what Conda environment you are currently in. Namely, the Anaconda base environment.

Anaconda also installs new commands for the command line.

You can create new anaconda environments with the command:

(base) username@users-MacBook-Pro ~ % conda create --name myenv

(replace myenv with your desired environment name)

you can also specify which python version you want.

(base) username@users-MacBook-Pro ~ % conda create -n myenv python=3.6

Personally, I work with two environments:

My (base) environment is where I keep all my unmodified programs. This repository modifies 
some of the python packages we're going to install and having a clean copy somewhere is always
a good idea.

I then have my work environment that I called (parker) but the name doesnt really matter for
Python environment variables.

Once you've created a new environment that you'd like to work in, you switch to your
new environment with the command:

(base) username@users-MacBook-Pro ~ % conda activate parker

You will then see


(parker) username@users-MacBook-Pro ~ %

which indicates that you have changed Conda Environments.

From here you can set your environment variables.

Some important environment variables:

Password and Username for the FIELDS team: 
'PSP_FIELDS_ID',
'PSP_FIELDS_PW'

Password and Username for the SWEAP team:
'PSP_SWEAP_ID',
'PSP_SWEAP_PW'

Password and Username for the ISOIS team:
'PSP_ISOIS_ID',
'PSP_ISOIS_PW'

If you do not have credentials for these, pyspedas will download data from the public servers only.

Paths to store your data: 
'SPEDAS_DATA_DIR',
'PSP_DATA_DIR'

If you dont have one of these set, pyspedas will store any downloaded data in whatever 
directory your Terminal/Command Prompt is in currently. Kind of annoying.

To set environment variables:

(your env) user@users-MacBook-Pro ~ % conda env config vars set 'PSP_FIELDS_ID'='your ID'

More on managing Anaconda Environments here:
https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands

# Using this repository

Once the environment variables have been set we can start working.

pip install packages you want/need; namely: pyspedas

Run the command: 

(your env) user@users-MacBook-Pro ~ % pip install pyspedas

and this should install everything you need.

Once that is done, download this repository.

Pyspedas on its own doesnt know where or how to download data from servers other than
NASAs public servers. In this repository we've told it how to do that.

Locate the folder that your pyspedas package is stored in.

For Mac that folder is:

/Users/*username*/opt/anaconda3/envs/*your env*/lib/python3.8/site-packages/pyspedas/

Inside the pyspedas folder, locate the "psp" folder.

*Replace that "psp" folder with the same folder but from this repository.*

Once that is done, you are ready to begin work!

# Download and view data example:

Run:

(your env) user@users-MacBook-Pro ~ % python

This will open Pythons interactive shell.

On the new line:

"""
import pyspedas
from pytplot import tplot

fields_vars = pyspedas.psp.fields(trange=['2018-11-5', '2018-11-5/06:00'], datatype='mag_RTN', level='l2') 

tplot('psp_fld_l2_mag_RTN')
"""

More examples: https://github.com/spedas/pyspedas/tree/master/pyspedas/psp

If you are familiar with IDLs SPEDAS routines, several of the original and most useful functions are present
in the python release.

"""
import pyspedas as pys
import pytplot as pyt
from matplotlib import pyplot as plt

mag = pys.psp.fields(trange=['2018-11-5', '2018-11-5/06:00'], datatype='mag_RTN', level='l2') 

mag_data = pyt.get_data('psp_fld_l2_mag_RTN')

mag_time_arr = mag_data[0]  # the Epoch time of N magnetic field data points.
mag_data_arr = mag_data[1] # 2d array with 3xN data points. (3 magnetic field channels, N data points)

date_string = pys.time_string(mag_time_arr[0],fmt='%Y-%m-%d %H:%M:%S')

plt.plot(mag_data_arr[:,0], linewidth=0.2,label='Br')
plt.plot(mag_data_arr[:,1],linewidth=0.2,label='Bt')
plt.plot(mag_data_arr[:,2],linewidth=0.2,label='Bn')
plt.title(date_string+' Magnetic Field Data! In Python!')
plt.set_ylabel('mag_rtn (nT)')

leg = plt.legend(loc='upper left')
leg.legendHandles[0].set_linewidth(1.0)
leg.legendHandles[1].set_linewidth(1.0)
leg.legendHandles[2].set_linewidth(1.0)

plt.show()
"""
