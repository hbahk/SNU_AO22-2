#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 24 20:01:51 2021

@author: hbahk
"""
#%%
import pandas as pd
from pathlib import Path
import numpy as np
import astropy.io.ascii as ascii

HOME = Path.home()
WD = HOME/'class'/'ao22'/'SNU_AO22-2'/'data'

specpath = WD/'mhr9087.dat'
specpath2 = WD/'fhr9087.dat'

df = pd.read_csv(specpath, names=['wave', 'mag', 'width'], delim_whitespace=True)
df2 = pd.read_csv(specpath2, names=['wave', 'flux', 'flux2', 'width'], delim_whitespace=True)
print(df)
stdfile = WD/'hr9087.csv'
df.to_csv(stdfile)


def _mag2flux(wave, mag, zeropt=48.60):
    '''
    Convert magnitudes to flux units. This is important for dealing with standards
    and files from IRAF, which are stored in AB mag units. To be clear, this converts
    to "PHOTFLAM" units in IRAF-speak. Assumes the common flux zeropoint used in IRAF

    Parameters
    ----------
    wave : 1d numpy array
        The wavelength of the data points
    mag : 1d numpy array
        The magnitudes of the data
    zeropt : float, optional
        Conversion factor for mag->flux. (Default is 48.60)

    Returns
    -------
    Flux values!
    '''

    c = 2.99792458e18 # speed of light, in A/s
    flux = 10.0**( (mag + zeropt) / (-2.5) )
    return flux * (c / wave**2.0)

mflux = _mag2flux(df['wave'].values, df['mag'].values)
fflux = df2['flux'].values*1e-16
import matplotlib.pyplot as plt

plt.plot(df['wave'].values,mflux)
plt.plot(df['wave'].values,fflux)
plt.plot(df['wave'].values,fflux - mflux)

#%%

stddata = ascii.read(stdfile)
std_wave, std_mag, std_wth  = stddata['wave'],stddata['mag'],stddata['width']
std_flux = _mag2flux(std_wave,std_mag)