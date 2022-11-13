#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy.interpolate import interp1d

def err_weighted_combine(all_wave, all_flux, all_error, dispersion=None,
                         **kwargs):
    """
    Weigthed Combine
    - from costool procedure
      COADDITION: flux elements are coadded pixel by pixel according to several
      methods set by the keyword method. In our extensive testing, modified
      exposure time weighting with flagging seems to produce the best
      coadditions with the least spurious features. Thus we have made method=1
      the default.
        method = -1 - simple mean of pixel values, gives too much weight to
                      exposures
        method =  1 - modified exposure weighting: exposure time is modified at
                      each pixel location by flanging and wire shadows
                      (if selected).
        method =  2 - err^-2 weighting, allows error array tuning, but tends to
                      weight toward lower-flux pixels.
        method =  3 - (S/N)^2 weighting, allows for error array tuning, but
                      tends to weight toward higher-flux pixels.

    This code corresponds to method=2, the error weighted combine.
    
    This code is based on the repo below.
    https://github.com/hamogu/spectrum/blob/master/spectrum/coadd.py

    Parameters
    ----------
    all_wave : numpy array
        stacked wavelength array.
    all_flux : numpy array
        stacked flux array.
    all_error : numpy array
        stacked error array.
    dispersion : TYPE, optional
        dispersion (wavelength array) for result spectra. The default is None,
        which selects first wavelength array of given spectra.
    
    **kwargs : kwargs for scipy.interpolation.interp1d

    the number of spectra to be combined should be identical for all_wave,
    all_flux, all_error.

    Returns
    -------
    error-weighted spectrum (wavelength, flux, error)

    """
    n_spec = len(all_flux)
    
    if dispersion is None:
        dispersion = all_wave[0]
    
    spec_shape = (n_spec,len(dispersion))
    fluxes, errors = np.ma.zeros(spec_shape), np.ma.zeros(spec_shape)
    
    for i in range(n_spec):
        f_interp = interp1d(all_wave[i], all_flux[i], **kwargs)
        e_interp = interp1d(all_wave[i], all_error[i], **kwargs)
        f_new, e_new = f_interp(dispersion), e_interp(dispersion)
        fluxes[i,:] = f_new
        errors[i,:] = e_new
        
    # First, make sure there is no flux defined if there is no error.
    errors = np.ma.fix_invalid(errors)
    if np.ma.is_masked(errors):
        fluxes[errors.mask] = np.ma.masked
    # This can be simplified considerably as soon as masked quantities exist.
    fluxes = np.ma.fix_invalid(fluxes)
    # There are no masked quantities yet, so make sure they are filled here.
    flux = np.ma.average(fluxes, axis=0, weights = 1./errors**2.).filled(np.nan)
    error = np.sqrt(1. / np.ma.sum(1./errors**2., axis=0).filled(np.nan))
    
    return dispersion, flux, error