# This script aims at computing and plotting the IUH field along with the surface precipitation and hail fields for a given time shot,
# in order to asses how they match or deviate from each other

import numpy as np
import matplotlib.pyplot as plt
import time as chrono
from datetime import date, time, datetime
import pandas as pd
import xarray as xr
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm, BoundaryNorm
from matplotlib.ticker import ScalarFormatter

# GLOBAL VARIABLES AND FUNCTIONS
#========================================================================================================================================

#global constants
Rm = 6370000. # mean Earth's radius (m)
g = 9.80665 # standard gravity at sea level (m/s^2)
Ra = 287.05 #  Individual Gas Constant for air (J/K/kg)


def zeta_plev(fname_p, fname_s, plev):
    """
    Computes the relative vertical vorticity 2D field over a given pressure level at a single time shot

    Parameters
    ----------
    fname_p : str
        path to the 1h 3D pressure file containing the wind fields
    fname_s : str
        path to the 1h 2D surface file containing the surface pressure
    plev : int
        index of the considered pressure level, ie index in [200, 300, 400, 500, 600, 700, 850, 925] hPa

    Returns
    -------
    zeta : 2D array
        relative vertical vorticity 2D field
    """
        
    # loading of the required variables
    with xr.open_dataset(fname_p) as dataset:
        lats = dataset.variables['lat']
        lons = dataset.variables['lon']
        pres = dataset.variables['pressure']
        u = dataset['U'][0][plev] # 2D: lat, lon
        v = dataset['V'][0][plev] # same. U, V are unstaggered
    
    with xr.open_dataset(fname_s) as dset:
        ps = dset['PS'][0] # 2D: lat, lon
    
    # select above ground level grid points
    AGL_bin = ps > pres[plev]
    
    # computation of horizontal grid spacing
    dlon = np.deg2rad(np.lib.pad(lons, ((0,0),(0,1)), mode='constant', constant_values=np.nan)[:,1:] - np.lib.pad(lons, ((0,0),(1,0)), mode='constant', constant_values=np.nan)[:,:-1])
    dlat =  np.deg2rad(np.lib.pad(lats, ((0,1),(0,0)), mode='constant', constant_values=np.nan)[1:,:] - np.lib.pad(lats, ((1,0),(0,0)), mode='constant', constant_values=np.nan)[:-1,:])
    dx = Rm*np.cos(np.deg2rad(lats))*dlon
    dy = Rm*dlat
    
    # differentiate v and u with respect to x/lon and y/lat respectively
    dv = np.lib.pad(v, ((0,0),(0,1)), mode='constant', constant_values=np.nan)[:,1:] - np.lib.pad(v, ((0,0),(1,0)), mode='constant', constant_values=np.nan)[:,:-1]
    du = np.lib.pad(u, ((0,1),(0,0)), mode='constant', constant_values=np.nan)[1:,:] - np.lib.pad(u, ((1,0),(0,0)), mode='constant', constant_values=np.nan)[:-1,:]
    
    # finally relative vertical vorticity
    zeta = np.array(dv/dx - du/dy)
    
    # set below ground level values to nans
    zeta = np.where(AGL_bin, zeta, np.nan)
    
    return zeta


# function computing updraft helicity field on a certain pressure level given the wind field contained in a single time shot file
def UH_plev(fname_p, plev):
    #input: fname_p (str): complete file path containing the wind fields (3D)
    #       fname_s (str): complete file path containing the surface pressure (2D)
    #       plev (int): pressure level considered, index in [200, 300, 400, 500, 600, 700, 850, 925] hPa
    #output: UH (2D array)
    
    # definition of the required variables
    with xr.open_dataset(fname_p) as dataset:
        lats = dataset.variables['lat']
        lons = dataset.variables['lon']
        u = dataset['U'][0][plev] # 2D: lat, lon
        v = dataset['V'][0][plev] # same. U, V and W are unstaggered
        w = dataset['W'][0][plev]
    
    # select updrafts only, the rest goes to 0
    wbin = w > 0
    w_updraft = wbin*w
    
    # computation of horizontal grid spacing
    dlon = np.deg2rad(np.lib.pad(lons, ((0,0),(0,1)), mode='constant', constant_values=np.nan)[:,1:] - np.lib.pad(lons, ((0,0),(1,0)), mode='constant', constant_values=np.nan)[:,:-1])
    dlat =  np.deg2rad(np.lib.pad(lats, ((0,1),(0,0)), mode='constant', constant_values=np.nan)[1:,:] - np.lib.pad(lats, ((1,0),(0,0)), mode='constant', constant_values=np.nan)[:-1,:])
    dx = Rm*np.cos(np.deg2rad(lats))*dlon
    dy = Rm*dlat
    
    # differentiate v and u with respect to x/lon and y/lat respectively
    dv = np.lib.pad(v, ((0,0),(0,1)), mode='constant', constant_values=np.nan)[:,1:] - np.lib.pad(v, ((0,0),(1,0)), mode='constant', constant_values=np.nan)[:,:-1]
    du = np.lib.pad(u, ((0,1),(0,0)), mode='constant', constant_values=np.nan)[1:,:] - np.lib.pad(u, ((1,0),(0,0)), mode='constant', constant_values=np.nan)[:-1,:]
    
    # finally vertical vorticity and updraft helicity
    zeta = dv/dx - du/dy
    uh = zeta*w_updraft
    
    return uh


# function computing the integrated updraft helicity (IUH) from 775 to 550 hPa
def IUH(fname_p, fname_s):
    #input: fname_p (str): complete file path containing the wind fields (3D)
    #       fname_s (str): complete file path containing the surface pressure (2D)
    #output: IUH (2D array)
    
    # definition of the required variables
    with xr.open_dataset(fname_p) as dset:
        pres = np.array(dset.variables['pressure'])
        iuh = np.zeros(np.shape(dset.variables['lon']))
        temp = np.array(dset['T'][0]) # 3D : (pres, lat, lon)
    
    with xr.open_dataset(fname_s) as dset:
        ps = dset['PS'][0] # 2D: lat, lon
    
    # select grid points on the 700 hPa isobar lying above surface
    above_surface_bin = ps > 70000.
    below_surface_bin = ps < 70000.
    
    # integration over the 2 pressure levels, using hydrostatic approximation and ideal gas law
    
    # intermediate layer (550-650 hPa) common to all grid points
    uh = UH_plev(fname_p, 4)
    dp =  (pres[5] - pres[3])/2
    iuh = iuh + uh*Ra*temp[4]*dp/(g*pres[4]) # dz=-Ra*T*dp/(g*p) (here with positive sign because we reverted the integral bounds)
    
    # regular lower layer (650-775 hPa) for above surface grid points on the 700 hPa isobar (large majority of them)
    uh_reg = UH_plev(fname_p, 5)
    dp =  (pres[6] - pres[4])/2
    iuh_reg = iuh + uh_reg*Ra*temp[5]*dp/(g*pres[5])
    iuh_reg = iuh_reg*above_surface_bin # select the regular grid points and leave the others to 0
    
    # upper layer (425-550 hPa) for points lying below surface on the 700 hPa isobar
    uh_irreg = UH_plev(fname_p, 3)
    dp =  55000. - 42500. # the layer depth in extended to match the regular integration depth
    iuh_irreg = iuh + uh_irreg*Ra*temp[3]*dp/(g*pres[3])
    iuh_irreg = iuh_irreg*below_surface_bin # select the irregular grid points and leave the others to 0
    
    # fill in the holes so that IUH is defined at every single grid point; discard the NaNs on the edges and transform into an numpy array
    return np.array(iuh_reg + iuh_irreg) #[1:-1,1:-1] keep the orginal size, despite nans


#compute and plot the one time shot IUH 2D field, together with the precipitatin and hail fields
def plot_IUH_prec_hail(fname_p, fname_s, prec_fname, hail_fname):
    #input: fname (str): complete file path (a single time shot)
    #output: plot of the 2D IUH
    
    dtstr = fname_p[-18:-4] #adjust depending on the filename format !
    dtobj = datetime.strptime(dtstr, "%Y%m%d%H%M%S")
    dtdisp = dtobj.strftime("%d/%m/%Y %H:%M:%S")
    
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    
    # static fields
    with xr.open_dataset(fname_s) as dset:
        lats = dset.variables['lat']
        lons = dset.variables['lon']
    
    # IUH data
    iuh = np.array(IUH(fname_p, fname_s))
    iuh[abs(iuh)<50] = np.nan # mask regions of very small IUH to smoothen the background
    iuh_max = 150 # set here the maximum (or minimum in absolute value) IUH that you want to display
    norm_iuh = TwoSlopeNorm(vmin=-iuh_max, vcenter=0, vmax=iuh_max)
    #levels_iuh = np.linspace(-iuh_max, iuh_max, 23)
    ticks_iuh = np.arange(-iuh_max, iuh_max+1, 25)
    
    # Precipitation data
    with xr.open_dataset(prec_fname) as dset:
        prec = np.array(dset['TOT_PREC'][0])*12
    prec[prec<0.1] = np.nan # mask regions of very small precipitation to smoothen the backgroud
    prec_max = 60 # set here the maximum rain rate you want to display, threshold + prominence
    norm_prec = TwoSlopeNorm(vmin=0, vcenter=0.5*prec_max, vmax=prec_max)
    #levels_prec = np.linspace(0, prec_max, 23)
    ticks_prec = np.arange(0, prec_max+1, 5)
    
    # Hail data
    with xr.open_dataset(hail_fname) as dset:
        hail = np.array(dset['DHAIL_MX'][0])
    hail[hail<0.2] = np.nan # mask regions of very small hail to smoothen the backgroud
    hail_max = 40 # set here the maximum hail diameter you want to display
    norm_hail = TwoSlopeNorm(vmin=0, vcenter=0.5*hail_max, vmax=hail_max)
    
    # plot
    fig = plt.figure(figsize=(6,12))
    
    ax = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
    ax.add_feature(ocean, linewidth=0.2)
    cont = ax.pcolormesh(lons, lats, iuh, cmap="RdBu_r", norm=norm_iuh, transform=ccrs.PlateCarree())
    plt.colorbar(cont, ticks=ticks_iuh, orientation='horizontal', label=r"IUH ($m^2/s^2$)")
    plt.title(dtdisp)
    
    ax = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
    ax.add_feature(ocean, linewidth=0.2)
    cont = ax.pcolormesh(lons, lats, prec, cmap="plasma", norm=norm_prec, transform=ccrs.PlateCarree())
    plt.colorbar(cont, ticks=ticks_prec, orientation='horizontal', label="Rain rate (mm/h)")
    
    ax = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
    ax.add_feature(ocean, linewidth=0.2)
    cont = ax.pcolormesh(lons, lats, hail, cmap="plasma", norm=norm_hail, transform=ccrs.PlateCarree())
    plt.colorbar(cont, orientation='horizontal', label="Maximum hail diameter (mm)")   
    


#compute and plot the one time shot IUH 2D field
def plot_IUH(fname_p, fname_s):
    #input: fname (str): complete file path (a single time shot)
    #output: plot of the 2D IUH,
    
    dtstr = fname_p[-18:-4] #adjust depending on the filename format !
    dtobj = datetime.strptime(dtstr, "%Y%m%d%H%M%S")
    dtdisp = dtobj.strftime("%d/%m/%Y %H:%M:%S")
    
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    
    # static fields
    with xr.open_dataset(fname_s) as dset:
        lats = dset.variables['lat']
        lons = dset.variables['lon']
    
    # IUH data
    iuh = np.array(IUH(fname_p, fname_s))
    iuh[abs(iuh)<50] = np.nan # mask regions of very small IUH to smoothen the background
    iuh_max = 150 # set here the maximum (or minimum in absolute value) IUH that you want to display
    norm = TwoSlopeNorm(vmin=-iuh_max, vcenter=0, vmax=iuh_max)
    #levels_iuh = np.linspace(-iuh_max, iuh_max, 23)
    ticks_iuh = np.arange(-iuh_max, iuh_max+1, 25)

    # plot
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
    ax.add_feature(ocean, linewidth=0.2)
    cont = plt.pcolormesh(lons, lats, iuh, cmap="RdBu_r", norm=norm, transform=ccrs.PlateCarree())
    plt.colorbar(cont, ticks=ticks_iuh, orientation='horizontal', label=r"IUH ($m^2/s^2$)")
    plt.title(dtdisp)


def plot_zeta_plev(fname_p, fname_s, plev, w_th=None, save=False):
    """
    Plots the relative vertical vorticity 2D field over a given pressure level at a single time shot

    Parameters
    ----------
    fname_p : str
        path to the 1h 3D pressure file containing the wind fields
    fname_s : str
        path to the 1h 2D surface file containing the surface pressure
    plev : int
        index of the considered pressure level, ie index in [200, 300, 400, 500, 600, 700, 850, 925] hPa
    w_th : float
        updraught velocity threshold; the vorticity field will be plotted on the w > w_th masks
    save : bool
        option to save the figure

    Returns
    -------
    plots zeta and saves the figure if requested
    """
    
    # zeta data
    zeta = zeta_plev(fname_p, fname_s, plev)
    bounds = np.array([-9, -7, -5, -3, -1, 1, 3, 5, 7, 9])*1e-3
    #norm = TwoSlopeNorm(vcenter=0)
    norm = BoundaryNorm(boundaries=bounds, ncolors=256, extend='both')    
    
    # pressure level and static fields
    with xr.open_dataset(fname_p) as dset:
        lats = dset.variables['lat']
        lons = dset.variables['lon']
        p = int(dset.variables['pressure'][plev])
        if w_th:
            w = dset['W'][0][plev]
    
    # timestamp
    dtstr = fname_p[-18:-4] #adjust depending on the filename format !
    dtobj = pd.to_datetime(dtstr, format="%Y%m%d%H%M%S")
    dtdisp = dtobj.strftime("%d/%m/%Y %H:%M:%S")
    
    # updraught velocity patches and figure title & name
    if w_th:
        wbin = np.array(w > w_th)
        zeta = wbin*zeta
        title = dtdisp + "; " + str(round(p/100)) +  " hPa; w>" + str(w_th) + " m/s"
        figname = "zeta_" + dtstr + "_" + str(round(p/100)) +  "hPa_wth" + str(w_th) + ".png"
    else:
        title = dtdisp + " ; " + str(round(p/100)) +  " hPa isobar"
        figname = "zeta_" + dtstr + "_" + str(round(p/100)) +  "hPa.png"
    
    zeta[np.abs(zeta)<0.0005] = np.nan # smoothen the background -> omit tiny values and turn masked values into nans
    
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', scale=resol, facecolor='none')
    rp = ccrs.RotatedPole(pole_longitude = -170, pole_latitude = 43)
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=rp)
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    ax.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.2)
    cont = ax.pcolormesh(lons, lats, zeta, cmap="RdBu_r", norm=norm, transform=ccrs.PlateCarree())
    cbar = plt.colorbar(cont, orientation='horizontal', label=r"$\zeta$ ($s^{-1}$)", format='%1.0e')
    formatter = ScalarFormatter(useMathText=True, useOffset=False)
    formatter.set_powerlimits((0, 0))
    cbar.ax.xaxis.set_major_formatter(formatter)
    
    plt.title(title)
    if save:
        fig.savefig(figname, dpi=300)
    
    return


def plot_w_plev(fname_p, fname_s, plev, save=False):
    """
    Plots the updraught velocity 2D field together with the vorticity contours over a given pressure level at a single time shot

    Parameters
    ----------
    fname_p : str
        path to the 1h 3D pressure file containing the wind fields
    fname_s : str
        path to the 1h 2D surface file containing the surface pressure
    plev : int
        index of the considered pressure level, ie index in [200, 300, 400, 500, 600, 700, 850, 925] hPa
    save : bool
        option to save the figure

    Returns
    -------
    plots w as well as the zeta contours and saves the figure if requested
    """   
    
    # w, pressure level and static fields
    with xr.open_dataset(fname_p) as dset:
        lats = dset.variables['lat']
        lons = dset.variables['lon']
        p = int(dset.variables['pressure'][plev])
        w = dset['W'][0][plev]
    
    with xr.open_dataset(fname_s) as dset:
        ps = dset['PS'][0]
    
    # w
    AGL_bin = ps > p
    w = np.where(AGL_bin, w, np.nan) # convert the unphysical values ot nans
    bounds_w = [0,4,5,7,9,11]
    norm_w = BoundaryNorm(boundaries=bounds_w, ncolors=256, extend='max')
    
    # zeta 
    zeta = zeta_plev(fname_p, fname_s, plev) # unphysical values already nans
    zeta[np.abs(zeta)<0.0005] = np.nan # smoothen the background -> omit tiny values and turn masked values into nans
    bounds_zeta = np.array([-4, 4])*1e-3 
    
    # timestamp
    dtstr = fname_p[-18:-4] #adjust depending on the filename format !
    dtobj = pd.to_datetime(dtstr, format="%Y%m%d%H%M%S")
    dtdisp = dtobj.strftime("%d/%m/%Y %H:%M:%S")
    
    # load geographic features
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', scale=resol, facecolor='none')
    rp = ccrs.RotatedPole(pole_longitude = -170, pole_latitude = 43)
    
    fig = plt.figure(figsize=(10,10))
    ax = plt.axes(projection=rp)
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1, linewidth=0.2)
    ax.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.2)
    cont_w = ax.pcolormesh(lons, lats, w, cmap="Blues", norm=norm_w, transform=ccrs.PlateCarree())
    ax.contour(lons, lats, zeta, bounds_zeta, colors='r', linewidths=0.8, transform=ccrs.PlateCarree())
    #ax.clabel(cont_zeta, inline=True, fontsize=8)
    plt.colorbar(cont_w, orientation='horizontal', label=r"$w$ (m/s)")
    
    plt.title(dtdisp + " ; " + str(round(p/100)) +  r" hPa isobar ; $|\zeta|=0.004\,s^{-1}$ contours")
    if save:
        figname = "w_shade_zeta_cont_" + dtstr + "_" + str(round(p/100)) +  "hPa.png"
        fig.savefig(figname, dpi=300)
    
    return


def zeta_distribution_CSs(repo_path, CS_days, CS_ranges, cuts, plev, w_th=None):
    """
    Aggregates the relative vertical vorticity values over a given pressure level and all the case studies
    option to filter vorticity values out below a given updraught velocity threshold

    Parameters
    ----------
    repo_path : str
        path to the 1h_2D and 1h_3D_plev cut case studies files
    CS_days : list of str
        days of the case studies, "YYYYmmdd"
    CS_ranges : list of integer ranges
        hourly ranges of the respective case studies, eg. np.arange(14,22)
    cuts : list of str
        cuts of the respective case studies, ie "largecut" or "swisscut"
    plev : int
        index of the considered pressure level, ie index in [200, 300, 400, 500, 600, 700, 850, 925] hPa
    w_th : float
        updraught velocity threshold; the vorticity will be considered on the w > w_th masks

    Returns
    -------
    zeta_values : flattened array of floats
        aggregated vorticity values
    """
    
    zeta_values = []
    for i, day in enumerate(CS_days):
        for h in CS_ranges[i]:
            fname_p = repo_path + cuts[i] + "_lffd" + day + str(h).zfill(2) + "0000p.nc"
            fname_s = repo_path + cuts[i] + "_PSlffd" + day + str(h).zfill(2) + "0000.nc"
            zeta = zeta_plev(fname_p, fname_s, plev)
            
            if w_th:
                with xr.open_dataset(fname_p) as dset:
                    w = dset['W'][0][plev]
                zeta = np.where(np.array(w > w_th), zeta, np.nan)
            
            zeta_values.extend(np.ndarray.flatten(zeta))
    
    return zeta_values


def w_distribution_CSs(repo_path, CS_days, CS_ranges, cuts, plev, zeta_th=None):
    """
    Aggregates the updraught velocities over a given pressure level and all the case studies
    option to filter updraught velocities out below a given relative vertical vorticity threshold

    Parameters
    ----------
    repo_path : str
        path to the 1h_2D and 1h_3D_plev cut case studies files
    CS_days : list of str
        days of the case studies, "YYYYmmdd"
    CS_ranges : list of integer ranges
        hourly ranges of the respective case studies, eg. np.arange(14,22)
    cuts : list of str
        cuts of the respective case studies, ie "largecut" or "swisscut"
    plev : int
        index of the considered pressure level, ie index in [200, 300, 400, 500, 600, 700, 850, 925] hPa
    zeta_th : float
        relative vertical vorticity threshold; the updraught velocity will be considered on the zeta > zeta_th masks

    Returns
    -------
    w_values : flattened array of floats
        aggregated updraught velocities
    """
    
    w_values = []
    for i, day in enumerate(CS_days):
        for h in CS_ranges[i]:
            
            fname_p = repo_path + cuts[i] + "_lffd" + day + str(h).zfill(2) + "0000p.nc"
            fname_s = repo_path + cuts[i] + "_PSlffd" + day + str(h).zfill(2) + "0000.nc"
            with xr.open_dataset(fname_p) as dset:
                w = np.array(dset['W'][0][plev])
            
            if zeta_th:
                zeta = zeta_plev(fname_p, fname_s, plev)
                w = np.where(np.abs(zeta)>zeta_th, w, np.nan)
            
            w = np.where(w>0, w, np.nan)
            w_values.extend(np.ndarray.flatten(w))
    
    return w_values

#================================================================================================================================
# MAIN
#================================================================================================================================
## zeta histogram ##

# repo_path = "/scratch/snx3000/mblanc/UHfiles/"
# CS_days = ['20120630', '20130727', '20130729', '20140625', '20170801', '20190610', '20190611', '20190613', '20190614',
#             '20190820', '20210620', '20210708', '20210712', '20210713']
# CS_ranges = [np.arange(14,24), np.arange(14,23), np.arange(7,16), np.arange(10,17), np.arange(18,24), np.arange(16,21), np.arange(9,17),
#               np.arange(17,20), np.arange(18,24), np.arange(13,22), np.arange(13,19), np.arange(13,17), np.arange(17,20), np.arange(11,16)]
# cuts = ['largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut',
#         'swisscut', 'largecut', 'swisscut', 'swisscut']
# plevs = [3,4,5]
# w_ths = [3,5,7,9]

# for w_th in w_ths:
#     for plev in plevs:
#         if plev == 3:
#             p = '500'
#         elif plev == 4:
#             p = '600'
#         else:
#             p = '700'
#         zeta_values = zeta_distribution_CSs(repo_path, CS_days, CS_ranges, cuts, plev, w_th=w_th)
#         # histogram
#         fig = plt.figure()
#         plt.hist(np.abs(zeta_values), bins=30, edgecolor='black', alpha=0.6)
#         plt.xlabel(r"$|\zeta|$ ($s^{-1}$)")
#         plt.ylabel("frequency")
#         plt.title(r"Aggregated case studies ; " + p + " hPa isobar ; w>" + str(w_th) + " m/s")
#         fig.savefig("zeta_hist_CSs_" + p + "hPa_wth" + str(w_th) + ".png")

#================================================================================================================================
## w histogram ##

# repo_path = "/scratch/snx3000/mblanc/UHfiles/"
# CS_days = ['20120630', '20130727', '20130729', '20140625', '20170801', '20190610', '20190611', '20190613', '20190614',
#             '20190820', '20210620', '20210708', '20210712', '20210713']
# CS_ranges = [np.arange(14,24), np.arange(14,23), np.arange(7,16), np.arange(10,17), np.arange(18,24), np.arange(16,21), np.arange(9,17),
#               np.arange(17,20), np.arange(18,24), np.arange(13,22), np.arange(13,19), np.arange(13,17), np.arange(17,20), np.arange(11,16)]
# cuts = ['largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut',
#         'swisscut', 'largecut', 'swisscut', 'swisscut']
# plevs = [3,4,5]
# zeta_ths = [0.003, 0.004, 0.005, 0.006]

# for zeta_th in zeta_ths:
#     for plev in plevs:
#         if plev == 3:
#             p = '500'
#         elif plev == 4:
#             p = '600'
#         else:
#             p = '700'
#         w_values = w_distribution_CSs(repo_path, CS_days, CS_ranges, cuts, plev, zeta_th=zeta_th)
#         # histogram
#         fig = plt.figure()
#         plt.hist(w_values, bins=30, edgecolor='black', alpha=0.6)
#         plt.xlabel(r"$w$ (m/s)")
#         plt.ylabel("frequency")
#         plt.title(r"Aggregated case studies ; " + p + r" hPa isobar ; $|\zeta|$>" + str(zeta_th) + r" $s^{-1}$")
#         fig.savefig("w_hist_CSs_" + p + "hPa_zetath" + str(zeta_th) + ".png")

#================================================================================================================================
## plot zeta on a pressure level ##

# day = "20210713" # date to be filled
# hours = np.arange(11,16) # to be filled according to the considered period of the day
# cut = "swisscut" # to be filled according to the cut type
# plev = 4

# repo_path = "/scratch/snx3000/mblanc/UHfiles/"
# for h in hours:
#     fname_p = repo_path + cut + "_lffd" + day + str(h).zfill(2) + "0000p.nc"
#     fname_s = repo_path + cut + "_PSlffd" + day + str(h).zfill(2) + "0000.nc"
#     #plot_zeta_plev(fname_p, fname_s, plev, w_th=10)
#     plot_w_plev(fname_p, fname_s, plev)

#========================================================================================================================================
## loop over every hourly time step of the case studies ##

# CS_days = ['20120630', '20130727', '20130729', '20140625', '20170801', '20190610', '20190611', '20190613', '20190614',
#             '20190820', '20210620', '20210708', '20210712', '20210713']
# CS_ranges = [np.arange(14,24), np.arange(14,23), np.arange(7,16), np.arange(10,17), np.arange(18,24), np.arange(16,21), np.arange(9,17),
#               np.arange(17,20), np.arange(18,24), np.arange(13,22), np.arange(13,19), np.arange(13,17), np.arange(17,20), np.arange(11,16)]
# cuts = ['largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut', 'largecut',
#         'swisscut', 'largecut', 'swisscut', 'swisscut']
# plevs = [3,4,5]
# #w_ths = [5,10]
# repo_path = "/scratch/snx3000/mblanc/UHfiles/"

# # mins = []
# # maxs = []
# #for w_th in w_ths:
# for plev in plevs:
#     for i, day in enumerate(CS_days):
#         for h in CS_ranges[i]:
#             fname_p = repo_path + cuts[i] + "_lffd" + day + str(h).zfill(2) + "0000p.nc"
#             fname_s = repo_path + cuts[i] + "_PSlffd" + day + str(h).zfill(2) + "0000.nc"
#             plot_w_plev(fname_p, fname_s, plev, save=True)
#             # zeta = zeta_plev(fname_p, fname_s, plev)
#             # mins.append(np.nanmin(zeta))
#             # maxs.append(np.nanmax(zeta))
# min(zeta) = -0.010068
# max(zeta) = 0.01259  
#========================================================================================================================================
## measure computing time of whole domain IUH 2D field determination ##

#fname_p = "/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_3D_plev/lffd20190614220000p.nc"
#fname_s = "/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_2D/lffd20190614220000.nc"
#plot_IUH(fname_p, fname_s)

#t1 = chrono.time()
#iuh = IUH(fname_p, fname_s)
#t2 = chrono.time()
#dur = t2 - t1
#print(dur) 3.2 s
