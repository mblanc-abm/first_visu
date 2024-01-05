# This script aims at computing and plotting the IUH field along with the surface precipitation and hail fields for a given time shot,
# in order to asses how they match or deviate from each other

import numpy as np
import matplotlib.pyplot as plt
import time as chrono
from datetime import date, time, datetime
import xarray as xr
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm

# GLOBAL VARIABLES AND FUNCTIONS
#========================================================================================================================================

#global constants
Rm = 6370000. # mean Earth's radius (m)
g = 9.80665 # standard gravity at sea level (m/s^2)
Ra = 287.05 #  Individual Gas Constant for air (J/K/kg)

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
    
    return iuh_reg + iuh_irreg # fill in the holes so that IUH is defined at every single grid point


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
    levels_iuh = np.linspace(-iuh_max, iuh_max, 23)
    ticks_iuh = np.arange(-iuh_max, iuh_max+1, 25)
    
    # Precipitation data
    with xr.open_dataset(prec_fname) as dset:
        prec = np.array(dset['TOT_PREC'][0])*12
    prec[prec<0.1] = np.nan # mask regions of very small precipitation to smoothen the backgroud
    prec_max = 60 # set here the maximum rain rate you want to display, threshold + prominence
    norm_prec = TwoSlopeNorm(vmin=0, vcenter=0.5*prec_max, vmax=prec_max)
    levels_prec = np.linspace(0, prec_max, 23)
    ticks_prec = np.arange(0, prec_max+1, 5)
    
    # Hail data
    with xr.open_dataset(hail_fname) as dset:
        hail = np.array(dset['DHAIL_MX'][0])
    hail[hail<0.2] = np.nan # mask regions of very small hail to smoothen the backgroud
    hail_max = 40 # set here the maximum hail diameter you want to display
    levels_hail = np.linspace(0, hail_max, 22) # adjust the number of levels at your convenience
    
    # plot
    fig = plt.figure(figsize=(6,12))
    
    ax = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
    cont = ax.contourf(lons, lats, iuh, cmap="RdBu_r", norm=norm_iuh, levels=levels_iuh, extend="both", transform=ccrs.PlateCarree())
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
    ax.add_feature(ocean, linewidth=0.2)
    plt.colorbar(cont, ticks=ticks_iuh, orientation='horizontal', label=r"IUH ($m^2/s^2$)")
    plt.title(dtdisp)
    
    ax = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
    cont = ax.contourf(lons, lats, prec, cmap="plasma", norm=norm_prec, levels=levels_prec, extend="max", transform=ccrs.PlateCarree())
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
    ax.add_feature(ocean, linewidth=0.2)
    plt.colorbar(cont, ticks=ticks_prec, orientation='horizontal', label="Rain rate (mm/h)")
    
    ax = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
    cont = ax.contourf(lons, lats, hail, cmap="plasma", levels=levels_hail, transform=ccrs.PlateCarree())
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
    ax.add_feature(ocean, linewidth=0.2)
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
    cont = plt.pcolormesh(lons, lats, iuh, cmap="RdBu_r", norm=norm, transform=ccrs.PlateCarree())
    ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
    ax.add_feature(ocean, linewidth=0.2)
    plt.colorbar(cont, ticks=ticks_iuh, orientation='horizontal', label=r"IUH ($m^2/s^2$)")
    plt.title(dtdisp)
   

#================================================================================================================================

# MAIN
#================================================================================================================================

#import files with wind variables U, V, W of a certain day, considering switzerland
# day = date(2021, 6, 28) # date to be filled
# hours = np.array(range(13,14)) # to be filled according to the considered period of the day
# mins = 0 # to be filled according to the output names
# secs = 0 # to be filled according to the output names
# cut = "largecut" # to be filled according to the cut type

# repo_path = "/scratch/snx3000/mblanc/UHfiles/" # + day.strftime("%Y%m%d") + "/"
# filename_p = cut + "_lffd" + day.strftime("%Y%m%d") # without .nc
# filename_s = cut + "_PSlffd" + day.strftime("%Y%m%d") # without .nc
# #filename_prec = cut + "_PREClffd" + day.strftime("%Y%m%d") # without .nc
# #filename_hail = cut + "_HAILlffd" + day.strftime("%Y%m%d") # without .nc

# alltimes = [] # all times within the considered period
# for h in hours:
#     t = time(h, mins, secs)
#     alltimes.append(t.strftime("%H%M%S"))
        
# allfiles_p = [] # all files to be plotted in the directory
# allfiles_s = []
# #allfiles_prec = []
# #allfiles_hail = []
# for t in alltimes:
#     allfiles_p.append(repo_path + filename_p + t + "p.nc")
#     allfiles_s.append(repo_path + filename_s + t + ".nc")
#     #allfiles_prec.append(repo_path + filename_prec + t + ".nc")
#     #allfiles_hail.append(repo_path + filename_hail + t + ".nc")

# #plot the chosen time shots
# for i in range(len(allfiles_p)):
#     plot_IUH(allfiles_p[i], allfiles_s[i])


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
