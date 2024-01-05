# This scripts aims at computing the Bunkers motions fields

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import date, time
import xarray as xr
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm
from scipy.signal import convolve2d
from skimage.morphology import disk
from CaseStudies import IUH


# GLOBAL VARIABLES AND FUNCTIONS
#========================================================================================================================================


# function computing the 0-6 km wind shear at every grid point, plotting the resulting shear magnitude and vector
# as well as the IUH 2D fields on case study domain
def wind_shear(fname_p, fname_s, plot=True, ret=False):
    #input: fname_p (str): complete file path containing the wind fields (3D)
    #       fname_s (str): complete file path containing the surface pressure (2D)
    #       plot (bool): plotting option
    #       ret (bool): wind shear returning option
    #output: 2D 0-6 km wind shear field
    #        if requested, plots the wind shear magnitude together with the IUH
    #        if requested, returns the wind shear vector
    
    with xr.open_dataset(fname_s) as dset:
        ps = dset['PS'][0] # 2D: lat, lon
    
    with xr.open_dataset(fname_p) as dataset:
        u = dataset['U'][0] # 3D: p, lat, lon
        v = dataset['V'][0] # same. U and V are unstaggered
        if plot:
            lats = dataset.variables['lat']
            lons = dataset.variables['lon']
    
    #upper bound, common to every grid point: average between 400 hPa (~7 km) and 500 hPa (~5.4 km) winds
    #u_up = (u[2] + u[3])/2
    #v_up = (v[2] + v[3])/2
    
    #lower bound: differs depending on topography
    surf_bin = ps > 92500. # low areas 
    mid_bin = (92500. > ps)*(ps > 85000.) # mid areas
    up_bin = ps < 85000. #elevated areas; we neglect the ~100 grid points above 750 hPa
    u_down = u[7]*surf_bin + u[6]*mid_bin + u[5]*up_bin 
    v_down = v[7]*surf_bin + v[6]*mid_bin + v[5]*up_bin
    #take resectively lower bound at 925 (~753m), 850 (~1430m), and 700 (~3000m) hPa, without averaging
    
    #upper bound: also depends on topography -> keep a consistent height range
    u_up = 0.5*(u[2]+u[3])*surf_bin + u[2]*mid_bin + u[1]*up_bin
    v_up = 0.5*(v[2]+v[3])*surf_bin + v[2]*mid_bin + v[1]*up_bin
    #take resectively upper bound at 400-500 hPa (~7-5.4 km) average at ~6.2 km -> height range ~5.45 km
    #                                400 hPa (~7 km) -> height range ~5.57 km
    #                                300 hPa (~8.9 km) -> height range ~5.96 km
    
    #vertical wind shear
    du = u_up - u_down
    dv = v_up - v_down
    
    if plot:
        dtstr = fname_p[-18:-4] #adjust depending on the filename format !
        dtobj = pd.to_datetime(dtstr, format="%Y%m%d%H%M%S")
        dtdisp = dtobj.strftime("%d/%m/%Y %H:%M:%S")
            
        resol = '10m'  # use data at this scale
        bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
        ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
        
        iuh = np.array(IUH(fname_p, fname_s))
        iuh[abs(iuh)<50] = np.nan # mask regions of very small IUH to smoothen the background
        iuh_max = 150 # set here the maximum (or minimum in absolute value) IUH that you want to display
        norm_iuh = TwoSlopeNorm(vmin=-iuh_max, vcenter=0, vmax=iuh_max)
        #levels_iuh = np.linspace(-iuh_max, iuh_max, 23)
        ticks_iuh = np.arange(-iuh_max, iuh_max+1, 25)
        
        S = np.array(np.sqrt(du**2 + dv**2)) #wind shear magnitude field
        Smax = 50 # set here the maximum shear magnitude you want to display
        Snorm = TwoSlopeNorm(vmin=0, vcenter=20, vmax=Smax)
        skip = 10 #display arrow every skip grid point, for clarity
        
        fig = plt.figure(figsize=(6,12))
        
        ax = fig.add_subplot(3, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
        ax.add_feature(ocean, linewidth=0.2)
        cont = ax.pcolormesh(lons, lats, iuh, cmap="RdBu_r", norm=norm_iuh, transform=ccrs.PlateCarree())
        plt.colorbar(cont, orientation='horizontal', ticks=ticks_iuh, label=r"IUH ($m^2/s^2$)")
        plt.title(dtdisp)
        
        ax = fig.add_subplot(3, 1, 2, projection=ccrs.PlateCarree())
        ax.add_feature(ocean, linewidth=0.2)
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
        plt.pcolormesh(lons, lats, S, cmap="RdBu_r", transform=ccrs.PlateCarree(), norm=Snorm)
        plt.colorbar(orientation='horizontal', label="0-6 km vertical wind shear magnitude (m/s)")
        
        ax = fig.add_subplot(3, 1, 3, projection=ccrs.PlateCarree())
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
        ax.add_feature(ocean, linewidth=0.2)
        plt.quiver(lons[::skip,::skip], lats[::skip,::skip], du[::skip,::skip], dv[::skip,::skip], transform=ccrs.PlateCarree())
                

    if ret:
        return np.array([du, dv])



# function computing the 0-6 km mean wind at every grid point 
def mean_wind(fname_p, fname_s):
    #input: fname_p (str): complete file path containing the wind fields (3D)
    #       fname_s (str): complete file path containing the surface pressure (2D)
    #output: 2D 0-6 km mean wind vector
    
    # surface pressure data
    with xr.open_dataset(fname_s) as dset:
        ps = np.array(dset['PS'][0]) # 2D: lat, lon
    
    # wind field data
    with xr.open_dataset(fname_p) as dataset:
        u = np.array(dataset['U'][0]) # 3D: p, lat, lon
        v = np.array(dataset['V'][0]) # same. U and V are unstaggered
    
    # both bounds differ depending on topography
    surf_bin = ps > 92500. # low areas
    mid_bin = (92500. > ps)*(ps > 85000.) # mid areas
    up_bin = ps < 85000. #elevated areas; we neglect the ~100 grid points above 750 hPa
    
    #same bouds as for the wind shear
    u_surf = (u[7] + u[6] + u[5] + u[4] + u[3] + 0.5*(u[2]+u[3]))/6
    u_mid = (u[6] + u[5] + u[4] + u[3] + u[2])/5
    u_up = (u[5] + u[4] + u[3] + u[2] + u[1])/5
    u_mean = surf_bin*u_surf + mid_bin*u_mid + up_bin*u_up
    
    v_surf = (v[7] + v[6] + v[5] + v[4] + v[3] + 0.5*(v[2]+v[3]))/6
    v_mid = (v[6] + v[5] + v[4] + v[3] + v[2])/5
    v_up = (v[5] + v[4] + v[3] + v[2] + v[1])/5
    v_mean = surf_bin*v_surf + mid_bin*v_mid + up_bin*v_up
    
    return np.array([u_mean, v_mean])
    
     


# function computing the Bunkers deviant motions vectors (right and left movers) for every grid point
def bunkers_motion_raw(fname_p, fname_s, plot=True, ret=False):
    #input: fname_p (str): complete file path containing the wind fields (3D)
    #       fname_s (str): complete file path containing the surface pressure (2D)
    #output: 2D Bunkers velocity vectors for the right and left movers (RM and LM)
    
    S = wind_shear(fname_p, fname_s, plot=False, ret=True) # 2D vector
    V_mean = mean_wind(fname_p, fname_s) # 2D vector
    D = 7.5 # magnitude of Bunkers motion deviation from the 0–6-km mean wind (m/s)
    Sk = np.array([S[1], -S[0]]) # cross product between S and k=(0,0,1), on the horizontal 2D plane
    
    V_RM = V_mean + D*Sk/np.linalg.norm(S, axis=0)
    V_LM = V_mean - D*Sk/np.linalg.norm(S, axis=0)
    
    if plot:
        with xr.open_dataset(fname_s) as dset:
            lats = dset.variables['lat']
            lons = dset.variables['lon']
        
        dtstr = fname_p[-18:-4] #adjust depending on the filename format !
        dtobj = pd.to_datetime(dtstr, format="%Y%m%d%H%M%S")
        dtdisp = dtobj.strftime("%d/%m/%Y %H:%M:%S")
            
        resol = '10m'  # use data at this scale
        bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
        ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
        skip = 10 #display arrow every skip grid point, for clarity
        
        iuh = np.array(IUH(fname_p, fname_s))
        iuh[abs(iuh)<20] = np.nan # mask regions of very small IUH to smoothen the background
        iuh_max = 150 # set here the maximum (or minimum in absolute value) IUH that you want to display
        norm_iuh = TwoSlopeNorm(vmin=-iuh_max, vcenter=0, vmax=iuh_max)
        #levels_iuh = np.linspace(-iuh_max, iuh_max, 23)
        ticks_iuh = np.arange(-iuh_max, iuh_max+1, 25)
        
        fig = plt.figure(figsize=(6,12))
        
        ax = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
        ax.add_feature(ocean, linewidth=0.2)
        cont = ax.pcolormesh(lons, lats, iuh, cmap="RdBu_r", norm=norm_iuh, transform=ccrs.PlateCarree())
        plt.quiver(lons[::skip,::skip], lats[::skip,::skip], V_RM[0][::skip,::skip], V_RM[1][::skip,::skip], transform=ccrs.PlateCarree())
        plt.colorbar(cont, orientation='horizontal', ticks=ticks_iuh, label=r"IUH ($m^2/s^2$) and right mover raw motion")
        plt.title(dtdisp)
        
        ax = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
        ax.add_feature(ocean, linewidth=0.2)
        cont = ax.pcolormesh(lons, lats, iuh, cmap="RdBu_r", norm=norm_iuh, transform=ccrs.PlateCarree())
        plt.quiver(lons[::skip,::skip], lats[::skip,::skip], V_LM[0][::skip,::skip], V_LM[1][::skip,::skip], transform=ccrs.PlateCarree())
        plt.colorbar(cont, orientation='horizontal', ticks=ticks_iuh, label=r"IUH ($m^2/s^2$) and left mover raw motion")
        
    if ret:
        return V_RM, V_LM


# function computing the Bunkers deviant motions vectors (right and left movers), convolution averaged for every grid point
def bunkers_motion(fname_p, fname_s, r_conv, z=False, skip=10, plot=True, ret=False):
    #input: fname_p (str): complete file path containing the wind fields (3D)
    #       fname_s (str): complete file path containing the surface pressure (2D)
    #output: 2D Bunkers velocity vectors for the right and left movers (RM and LM)
    
    V_RM, V_LM = bunkers_motion_raw(fname_p, fname_s, plot=False, ret=True)
    footprint = disk(r_conv)
    #the bunkers motion at every grid point is taken to be the average of the circular neighbouring raw bunkers motion
    for i in range(2):
        V_RM[i] = convolve2d(V_RM[i], footprint, mode='same')/np.sum(footprint)
        V_LM[i] = convolve2d(V_LM[i], footprint, mode='same')/np.sum(footprint)
    
    if plot:
        with xr.open_dataset(fname_s) as dset:
            lats = dset.variables['lat']
            lons = dset.variables['lon']
        
        dtstr = fname_p[-18:-4] #adjust depending on the filename format !
        dtobj = pd.to_datetime(dtstr, format="%Y%m%d%H%M%S")
        dtdisp = dtobj.strftime("%d/%m/%Y %H:%M:%S")
        if not z:
            z = r_conv # the "zoom": discards the edges in the plot to get rid of the edge effect
            figname = dtstr
        else:
            figname = dtstr + "_zoom"
            
        resol = '10m'  # use data at this scale
        bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
        ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
        
        iuh = np.array(IUH(fname_p, fname_s))
        iuh[abs(iuh)<50] = np.nan # mask regions of very small IUH to smoothen the background
        iuh_max = 150 # set here the maximum (or minimum in absolute value) IUH that you want to display
        norm_iuh = TwoSlopeNorm(vmin=-iuh_max, vcenter=0, vmax=iuh_max)
        #levels_iuh = np.linspace(-iuh_max, iuh_max, 23)
        ticks_iuh = np.arange(-iuh_max, iuh_max+1, 25)
        #bound_iuh = [-150,-125,-100,-75,-50,50,75,100,125,150]
        
        fig = plt.figure(figsize=(6,12))
        
        ax = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
        ax.add_feature(ocean, linewidth=0.2)
        cont = ax.pcolormesh(lons[z:-z,z:-z], lats[z:-z,z:-z], iuh[z:-z,z:-z], cmap="RdBu_r", norm=norm_iuh, transform=ccrs.PlateCarree())
        plt.quiver(lons[z:-z,z:-z][::skip,::skip], lats[z:-z,z:-z][::skip,::skip], V_RM[0][z:-z,z:-z][::skip,::skip], V_RM[1][z:-z,z:-z][::skip,::skip], transform=ccrs.PlateCarree())
        plt.colorbar(cont, orientation='horizontal', ticks=ticks_iuh, label=r"IUH ($m^2/s^2$) and RM motion conv. averaged with r="+str(round(2.2*r_conv,1))+"km")
        plt.title(dtdisp)
        
        ax = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
        ax.add_feature(ocean, linewidth=0.2)
        cont = ax.pcolormesh(lons[z:-z,z:-z], lats[z:-z,z:-z], iuh[z:-z,z:-z], cmap="RdBu_r", norm=norm_iuh, transform=ccrs.PlateCarree())
        plt.quiver(lons[z:-z,z:-z][::skip,::skip], lats[z:-z,z:-z][::skip,::skip], V_LM[0][z:-z,z:-z][::skip,::skip], V_LM[1][z:-z,z:-z][::skip,::skip], transform=ccrs.PlateCarree())
        plt.colorbar(cont, orientation='horizontal', ticks=ticks_iuh, label=r"IUH ($m^2/s^2$) and LM motion conv. averaged with r="+str(round(2.2*r_conv,1))+"km")
        
        plt.savefig(figname)
        
    if ret:
        return V_RM, V_LM

#================================================================================================================================
# MAIN
#================================================================================================================================

# Plot wind shear magnitude together with IUH 2D fields

#import case studies files
day = date(2021, 6, 28) # date to be filled
hours = np.array(range(13,21)) # to be filled according to the considered period of the day
mins = 0 # to be filled according to the output names
secs = 0 # to be filled according to the output names
cut = "largecut" # to be filled according to the cut type

repo_path = "/scratch/snx3000/mblanc/UHfiles/"
filename_p = cut + "_lffd" + day.strftime("%Y%m%d") # without .nc
filename_s = cut + "_PSlffd" + day.strftime("%Y%m%d") # without .nc

alltimes = [] # all times within the considered period
for h in hours:
    t = time(h, mins, secs)
    alltimes.append(t.strftime("%H%M%S"))
        
allfiles_p = [] # all files to be plotted in the directory
allfiles_s = []
for t in alltimes:
    allfiles_p.append(repo_path + filename_p + t + "p.nc")
    allfiles_s.append(repo_path + filename_s + t + ".nc")

# plot the chosen time shots
for i in range(np.size(allfiles_p)):
    bunkers_motion(allfiles_p[i], allfiles_s[i], z=80, r_conv=15, skip=7)
