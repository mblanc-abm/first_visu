# This scripts aims at computing the Bunkers motion of given supercells

import numpy as np
import matplotlib.pyplot as plt
from datetime import date, time, datetime
import xarray as xr
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm
from CaseStudies import IUH


# GLOBAL VARIABLES AND FUNCTIONS
#========================================================================================================================================

# function computing the 0-6 km wind shear and plotting the resulting shear magnitude and IUH 2D fields on case study domain
def vertical_wind_shear(fname_p, fname_s, plot=True, ret=False):
    #input: fname_p (str): complete file path containing the wind fields (3D)
    #       fname_s (str): complete file path containing the surface pressure (2D)
    #       plot (bool): plotting option
    #       ret (bool): wind shear returning option
    #output: 2D 0-6 km wind shear magnitude field
    #        if requested, plots the wind shear magnitude together with the IUH
    #        if requested, returns the wind shear vector
    
    dset = xr.open_dataset(fname_s)
    ps = dset['PS'][0] # 2D: lat, lon
    
    dataset = xr.open_dataset(fname_p)
    u = dataset['U'][0] # 3D: p, lat, lon
    v = dataset['V'][0] # same. U and V are unstaggered
    if plot:
        lats = dataset.variables['lat']
        lons = dataset.variables['lon']
    
    #upper bound, common to every grid point: average between 400 hPa (~7 km) and 500 hPa (~5.4 km) winds
    u_up = (u[2] + u[3])/2
    v_up = (v[2] + v[3])/2
    
    #lower bound: differs depending on topography
    surf_bin = ps > 92500. # low areas 
    mid_bin = (92500. > ps)*(ps > 85000.) # mid areas
    up_bin = ps < 85000. #elevated areas; we neglect the ~100 grid points above 750 hPa
    u_down = u[7]*surf_bin + u[6]*mid_bin + u[5]*up_bin 
    v_down = v[7]*surf_bin + v[6]*mid_bin + v[5]*up_bin
    #take resectively lower bound at 925 (~753m), 850 (~1430m), and 700 (~3000m) hPa, without averaging
    
    #vertical wind shear
    du = u_up - u_down
    dv = v_up - v_down
    S = np.array(np.sqrt(du**2 + dv**2))
    
    if plot:
        dtstr = fname_p[-18:-4] #adjust depending on the filename format !
        dtobj = datetime.strptime(dtstr, "%Y%m%d%H%M%S")
        dtdisp = dtobj.strftime("%d/%m/%Y %H:%M:%S")
    
        resol = '10m'  # use data at this scale
        bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
        ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
        
        iuh = np.array(IUH(fname_p, fname_s))
        iuh[abs(iuh)<5] = np.nan # mask regions of very small IUH to smoothen the background
        iuh_max = 170 # set here the maximum (or minimum in absolute value) IUH that you want to display
        norm = TwoSlopeNorm(vmin=-iuh_max, vcenter=0, vmax=iuh_max)
        
        
        Smax = 50 # set here the maximum shear magnitude you want to display
        Snorm = TwoSlopeNorm(vmin=0, vcenter=20, vmax=Smax)
        
        fig = plt.figure(figsize=(6,12))
        
        ax = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
        cont = ax.contourf(lons, lats, iuh, cmap="RdBu_r", norm=norm, levels=22, transform=ccrs.PlateCarree())
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
        ax.add_feature(ocean, linewidth=0.2)
        plt.colorbar(cont, orientation='horizontal', label="IUH (m^2/s^2)")
        plt.title(dtdisp)
        
        ax = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
        plt.contourf(lons, lats, S, cmap="RdBu_r", transform=ccrs.PlateCarree(), levels=22, norm=Snorm)
        ax.add_feature(ocean, linewidth=0.2)
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
        plt.colorbar(orientation='horizontal', label="0-6 km vertical wind shear magnitude (m/s)")
        
    if ret:
        return du, dv


# function computing the Bunkers deviant motion a given supercell at a single time shot
def bunkers_motion(fname_p, fname_s, SC_grid_coord):
    #input: fname_p (str): complete file path containing the wind fields (3D)
    #       fname_s (str): complete file path containing the surface pressure (2D)
    #       SC_grid_coord (str): 
    #output: 3D Bunkers velocity vectors for the right and left movers
    
    return


#================================================================================================================================

# MAIN
#================================================================================================================================

## Plot wind shear magnitude together with IUH 2D fields

#import case studies files
day = date(2019, 6, 13) # date to be filled
hours = np.array(range(17,20)) # to be filled according to the considered period of the day
mins = 0 # to be filled according to the output names
secs = 0 # to be filled according to the output names
cut = "largecut" # to be filled according to the cut type

repo_path = "/scratch/snx3000/mblanc/UHfiles/"
filename_p = cut + "_lffd" + day.strftime("%Y%m%d") # without .nc
filename_s = cut + "_PSlffd" + day.strftime("%Y%m%d") # without .nc
filename_prec = cut + "_PREClffd" + day.strftime("%Y%m%d") # without .nc
filename_hail = cut + "_HAILlffd" + day.strftime("%Y%m%d") # without .nc

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
    vertical_wind_shear(allfiles_p[i], allfiles_s[i])
