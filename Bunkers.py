# This scripts aims at computing the Bunkers motion of given supercells

import numpy as np
import matplotlib.pyplot as plt
from datetime import date, time, datetime
import xarray as xr
import cartopy.feature as cfeature
import cartopy.crs as ccrs


# GLOBAL VARIABLES AND FUNCTIONS
#========================================================================================================================================

# function computing the 0-6 km wind shear in magnitude and plotting the resulting shear magnitude 2D field on case studies
def shear_magnitude(fname_p, fname_s, plot=False):
    #input: fname_p (str): complete file path containing the wind fields (3D)
    #       fname_s (str): complete file path containing the surface pressure (2D)
    #       plot (bool): plotting option
    #output: 2D 0-6 km wind shear magnitude field
    #        if requested, plot
    
    dset = xr.open_dataset(fname_s)
    ps = dset['PS'][0] # 2D: lat, lon
    
    dataset = xr.open_dataset(fname_p)
    u = dataset['U'][0] # 3D: p, lat, lon
    v = dataset['V'][0] # same. U and V are unstaggered
    if plot:
        lats = dataset.variables['lat']
        lons = dataset.variables['lon']
    
    #upper bound
    u_up = (u[2] + u[3])/2 # average between 400 hPa (~7 km) and 500 hPa (~5.4 km) winds
    v_up = (v[2] + v[3])/2 # average between 400 hPa (~7 km) and 500 hPa (~5.4 km) winds
    
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
    S = np.sqrt(du**2 + dv**2)
    
    if plot:
        dtstr = fname_p[-18:-4] #adjust depending on the filename format !
        dtobj = datetime.strptime(dtstr, "%Y%m%d%H%M%S")
        dtdisp = dtobj.strftime("%d/%m/%Y %H:%M:%S")
    
        resol = '10m'  # use data at this scale
        bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
        ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
        
        plt.figure()
        ax = plt.axes(projection=ccrs.PlateCarree())
        plt.contourf(lons, lats, S, transform=ccrs.PlateCarree())
        ax.add_feature(ocean, linewidth=0.2)
        ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
        plt.colorbar(orientation='horizontal', label="0-6 km vertical wind shear magnitude (m/s)")
        plt.title(dtdisp)
    
    return S


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

fname_p = "/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_3D_plev/lffd20201231230000p.nc"
fname_s = "/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_2D/lffd20201231230000.nc"
shear_magnitude(fname_p, fname_s, True)

