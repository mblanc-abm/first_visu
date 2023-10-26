import numpy as np
import matplotlib.pyplot as plt
from datetime import date, time, datetime
import xarray as xr
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm

# GLOBAL VARIABLES AND FUNCTIONS
#========================================================================================================================================

Rm = 6370000. # mean Earth's radius
g = 9.80665 # standard gravity at sea level

# function computing the distance from Earth's centre depending on the latitude and altitude
# calculations and values taken from https://rechneronline.de/earth-radius/
def R(lat, alt):
    #input: lat (degree): latitude
    #       alt (meter): altitude above sea level
    #output: distance from Earth's centre in meter
    R_equ = 6378137.
    R_pole = 6356752.
    lat = np.deg2rad(lat)
    rs = ((R_equ**2 * np.cos(lat))**2 + (R_pole**2 * np.sin(lat))**2)/((R_equ * np.cos(lat))**2 + (R_pole * np.sin(lat))**2)
    # sqrt(rs) is the Earth's radius at sea level at latitude lat
    return np.sqrt(rs) + alt


# function computing vertical vorticity (zeta) and updraft helicity fields on a certain pressure level given
# the wind field contained in a single time shot file
def zeta_plev(fname, plev, R_const=True, alt0=False, np_grad=False):
    #input: fname (str): complete file path
    #       plev (int): pressure level considered, index in [200, 300, 400, 500, 600, 700, 850, 925] hPa
    #output: zeta (2D array), UH (2D array)
    dataset = xr.open_dataset(fname)
    lats = dataset.variables['lat']
    lons = dataset.variables['lon']
    #pres = dataset.variables['pressure']
    Z = dataset.variables['FI'][0][plev]/g # geopotental height, 2D: (lat,lon). We assume here that geop height approximates well altitude
    nlat, nlon = np.shape(lats)
    u = dataset['U'][0][plev] # 2D: lat, lon
    v = dataset['V'][0][plev] # same. ARE U, V and W staggered ?? here I assume them unstaggered
    #w = dataset['W'][0][plev]
    
    # Computation of the vertical vorticity and instataneous updraft (w>0) helicity.
    # For simplification, we omit the values at the boundaries, leaving them to 0.
    zeta = np.zeros(np.shape(lats))
    #uh = np.zeros(np.shape(lats))
    for i in range(1, nlat-1):
        for j in range(1, nlon-1):
            dlon = np.deg2rad(lons[i,j+1] - lons[i,j-1])
            dlat =  np.deg2rad(lats[i+1,j] - lats[i-1,j])
            
            if R_const and alt0:
                r = Rm
            elif R_const and not alt0:
                r = Rm + Z[i,j]
            else:
                r = R(lats[i,j], Z[i,j])
            
            dx = r*np.cos(np.deg2rad(lats[i,j]))*dlon
            dy = r*dlat
            zet = (v[i,j+1]-v[i,j-1])/dx - (u[i+1,j]-u[i-1,j])/dy
            zeta[i,j] = zet
            
            #if w[i,j] > 0:
            #    uh[i,j] = zet*w[i,j]
    
    return zeta#, uh


# function computing the root mean square error of a matrix with respect to a reference matrix of the same shape
def RMSE(M, Mref):
    return np.sqrt(np.sum((M-Mref)**2)/np.size(M))

#================================================================================================================================

# MAIN
#================================================================================================================================

fname = "/scratch/snx3000/mblanc/UHfiles/swisscut_lffd20210713120000p.nc"
plev = 4

zeta_Rconst_alt0 = zeta_plev(fname, plev, True, True)
zeta_Rconst_alt = zeta_plev(fname, plev, True, False)
zeta_R_alt = zeta_plev(fname, plev, False, False)

rmse_Rconst_alt0 = RMSE(zeta_Rconst_alt0, zeta_R_alt) #1.72e-7
rmse_Rconst_alt = RMSE(zeta_Rconst_alt, zeta_R_alt) # 4.61e-7
