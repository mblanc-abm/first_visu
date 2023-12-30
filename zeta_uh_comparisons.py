# This script aims at comparing the influence of the latitudinal variation of the Earth's radius and the altitulde upon the vertical vorticity,
# as well as comparing the computational costs of the differentiation methods

import numpy as np
import matplotlib.pyplot as plt
import time
import xarray as xr
#import cartopy.feature as cfeature
#import cartopy.crs as ccrs
#from matplotlib.colors import TwoSlopeNorm

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


# function computing vertical vorticity field on a certain pressure level given the wind field contained in a single time shot file
def zeta_plev(fname, plev, R_const=True, alt0=False, np_grad=False):
    #input: fname (str): complete file path
    #       plev (int): pressure level considered, index in [200, 300, 400, 500, 600, 700, 850, 925] hPa
    #output: zeta (2D array), UH (2D array)
    
    with xr.open_dataset(fname) as dataset:
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



# function computing vertical vorticity field on a certain pressure level given the wind field contained in a single time shot file
def zeta_grad_plev(fname, plev, npgrad=True, matr=False):
    #input: fname (str): complete file path
    #       plev (int): pressure level considered, index in [200, 300, 400, 500, 600, 700, 850, 925] hPa
    #output: UH (2D array)
    
    with xr.open_dataset(fname) as dataset:
        lats = dataset.variables['lat']
        lons = dataset.variables['lon']
        #pres = dataset.variables['pressure']
        #Z = dataset.variables['FI'][0][plev]/g # geopotental height, 2D: (lat,lon). We assume here that geop height approximates well altitude
        u = dataset['U'][0][plev] # 2D: lat, lon
        v = dataset['V'][0][plev] # same. ARE U, V and W staggered ?? here I assume them unstaggered
        #w = dataset['W'][0][plev]
    
    if npgrad or matr:
        dlon = np.deg2rad(np.lib.pad(lons, ((0,0),(0,1)), mode='constant', constant_values=np.nan)[:,1:] - np.lib.pad(lons, ((0,0),(1,0)), mode='constant', constant_values=np.nan)[:,:-1])
        dlat =  np.deg2rad(np.lib.pad(lats, ((0,1),(0,0)), mode='constant', constant_values=np.nan)[1:,:] - np.lib.pad(lats, ((1,0),(0,0)), mode='constant', constant_values=np.nan)[:-1,:])
        dx = Rm*np.cos(np.deg2rad(lats))*dlon
        dy = Rm*dlat
    
    if npgrad:  
        zeta = np.gradient(v, axis=1)/dx - np.gradient(u, axis=0)/dy
    elif matr:
        dv = np.lib.pad(v, ((0,0),(0,1)), mode='constant', constant_values=np.nan)[:,1:] - np.lib.pad(v, ((0,0),(1,0)), mode='constant', constant_values=np.nan)[:,:-1]
        du = np.lib.pad(u, ((0,1),(0,0)), mode='constant', constant_values=np.nan)[1:,:] - np.lib.pad(u, ((1,0),(0,0)), mode='constant', constant_values=np.nan)[:-1,:]
        zeta = dv/dx - du/dy
    else:
        nlat, nlon = np.shape(lats) # numbers of latitudes and longitudes
        zeta = np.zeros(np.shape(lats))
        for i in range(1, nlat-1):
            for j in range(1, nlon-1):
                dlon = np.deg2rad(lons[i,j+1] - lons[i,j-1])
                dlat =  np.deg2rad(lats[i+1,j] - lats[i-1,j])
                dx = Rm*np.cos(np.deg2rad(lats[i,j]))*dlon
                dy = Rm*dlat
                zeta[i,j] = (v[i,j+1]-v[i,j-1])/dx - (u[i+1,j]-u[i-1,j])/dy
                    
    return zeta


#================================================================================================================================

# MAIN
#================================================================================================================================

# 1) compare the influence of the latitudinal variation of the Earth's radius and the altitulde upon the vertical vorticity
fname = "/scratch/snx3000/mblanc/UHfiles/swisscut_lffd20210713140000p.nc"
p = 4

rmse_Rconst_alt0 = []
rmse_Rconst_alt = []
for p in range(8):
    zeta_Rconst_alt0 = zeta_plev(fname, p, True, True)
    zeta_Rconst_alt = zeta_plev(fname, p, True, False)
    zeta_R_alt = zeta_plev(fname, p, False, False)
    rmse_Rconst_alt0.append(RMSE(zeta_Rconst_alt0, zeta_R_alt))
    rmse_Rconst_alt.append(RMSE(zeta_Rconst_alt, zeta_R_alt))

print(np.mean(rmse_Rconst_alt0)) #3.24624542107405e-07
print(np.mean(rmse_Rconst_alt)) #3.52677091439429e-07

# compare the computatial time of the 3 methods
t1 = time.time()
zeta_Rconst_alt0 = zeta_plev(fname, p, True, True)
t2 = time.time()
dt_Rconst_alt0 = t2 - t1 # 243.8

t1 = time.time()
zeta_Rconst_alt = zeta_plev(fname, p, True, False)
t2 = time.time()
dt_Rconst_alt = t2 - t1 # 241.8

t1 = time.time()
zeta_R_alt = zeta_plev(fname, p, False, False)
t2 = time.time()
dt_R_alt = t2 - t1 #268.4


# 2) compare results and computatinal times between np.gradient, matrix and for loop differentiation methods
fname = "/scratch/snx3000/mblanc/UHfiles/swisscut_lffd20210712160000p.nc"
plev = 4
nlev = 21

t1 = time.time()
zetgrad = zeta_grad_plev(fname, plev, npgrad=True, matr=False)
t2 = time.time()
dt_grad = t2 - t1 # 5.34s / 0.07s

t1 = time.time()
zetmat = zeta_grad_plev(fname, plev, npgrad=False, matr=True)
t2 = time.time()
dt_mat = t2 - t1 # 0.022s / 0.02s

t1 = time.time()
zetloop = zeta_grad_plev(fname, plev, npgrad=False, matr=False)
t2 = time.time()
dt_loop = t2 - t1 # 250s / 247s

print(dt_grad, dt_mat, dt_loop)


fig, (ax0, ax1, ax2) = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(12, 6))

im = ax0.imshow(zetgrad)
plt.colorbar(im, ax=ax0, orientation='horizontal', label="Vertical vorticity (1/s)")
ax0.title.set_text('np.gradient')

im = ax1.imshow(zetmat)
plt.colorbar(im, ax=ax1, orientation='horizontal', label="Vertical vorticity (1/s)")
ax1.title.set_text('matrix diff.')

im = ax2.imshow(zetloop)
plt.colorbar(im, ax=ax2, orientation='horizontal', label="Vertical vorticity (1/s)")
ax2.title.set_text('loop diff.')


print(RMSE(zetgrad, zetloop), RMSE(zetmat, zetloop), RMSE(zetmat, zetgrad)) # 4.0138e-4 / 4.13e-11 / 4.0138e-4