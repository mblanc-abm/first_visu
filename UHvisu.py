import numpy as np
import matplotlib as cm
import matplotlib.pyplot as plt
from datetime import date, time, datetime
import xarray as xr

#import files with wind variables U, V, W of a certain day, considering switzerland
day = date(2021, 7, 12) # to be filled

repo_path = "/scratch/snx3000/mblanc/UHfiles/"
filename = "swisscut_lffd" + day.strftime("%Y%m%d") # without .nc

hours = np.array(range(12,24)) # to be filled according to the output names
mins = 0 # to be filled according to the output names
secs = 0 # to be filled according to the output names

alltimes = [] # all times within the considered period
for h in hours:
    t = time(h, mins, secs)
    alltimes.append(t.strftime("%H%M%S"))
        
allfiles = [] # all files to be plotted in the directory
for time in alltimes:
    allfiles.append(repo_path + filename + time + "p.nc")

#========================================================================================================================================

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



# characteristic length and pressure values taken from the National Oceanic and Atmospheric Administration (NOAA)
# https://en.wikipedia.org/wiki/Pressure_altitude
def pressure(z): #input: altitude in m
    p0 = 101325
    L = 8431
    return p0*np.exp(-z/L) # returns pressure in Pa



def altitude(p): # input: pressure in Pa
    L = 8431
    p0 = 101325
    return L*np.log(p0/p) # returns altitude in meters



# function computing vertical vorticity (zeta) and updraft helicity fields on a certain pressure level given
# the wind field contained in a single time shot file
def zeta_UH_plev(fname, plev):
    #input: fname (str): complete file path
    #       plev (int): pressure level considered, index in [200, 300, 400, 500, 600, 700, 850, 925] hPa
    #output: zeta (2D array), UH (2D array)
    dataset = xr.open_dataset(fname)
    lats = dataset.variables['lat']
    lons = dataset.variables['lon']
    pres = dataset.variables['pressure']
    nlat, nlon = np.shape(lats)
    u = dataset['U'][0][plev] # 2D: lat, lon
    v = dataset['V'][0][plev] # same. ARE U, V and W staggered ?? here I assume them unstaggered
    w = dataset['W'][0][plev]
    
    # computation of the vertical vorticity and instataneous updraft (w>0) helicity
    # we omit the values at the boundaries, leaving them to 0
    zeta = np.zeros(np.shape(lats))
    uh = np.zeros(np.shape(lats))
    for i in range(1, nlat-1):
        for j in range(1, nlon-1):
            dlon = np.deg2rad(lons[i,j+1] - lons[i,j-1])
            dlat =  np.deg2rad(lats[i+1,j] - lats[i-1,j])
            r = R(lats[i,j], altitude(pres[plev]))
            dx = r*np.cos(np.deg2rad(lats[i,j]))*dlon
            dy = r*dlat
            zet = (v[i,j+1]-v[i,j-1])/dx - (u[i+1,j]-u[i-1,j])/dy
            zeta[i,j] = zet
            if w[i,j] > 0:
                uh[i,j] = zet*w[i,j]
    
    return zeta, uh


#compute the integrated updraft helicity (IUH) from 775 to 550 hPa
def IUH(fname):
    L = 8431
    dset = xr.open_dataset(fname)
    pres = np.array(dset.variables['pressure'])
    iuh = np.zeros(np.shape(dset.variables['lon']))
    zeta, uh = zeta_UH_plev(fname, 4)
    dp =  (pres[5] - pres[3])/2
    iuh = iuh + uh*L*dp/pres[4]
    zeta, uh = zeta_UH_plev(fname, 5)
    dp =  (pres[6] - pres[4])/2
    iuh = iuh + uh*L*dp/pres[5]
    
    return iuh


#================================================================================================================================

# test of the function
iuh7 = IUH(allfiles[7])
iuh6 = IUH(allfiles[6])
iuh5 = IUH(allfiles[5])
iuh4 = IUH(allfiles[4])

levels = np.linspace(-140, 200, 23)
plt.figure()
plt.contourf(iuh7, cmap=plt.cm.coolwarm, levels=levels)
plt.colorbar(orientation='horizontal', label="IUH (m^2/s^2)")
plt.title("12/07/2021 - 19:00:00")
