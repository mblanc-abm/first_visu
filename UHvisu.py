import numpy as np
import matplotlib.pyplot as plt
from datetime import date, time, datetime
import xarray as xr
import cartopy.feature as cfeature
import cartopy.crs as ccrs
from matplotlib.colors import TwoSlopeNorm


# FUNCTIONS
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
    iuh = iuh + uh*L*dp/pres[4] #in our simple framework, dz=-L*dp/p (here with positive sign because we reverted the integral bounds)
    zeta, uh = zeta_UH_plev(fname, 5)
    dp =  (pres[6] - pres[4])/2
    iuh = iuh + uh*L*dp/pres[5]
    
    return iuh


#compute and plot the one time shot IUH 2D field on the Swiss map
def plot_IUH(fname, nlev):
    #input: fname (str): complete file path (a single time shot)
    #       nlev (int): number of levels in the colorbar
    #output: plot of the 2D field IUH over Switzerland
    
    dtstr = fname[-18:-4] #adjust depending on the filename format !
    dtobj = datetime.strptime(dtstr, "%Y%m%d%H%M%S")
    dtdisp = dtobj.strftime("%d/%m/%Y %H:%M:%S")
    
    resol = '10m'  # use data at this scale
    bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
    coastline = cfeature.NaturalEarthFeature('physical', 'coastline', scale=resol, facecolor='none')
    #land = cfeature.NaturalEarthFeature('physical', 'land', scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
    #ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
    #lakes = cfeature.NaturalEarthFeature('physical', 'lakes', scale=resol, edgecolor='b', facecolor=cfeature.COLORS['water'])
    #rivers = cfeature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resol, edgecolor='b', facecolor='none')
    
    dset = xr.open_dataset(fname)
    iuh = IUH(fname)
    absmax = max(np.max(iuh), abs(np.min(iuh)))
    lats = dset.variables['lat']
    lons = dset.variables['lon']
    norm = TwoSlopeNorm(vmin=-absmax, vcenter=0, vmax=absmax)
    
    plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    plt.contourf(lons, lats, iuh, cmap="RdBu_r", norm=norm, levels=nlev, transform=ccrs.PlateCarree())
    #ax.add_feature(ocean, linewidth=0.2)
    #ax.add_feature(lakes)
    #ax.add_feature(rivers, linewidth=0.2)
    ax.add_feature(bodr, linestyle='-', edgecolor='k', linewidth=0.2)
    ax.add_feature(coastline, linestyle='-', edgecolor='k', linewidth=0.4)
    plt.colorbar(orientation='horizontal', label="775-550 hPa IUH (m^2/s^2)")
    plt.title(dtdisp)

#================================================================================================================================

# MAIN
#================================================================================================================================

#import files with wind variables U, V, W of a certain day, considering switzerland
day = date(2021, 7, 12) # date to be filled

repo_path = "/scratch/snx3000/mblanc/UHfiles/"
filename = "swisscut_lffd" + day.strftime("%Y%m%d") # without .nc

hours = np.array(range(17,20)) # to be filled according to the considered period of the day
mins = 0 # to be filled according to the output names
secs = 0 # to be filled according to the output names

alltimes = [] # all times within the considered period
for h in hours:
    t = time(h, mins, secs)
    alltimes.append(t.strftime("%H%M%S"))
        
allfiles = [] # all files to be plotted in the directory
for t in alltimes:
    allfiles.append(repo_path + filename + t + "p.nc")

# plot the chosen time shots
for file in allfiles:
    plot_IUH(file, 23)
