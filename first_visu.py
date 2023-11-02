import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import json
from datetime import date, time, datetime
from matplotlib.animation import FuncAnimation
 
# hail cell masks
dset = xr.open_dataset("/store/c2sm/scclim/climate_simulations/present_day/hail_tracks/cell_masks_20190707.nc")
print(dset)
dset.variables.keys()
dset['cell_mask'][100].plot(cmap='jet')


# gap filled swath
dset = xr.open_dataset("/store/c2sm/scclim/climate_simulations/present_day/hail_tracks/gap_filled_20190707.nc")
print(dset)
dset.variables.keys()
dset['DHAIL_MX'].plot(cmap='jet')


# hail cell tracks
with open("/store/c2sm/scclim/climate_simulations/present_day/hail_tracks/cell_tracks_20190707.json", "r") as read_file:
    dset = json.load(read_file)
    
print(dset)

#==========================================================================================================================

#1h_2D outputs -> same type as 5min_2D outputs
dset = xr.open_dataset("/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_2D/lffd20160309170000.nc")
print(dset)
dset.variables.keys()
dset['cell_mask'][100].plot(cmap='jet')


#1h_3D outputs
dset = xr.open_dataset("/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_3D_plev/lffd20201231230000p.nc")
print(dset)
dset.variables.keys()
dset['cell_mask'][100].plot(cmap='jet')

#==========================================================================================================================

#swiss_cut hail visualisation
dset = xr.open_dataset("/scratch/snx3000/mblanc/20210713/swisscut_DHAILlffd20210713124000.nc")
#print(dset)
#dset.variables.keys()

DHAIL = dset.variables['DHAIL_MX'][0, :, :]
lats = dset.variables['lat']
lons = dset.variables['lon']

resol = '10m'  # use data at this scale
bodr = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
land = cartopy.feature.NaturalEarthFeature('physical', 'land', scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', scale=resol, edgecolor='b', facecolor=cfeature.COLORS['water'])
rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resol, edgecolor='b', facecolor='none')

ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, DHAIL, transform=ccrs.PlateCarree())
#ax.coastlines()
#ax.add_feature(land, facecolor='beige')
ax.add_feature(ocean, linewidth=0.2)
ax.add_feature(lakes)
ax.add_feature(rivers, linewidth=0.2)
ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
plt.colorbar(orientation='horizontal', label="Maximum hail diameter (mm)")
plt.show()

#==========================================================================================================================

#swiss_cut precipitation visualisation
dset = xr.open_dataset("/scratch/snx3000/mblanc/20210713/swisscut_PREClffd20210713124000.nc")
dtstr = "20210713124000"
dtobj = datetime.strptime(dtstr, "%Y%m%d%H%M%S")

PREC = dset.variables['TOT_PREC'][0, :, :]
lats = dset.variables['lat']
lons = dset.variables['lon']

resol = '10m'  # use data at this scale
bodr = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
land = cartopy.feature.NaturalEarthFeature('physical', 'land', scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', scale=resol, edgecolor='b', facecolor=cfeature.COLORS['water'])
rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resol, edgecolor='b', facecolor='none')

ax = plt.axes(projection=ccrs.PlateCarree())
plt.contourf(lons, lats, PREC, transform=ccrs.PlateCarree())
#ax.coastlines()
#ax.add_feature(land, facecolor='beige')
ax.add_feature(ocean, linewidth=0.2)
ax.add_feature(lakes)
ax.add_feature(rivers, linewidth=0.2)
ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
plt.colorbar(orientation='horizontal', label="Total precipitation amount (kg/m^2)")
plt.title(dtobj.strftime("%d/%m/%Y %H:%M:%S"))
plt.show()

#==========================================================================================================================
## TIME-LAPS ANIMATION ##

## filenames preparation

day = date(2019, 6, 15) # to be filled
varout = "PREC" # to be filled, variable name in the file names
varin = "TOT_PREC" # to be filled, variable name within the netcdf files

repo_path = "/scratch/snx3000/mblanc/" + day.strftime("%Y%m%d") + "/"
filename = "largecut_" + varout + "lffd" + day.strftime("%Y%m%d") # without .nc

hours = np.array(range(0,24)) # to be filled according to the output names
mins = np.array(range(0,60,5)) # to be filled according to the output names
secs = 0 # to be filled according to the output names

alltimes = [] # all times within a day, by steps of 5 min
for h in hours:
    for m in mins:
        t = time(h, m, secs)
        alltimes.append(t.strftime("%H%M%S"))
        
allfiles = [] # all files to be plotted in the directory
for t in alltimes:
    allfiles.append(repo_path + filename + t + ".nc")

## animated plot ##

# load geographic features
resol = '10m'  # use data at this scale
bodr = cartopy.feature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.7)
land = cartopy.feature.NaturalEarthFeature('physical', 'land', scale=resol, edgecolor='k', facecolor=cfeature.COLORS['land'])
ocean = cartopy.feature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])
lakes = cartopy.feature.NaturalEarthFeature('physical', 'lakes', scale=resol, edgecolor='b', facecolor=cfeature.COLORS['water'])
rivers = cartopy.feature.NaturalEarthFeature('physical', 'rivers_lake_centerlines', scale=resol, edgecolor='b', facecolor='none')

# first image on screen
dset0 = xr.open_dataset(allfiles[0])
PREC0 = dset0.variables[varin][0]
lats = dset0.variables['lat']
lons = dset0.variables['lon']
dt0str = day.strftime("%Y%m%d") + alltimes[0]
dt0obj = datetime.strptime(dt0str, "%Y%m%d%H%M%S")

# find variable maximum over the whole area and considered period
maxs = []
for fpath in allfiles:
    dset = xr.open_dataset(fpath)
    var = dset.variables[varin][0, :, :]
    maxs.append(float(np.max(var)))
MAX = max(maxs)
levels = np.linspace(0, MAX, 9) # adjust the number of levels at your convenience

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cont = plt.contourf(lons, lats, PREC0, levels=levels, transform=ccrs.PlateCarree())
ax.add_feature(ocean, linewidth=0.2)
ax.add_feature(lakes)
ax.add_feature(rivers, linewidth=0.2)
ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
if varout=="PREC":
    plt.colorbar(orientation='horizontal', label="Total precipitation amount (kg/m^2)")
elif varout=="HAIL" or varout =="DHAIL":
    plt.colorbar(orientation='horizontal', label="Maximum hail diameter (mm)")
plt.title(dt0obj.strftime("%d/%m/%Y %H:%M:%S"))

# animation function
def animate(i):
    global cont, lats, lons, day
    dtstr = day.strftime("%Y%m%d") + alltimes[i]
    dtobj = datetime.strptime(dtstr, "%Y%m%d%H%M%S")
    dset = xr.open_dataset(allfiles[i])
    PREC = dset.variables[varin][0, :, :]
    for c in cont.collections:
        c.remove()  # removes only the contours, leaves the rest intact
    cont = plt.contourf(lons, lats, PREC, levels=levels, transform=ccrs.PlateCarree())
    plt.title(dtobj.strftime("%d/%m/%Y %H:%M:%S"))
    return cont

anim = FuncAnimation(fig, animate, frames=len(allfiles), repeat=False)
anim.save('20190615_PREC.mp4')


#========================================================================================================================================
## checking dx and dy variation ##

Rm = 6370000
dset = xr.open_dataset("/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_2D/lffd20160309170000.nc")
lats = dset.variables['lat']
lons = dset.variables['lon']

dlon = np.deg2rad(np.lib.pad(lons, ((0,0),(0,1)), mode='constant', constant_values=np.nan)[:,1:] - np.lib.pad(lons, ((0,0),(1,0)), mode='constant', constant_values=np.nan)[:,:-1])
dlat =  np.deg2rad(np.lib.pad(lats, ((0,1),(0,0)), mode='constant', constant_values=np.nan)[1:,:] - np.lib.pad(lats, ((1,0),(0,0)), mode='constant', constant_values=np.nan)[:-1,:])
dx = np.array(Rm*np.cos(np.deg2rad(lats))*dlon)
dy = np.array(Rm*dlat)

print("min(dx)=", np.nanmin(dx))
print("mean(dx)=", np.nanmean(dx))
print("max(dx)=", np.nanmax(dx))
print("min(dy)=", np.nanmin(dy))
print("mean(dy)=", np.nanmean(dy))
print("max(dy)=", np.nanmax(dy))


#========================================================================================================================================
## checking beneath surface pressure levels ##

dset = xr.open_dataset("/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_3D_plev/lffd20210712190000p.nc")
pres = np.array(dset.variables['pressure'])
dset = xr.open_dataset("/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_2D/lffd20130712190000.nc")
ps = np.array(dset['PS'][0])

pbin = []
for p in pres:
    pbin.append(ps < p)

for i in range(np.shape(pbin)[0]):
    print("p=", pres[i], ": ", np.sum(pbin[i]), " grid points beneath surface (where p>ps)")

print("out of", np.size(pbin[0]))