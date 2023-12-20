import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy
import json
from datetime import date, time, datetime
from matplotlib.animation import FuncAnimation
from matplotlib.colors import TwoSlopeNorm

 
# hail cell masks
dset = xr.open_dataset("/scratch/snx3000/mblanc/cell_tracker/outfiles/cell_masks_20130727.nc")
print(dset)
dset.variables.keys()
dset['cell_mask'][53].plot(cmap='jet')


# gap filled swath
dset = xr.open_dataset("/store/c2sm/scclim/climate_simulations/present_day/hail_tracks/gap_filled_20190707.nc")
print(dset)
dset.variables.keys()
dset['cell_mask'][10].plot(cmap='jet')


# hail cell tracks
with open("/scratch/snx3000/mblanc/cell_tracker/outfiles/cell_tracks_20210713.json", "r") as read_file:
    dset = json.load(read_file)
    
print(dset)

#==========================================================================================================================

#1h_2D outputs -> same type as 5min_2D outputs
dset = xr.open_dataset("/scratch/snx3000/mblanc/cell_tracker/infiles/largecut_PREClffd20130729.nc")
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
day = date(2017, 8, 2) # to be filled
varout = "PREC" # to be filled, variable name in the file names
varin = "TOT_PREC" # to be filled, variable name within the netcdf files

hours = np.array(range(0,24)) # to be filled according to the output names
mins = np.array(range(0,60,5)) # to be filled according to the output names
secs = 0 # to be filled according to the output names

repo_path = "/scratch/snx3000/mblanc/" + day.strftime("%Y%m%d") + "/"
filename = "largecut_" + varout + "lffd" + day.strftime("%Y%m%d") # without .nc, to be filled according to the cut type
anim_name = day.strftime("%Y%m%d") + "_" + varout + ".mp4"

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
PREC0 = np.array(dset0[varin][0])*12
PREC0[PREC0<0.1] = np.nan # mask regions of very small precip / hail to smoothen the backgroud
lats = dset0.variables['lat']
lons = dset0.variables['lon']
dt0str = day.strftime("%Y%m%d") + alltimes[0]
dt0obj = datetime.strptime(dt0str, "%Y%m%d%H%M%S")

if varout=="PREC":
   prec_max = 60 # set here the maximum rain rate you want to display, threshold + prominencev
   norm = TwoSlopeNorm(vmin=0, vcenter=0.5*prec_max, vmax=prec_max)
   levels_prec = np.linspace(0, prec_max, 23)
   ticks_prec = np.arange(0, prec_max+1, 5)
elif varout=="HAIL" or varout =="DHAIL":
   levels = np.linspace(0, 30, 22) # adjust the number of levels at your convenience

fig = plt.figure()
ax = plt.axes(projection=ccrs.PlateCarree())
cont = plt.contourf(lons, lats, PREC0, cmap="plasma", norm=norm, levels=levels_prec, extend="max", transform=ccrs.PlateCarree())
ax.add_feature(ocean, linewidth=0.2)
ax.add_feature(lakes)
ax.add_feature(rivers, linewidth=0.2)
ax.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
if varout=="PREC":
    plt.colorbar(cont, ticks=ticks_prec, orientation='horizontal', label="Rain rate (mm/h)")
elif varout=="HAIL" or varout =="DHAIL":
    plt.colorbar(orientation='horizontal', label="Maximum hail diameter (mm)")
plt.title(dt0obj.strftime("%d/%m/%Y %H:%M:%S"))

# animation function
def animate(i):
    global cont, lats, lons, day
    dtstr = day.strftime("%Y%m%d") + alltimes[i]
    dtobj = datetime.strptime(dtstr, "%Y%m%d%H%M%S")
    dset = xr.open_dataset(allfiles[i])
    PREC = np.array(dset[varin][0])*12
    PREC[PREC<0.1] = np.nan # mask regions of very small precip / hail to smoothen the backgroud
    for c in cont.collections:
        c.remove()  # removes only the contours, leaves the rest intact
    cont = plt.contourf(lons, lats, PREC, levels=levels_prec, cmap="plasma", norm=norm, transform=ccrs.PlateCarree())
    plt.title(dtobj.strftime("%d/%m/%Y %H:%M:%S"))
    return cont

anim = FuncAnimation(fig, animate, frames=len(allfiles), repeat=False)
anim.save(anim_name)


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


#========================================================================================================================================
## checking geopotential height differences betweeen pressure levels ##

dset = xr.open_dataset("/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_3D_plev/lffd20201231230000p.nc")
Z = np.array(dset['FI'][0]/9.81)
pres = np.array(dset.variables['pressure'])

for i, p in enumerate(pres):
    print("plev=", round(p/100), "hPa: min(Z)=", np.min(Z[i]), ", mean(Z)=", np.mean(Z[i]), ", max(Z)=", np.max(Z[i]))

for i in range(7):
    print("dplev=", round(pres[i]/100), "-", round(pres[i+1]/100), "hPa: min(dZ)=", np.min(Z[i]-Z[i+1]), ", mean(dZ)=", np.mean(Z[i]-Z[i+1]), ", max(dZ)=", np.max(Z[i]-Z[i+1]))


#========================================================================================================================================
## dBZ <-> mm/h ##

# MeteoSwiss parameters
a = 316
b = 1.5
# R: rain rate in mm/h ; Lz: reflectivity in dBZ
def R(L):
    return (10**(L/10)/a)**(1/b)

def Lz(R):
    return 10*np.log10(a*R**b)
