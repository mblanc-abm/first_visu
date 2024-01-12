# this script aim at plotting supercells trajectories
import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


#data to be filled
day = "20210620"
cut = "swisscut"

# SC tracks
with open("/scratch/snx3000/mblanc/CS_tracker/output/supercell_" + day + ".json", "r") as read_file:
    SC_info = json.load(read_file)['SC_data']
SC_ids = [SC_info[i]['rain_cell_id'] for i in range(len(SC_info))]

# rain tracks
with open("/scratch/snx3000/mblanc/cell_tracker/outfiles/cell_tracks_" + day + ".json", "r") as read_file:
    rain_tracks = json.load(read_file)['cell_data']
ncells = len(rain_tracks)
#among the supercells, select their linked cells, namely parents, childs, merged to cells
# ids_to_add = []
# for sc_id in SC_ids:
#     if rain_tracks[sc_id]['parent']:
#         ids_to_add.append(rain_tracks[sc_id]['parent'])
#     elif rain_tracks[sc_id]['merged_to']:
#         ids_to_add.append(rain_tracks[sc_id]['merged_to'])
#     elif rain_tracks[sc_id]['child']:
#         ids_to_add.append(rain_tracks[sc_id]['child'][0])
# # and add them to ids to consider for supercell activity
# SC_ids.extend(ids_to_add)

# restrict rain tracks to the considered supercells / cells
rain_tracks = np.array(rain_tracks)
SC_tracks = rain_tracks[np.isin(np.arange(ncells), SC_ids)]


## open full observational dataset
usecols = ['ID','time','mesostorm','mesohailstorm','lon','lat','area','vel_x','vel_y','altitude','slope','max_CPC','mean_CPC','max_MESHS','mean_MESHS','p_radar','p_dz','p_z_0','p_z_100','p_v_mean','p_d_mean','n_radar','n_dz','n_z_0','n_z_100','n_v_mean','n_d_mean']
fullset = pd.read_csv("/scratch/snx3000/mblanc/observations/Full_dataset_thunderstorm_types.csv", sep=';', usecols=usecols)
fullset['time'] = pd.to_datetime(fullset['time'], format="%Y%m%d%H%M")

#selection based on a given ID
sel_ids = 2021062010000041
selection = fullset[np.isin(fullset['ID'], sel_ids)]


## plot of the trajectories

# load geographic features
resol = '10m'  # use data at this scale
bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])

infile = "/scratch/snx3000/mblanc/cell_tracker/infiles/" + cut + "_PREClffd" + day + ".nc" #file with the prec
with xr.open_dataset(infile) as dset:
    lats = dset.variables['lat'] # 2D matrix
    lons = dset.variables['lon'] # 2D matrix

# figure
fig = plt.figure(figsize=(6,8))
fig.suptitle("test")

ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax1.plot(lons, lats, mask[0], levels=levels_mask, cmap=cmap_mask, transform=ccrs.PlateCarree())
ax1.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
ax1.add_feature(ocean, linewidth=0.2)
cbar = plt.colorbar(cont_mask, ticks=ticks, orientation='horizontal', label="Cell mask")
cbar.locator = MultipleLocator(base=disp_label_base)

ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
cont_prec = ax2.pcolormesh(lons, lats, prec[0], cmap=cmap_prec, norm=norm_prec, transform=ccrs.PlateCarree())
ax1.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
ax1.add_feature(ocean, linewidth=0.2)
plt.colorbar(cont_prec, orientation='horizontal', label='Rain rate (mm/h); supercells marked with brown contour')