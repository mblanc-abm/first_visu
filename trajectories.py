# this script aims at plotting supercells trajectories from both observational and model data, on a case study
# first draft of SC_trajectories.py located in SC_tracker

import json
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

#=====================================================================================================================================================

# data to be filled
day = "20190611"

# SC info
with open("/scratch/snx3000/mblanc/SDT_output/CaseStudies/supercell_" + day + ".json", "r") as read_file:
    SC_info = json.load(read_file)['supercell_data']
SC_ids_mod = [SC_info[i]['rain_cell_id'] for i in range(len(SC_info))]

# rain tracks
with open("/scratch/snx3000/mblanc/cell_tracker/CaseStudies/outfiles/cell_tracks_" + day + ".json", "r") as read_file:
    rain_tracks = json.load(read_file)['cell_data']
ncells = len(rain_tracks)
# among the supercells, select their linked cells, namely parents, childs, merged to cells
# ids_to_add = []
# for sc_id in SC_ids_mod:
#     if rain_tracks[sc_id]['parent']:
#         ids_to_add.append(rain_tracks[sc_id]['parent'])
#     elif rain_tracks[sc_id]['merged_to']:
#         ids_to_add.append(rain_tracks[sc_id]['merged_to'])
#     elif rain_tracks[sc_id]['child']:
#         ids_to_add.append(rain_tracks[sc_id]['child'][0])
# # and add them to ids to consider for supercell activity
# SC_ids_mod.extend(ids_to_add)

# restrict rain tracks to the considered supercells / cells
rain_tracks = np.array(rain_tracks)
SC_tracks = rain_tracks[np.isin(np.arange(ncells), SC_ids_mod)]


# open full observational dataset
usecols = ['ID', 'time', 'mesostorm', 'mesohailstorm', 'lon', 'lat', 'area', 'vel_x', 'vel_y', 'altitude', 'slope', 'max_CPC', 'mean_CPC', 'max_MESHS',
           'mean_MESHS', 'p_radar', 'p_dz', 'p_z_0', 'p_z_100', 'p_v_mean', 'p_d_mean', 'n_radar', 'n_dz', 'n_z_0', 'n_z_100', 'n_v_mean', 'n_d_mean']
fullset = pd.read_csv("/scratch/snx3000/mblanc/observations/Full_dataset_thunderstorm_types.csv", sep=';', usecols=usecols)
fullset['time'] = pd.to_datetime(fullset['time'], format="%Y%m%d%H%M")
fullset['ID'] = round(fullset['ID'])

# selection based on a given ID
sel_ids = [2019061114300008, 2019061122050060, 2019061118550036, 2019061113400021, 2019061100000106]
sel_ids_disp = [j%1000 for j in sel_ids]
selection = fullset[np.isin(fullset['ID'], sel_ids)]
selection['ID'] = round(selection['ID'] % 10000)

# plot the trajectories

# load geographic features
resol = '10m'  # use data at this scale
bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])

# figure
fig = plt.figure()#figsize=(6, 8))
fig.suptitle(pd.to_datetime(day,format='%Y%m%d').strftime('%d/%m/%Y'))

ax1 = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax1.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
ax1.add_feature(ocean, linewidth=0.2)
ax1.plot(SC_tracks[0]['lon'], SC_tracks[0]['lat'], 'x--', linewidth=1, markersize=3, label=str(SC_tracks[0]['cell_id']), transform=ccrs.PlateCarree())
ax1.plot(SC_tracks[1]['lon'][9:], SC_tracks[1]['lat'][9:], 'x--', linewidth=1, markersize=3, label=str(SC_tracks[1]['cell_id']), transform=ccrs.PlateCarree())
ax1.plot(SC_tracks[2]['lon'], SC_tracks[2]['lat'], 'x--', linewidth=1, markersize=3, label=str(SC_tracks[2]['cell_id']), transform=ccrs.PlateCarree())
ax1.legend(loc='lower right', fontsize='8')
ax1.title.set_text("modeled supercells")

ax2 = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
ax2.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
for i, ID in enumerate(sel_ids_disp):
    ax2.plot(selection['lon'][selection['ID']==ID], selection['lat'][selection['ID']==ID], 'x--', linewidth=1, markersize=3, label=str(ID), transform=ccrs.PlateCarree())
ax2.add_feature(ocean, linewidth=0.2)
ax2.legend(loc='upper left', fontsize='7')
ax2.title.set_text("observed supercells")

#=====================================================================================================================================================

# data to be filled
day = "20170801"
cut = "largecut"

# SC tracks
with open("/scratch/snx3000/mblanc/CS_tracker/output/supercell_" + day + ".json", "r") as read_file:
    SC_info = json.load(read_file)['SC_data']
SC_ids_mod = [SC_info[i]['rain_cell_id'] for i in range(len(SC_info))]

# rain tracks
with open("/scratch/snx3000/mblanc/cell_tracker/outfiles/cell_tracks_" + day + ".json", "r") as read_file:
    rain_tracks = json.load(read_file)['cell_data']
ncells = len(rain_tracks)
# among the supercells, select their linked cells, namely parents, childs, merged to cells
ids_to_add = []
for sc_id in SC_ids_mod:
    if rain_tracks[sc_id]['parent']:
        ids_to_add.append(rain_tracks[sc_id]['parent'])
    elif rain_tracks[sc_id]['merged_to']:
        ids_to_add.append(rain_tracks[sc_id]['merged_to'])
    elif rain_tracks[sc_id]['child']:
        ids_to_add.extend(rain_tracks[sc_id]['child'])
# and add them to ids to consider for supercell activity
SC_ids_mod.extend(ids_to_add)
# then remove manually the linked cells which have nothing to do with supercells, by looking at SC_info

# restrict rain tracks to the considered supercells / cells
rain_tracks = np.array(rain_tracks)
SC_tracks = rain_tracks[np.isin(np.arange(ncells), SC_ids_mod)]


# open full observational dataset
usecols = ['ID', 'time', 'mesostorm', 'mesohailstorm', 'lon', 'lat', 'area', 'vel_x', 'vel_y', 'altitude', 'slope', 'max_CPC', 'mean_CPC', 'max_MESHS',
           'mean_MESHS', 'p_radar', 'p_dz', 'p_z_0', 'p_z_100', 'p_v_mean', 'p_d_mean', 'n_radar', 'n_dz', 'n_z_0', 'n_z_100', 'n_v_mean', 'n_d_mean']
fullset = pd.read_csv("/scratch/snx3000/mblanc/observations/Full_dataset_thunderstorm_types.csv", sep=';', usecols=usecols)
fullset['time'] = pd.to_datetime(fullset['time'], format="%Y%m%d%H%M")
fullset['ID'] = round(fullset['ID'])

# selection based on a given ID
sel_ids = [2017080121350019, 2017080118100168, 2017080117350013, 2017080114050065, 2017080116500049, 2017080113400043, 2017080116050003, 2017080114400086, 2017080114500105, 2017080114100055]
sel_ids_disp = [j%1000 for j in sel_ids]
selection = fullset[np.isin(fullset['ID'], sel_ids)]
selection['ID'] = round(selection['ID'] % 10000)

# plot the trajectories

# load geographic features
resol = '10m'  # use data at this scale
bodr = cfeature.NaturalEarthFeature(category='cultural', name='admin_0_boundary_lines_land', scale=resol, facecolor='none', alpha=0.5)
ocean = cfeature.NaturalEarthFeature('physical', 'ocean', scale=resol, edgecolor='none', facecolor=cfeature.COLORS['water'])

# get the geographic static fields
infile = "/scratch/snx3000/mblanc/cell_tracker/infiles/" + cut + "_PREClffd" + day + ".nc"  # file with the prec
with xr.open_dataset(infile) as dset:
    lats = dset.variables['lat']  # 2D matrix
    lons = dset.variables['lon']  # 2D matrix

# figure
fig = plt.figure()#figsize=(6, 8))
fig.suptitle(pd.to_datetime(day,format='%Y%m%d').strftime('%d/%m/%Y'))

ax1 = fig.add_subplot(2, 1, 1, projection=ccrs.PlateCarree())
ax1.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
ax1.add_feature(ocean, linewidth=0.2)
for i in range(len(SC_tracks)):
    ax1.plot(SC_tracks[i]['lon'], SC_tracks[i]['lat'], 'x--', linewidth=1, markersize=2, label=str(SC_tracks[i]['cell_id']), transform=ccrs.PlateCarree())
#ax1.legend(loc='upper right', fontsize='6')
ax1.title.set_text("modeled supercells")

ax2 = fig.add_subplot(2, 1, 2, projection=ccrs.PlateCarree())
ax2.add_feature(bodr, linestyle='-', edgecolor='k', alpha=1)
for ID in sel_ids_disp:
    ax2.plot(selection['lon'][selection['ID']==ID], selection['lat'][selection['ID']==ID], 'x--', linewidth=1, markersize=2, label=str(ID), transform=ccrs.PlateCarree())
ax2.add_feature(ocean, linewidth=0.2)
#ax2.legend(loc='lower right', fontsize='7')
ax2.title.set_text("observed supercells")

fig.savefig(day+"_traj.png", dpi=300)
