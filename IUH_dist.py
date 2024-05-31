# This script computes the IUH, zeta and w distributions accross whole domain or a case study

import numpy as np
import pandas as pd
#import xarray as xr
import json
from CaseStudies import IUH
from matplotlib import pyplot as plt
import os

# FUNCTIONS
#================================================================================================================================

def daily_IUH(day):
    """
    computes the hourly IUH fields of a given day (00-23 UTC), over the whole domain

    Parameters
    ----------
    day : str
        considered day, "YYYmmdd"

    Returns
    -------
    iuhs : 3D array
        hourly (24) stacked IUH fields of the given day (00-23 UTC)
    """
    
    path_p = "/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_3D_plev/"
    path_s = "/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_2D/"
    
    hours = np.arange(24)
    iuhs = []
    for h in hours:
        fname_p = path_p + "lffd" + day + str(h).zfill(2) + "0000p.nc"
        fname_s = path_s + "lffd" + day + str(h).zfill(2) + "0000.nc"
        iuh = IUH(fname_p, fname_s)
        iuhs.append(iuh)
    
    iuhs = np.stack(iuhs)
    return iuhs


def IUH_meso_CSs(CS_days, typ):
    """
    gather the IUH mean or max values of the case studies mesocyclones according to the requested type
    compatible with SDT1 only

    Parameters
    ----------
    CS_days : str
        list of case studies days, "YYYYmmdd"
    typ : str
        "mean" or "max", type of the IUH parameter

    Returns
    -------
    values : list of floats
        list of the IUH values
    """
    
    path = "/scratch/snx3000/mblanc/SDT1_output/CaseStudies/"
    
    values = []
    for day in CS_days:
        
        file = path + "supercell_" + day + ".json"
        
        with open(file, "r") as read_file:
            supercells = json.load(read_file)['supercell_data']
        
        for supercell in supercells:
            values.extend(supercell[typ + "_val"])
        
    return values


def meso_supercell_variable_CSs(path, CS_days, variable, zeta_th, w_th):
    """
    gather the requested supercell or mesocyclone variable, given the SDT2 thresholds, encompassing all the case studies supercells
    compatible with SDT2 only
    
    Parameters
    ----------
    path : str
        path to the SDT2 supercell tracks
    CS_days : str
        list of case studies days, "YYYYmmdd"
    variable : str
        mesocyclone variables: "max_zeta", "mean_zeta", "max_w", "mean_w", "meso_max_hail", "area"
        cell variables: "cell_max_hail", "cell_max_wind", "cell_max_rain"
    zeta_th : str
        vorticity threshold
    w_th : str
        updraught velocity threshold

    Returns
    -------
    values : list of floats
        list of the all the variable values of the case studies tracked supercells
    """
    
    values = []
    for day in CS_days:
        
        file = os.path.join(path, "supercell_zetath" + zeta_th + "_wth" + w_th + "_" + day + ".json")
        
        with open(file, "r") as read_file:
            supercells = json.load(read_file)['supercell_data']
        
        for supercell in supercells:
            SC_values = np.array(supercell[variable])
            if np.any(SC_values == None):
                SC_values[SC_values == None] = np.nan
                SC_values = SC_values.astype(float)
            if not np.all(np.isnan(SC_values)):
                values.extend(SC_values)
        
    return values


def IUH_meso_domain(season, typ, skipped_days=None):
    """
    gather the IUH mean or max values of the whole domain mesocyclones according to the requested type during a given season

    Parameters
    ----------
    season : str
        considered season, "YYYY"
    typ : str
        "mean" or "max", type of the IUH parameter
    skipped_days : list of str
        list of missing days which consequently must be skipped

    Returns
    -------
    values : list of floats
        list of the IUH values
    """
    
    path = "/scratch/snx3000/mblanc/SDT1_output/seasons/" + season + "/"
    
    start_day = pd.to_datetime(season + "0401")
    end_day = pd.to_datetime(season + "0930")
    daylist = pd.date_range(start_day, end_day)
    
    # remove skipped days from daylist
    if skipped_days:
        skipped_days = pd.to_datetime(skipped_days, format="%Y%m%d")
        daylist = [day for day in daylist if day not in skipped_days]
    
    values = []
    for day in daylist:
        
        file = path + "supercell_" + day.strftime("%Y%m%d") + ".json"
        
        with open(file, "r") as read_file:
            supercells = json.load(read_file)['supercell_data']
        
        for supercell in supercells:
            values.extend(supercell[typ + "_val"])
        
    return values

#================================================================================================================================
# MAIN
#================================================================================================================================

## EVERY GRID POINTS OF WHOLE DOMAIN ##

#day = "20190611"
#CS_days = ['20120630', '20130727', '20130729', '20140625', '20170801', '20190610', '20190611', '20190613', '20190614',
#            '20190820', '20210620', '20210628', '20210629', '20210708', '20210712', '20210713']
#iuh_thresh = 50

# # one histogram per CS day
# for day in CS_days:
#     iuh = daily_IUH(day)
#     iuh = np.ndarray.flatten(abs(iuh))

#     # histogram
#     plt.hist(iuh, bins=20, log=True, edgecolor='black', alpha=0.6)
#     plt.axvline(np.percentile(iuh, 99), color='m', linestyle='dashed', linewidth=1.3, label='99th percentile ~' + str(round(np.percentile(iuh, 99),1)))
#     plt.xlabel(r"|IUH| ($m^2/s^2$)")
#     plt.ylabel("frequency")
#     plt.legend(loc='upper right')
#     plt.title(pd.to_datetime(day).strftime("%d/%m/%Y"))
#     plt.show()


# # aggregated CSs
# iuh = []
# for i, day in enumerate(CS_days):
#     iuh.extend(np.ndarray.flatten(daily_IUH(day)))

# iuh_abs = np.abs(iuh)
# iuh_filtered = iuh_abs
# iuh_filtered[iuh_filtered<iuh_thresh] = np.nan
# figname = "aggregated_CSs_iuh" + str(iuh_thresh) + ".png"
# figtitle = "|IUH|>" + str(iuh_thresh) + "; aggregated case studies"

# # histogram
# fig = plt.figure()
# plt.hist(iuh_filtered, bins=30, range=(50,150), log=True, edgecolor='black', alpha=0.6)
# plt.axvline(np.nanpercentile(iuh_filtered, 70), color='y', linestyle='dashed', linewidth=1.3, label='70th percentile ~' + str(round(np.nanpercentile(iuh_filtered, 70),1)))
# plt.axvline(np.nanpercentile(iuh_filtered, 75), color='r', linestyle='dashed', linewidth=1.3, label='75th percentile ~' + str(round(np.nanpercentile(iuh_filtered, 75),1)))
# plt.axvline(np.nanpercentile(iuh_filtered, 80), color='g', linestyle='dashed', linewidth=1.3, label='80th percentile ~' + str(round(np.nanpercentile(iuh_filtered, 80),1)))
# plt.xlabel(r"|IUH| ($m^2/s^2$)")
# plt.ylabel("frequency")
# plt.legend(loc='upper right')
# plt.title("50<|IUH|<150; aggregated case studies") #figtitle)
# fig.savefig("aggregated_CSs_iuh50-150.png") #figname)

#================================================================================================================================
## CASE STUDIES MESOCYCLONES ##

# typ = "mean"

# IUH_values = IUH_meso_CSs(CS_days, typ)

# # histogram
# fig = plt.figure()
# plt.hist(np.abs(IUH_values), bins=30, edgecolor='black', alpha=0.6)
# plt.axvline(np.nanpercentile(np.abs(IUH_values), 20), color='m', linestyle='dashed', linewidth=1.3, label='20th percentile ~' + str(round(np.nanpercentile(np.abs(IUH_values), 20),1)))
# plt.axvline(np.nanpercentile(np.abs(IUH_values), 30), color='y', linestyle='dashed', linewidth=1.3, label='30th percentile ~' + str(round(np.nanpercentile(np.abs(IUH_values), 30),1)))
# plt.axvline(np.nanpercentile(np.abs(IUH_values), 40), color='g', linestyle='dashed', linewidth=1.3, label='40th percentile ~' + str(round(np.nanpercentile(np.abs(IUH_values), 40),1)))
# plt.axvline(np.nanpercentile(np.abs(IUH_values), 50), color='r', linestyle='dashed', linewidth=1.3, label='50th percentile ~' + str(round(np.nanpercentile(np.abs(IUH_values), 50),1)))
# plt.xlabel(r"|IUH| ($m^2/s^2$)")
# plt.ylabel("frequency")
# plt.legend(loc='upper right')
# plt.title("Aggregated case studies " + typ + " IUH")
# fig.savefig("meso_" + typ + "_aggregated_CSs.png")

#================================================================================================================================
## CASE STUDIES MESOCYCLONE/SUPERCELL VARIABLES ##

path = "/scratch/snx3000/mblanc/SDT/SDT2_output/current_climate/CaseStudies"
CS_days = CS_days = ['20120630', '20130727', '20130729', '20140625', '20170801', '20190610', '20190611', '20190613',
                     '20190614', '20190820', '20210620', '20210628', '20210629', '20210708', '20210712', '20210713']
zeta_ths = np.array([4,5,6])*1e-3
w_ths = [5,6,7]

# # meso zeta and w, mean and max
# for zeta_th in zeta_ths:
#     for w_th in w_ths:
#         zeta_th, w_th = str(zeta_th), str(w_th)
        
#         max_zeta = meso_supercell_variable_CSs(path, CS_days, "max_zeta", zeta_th, w_th)
#         mean_zeta = meso_supercell_variable_CSs(path, CS_days, "mean_zeta", zeta_th, w_th)

#         # histogram
#         fig = plt.figure()
#         plt.hist(max_zeta, bins=30, edgecolor='black', alpha=0.5, color='b', label="max")
#         plt.hist(mean_zeta, bins=30, edgecolor='black', alpha=0.3, color='r', label="mean")
#         plt.xlabel(r"$\zeta$ ($s^{-1}$)")
#         plt.ylabel("frequency")
#         plt.legend(loc='upper right')
#         plt.title(r"$\zeta_{th}=$" + zeta_th + "; $w_{th}=$" + w_th)
#         fig.savefig("meso_zeta_zetath"+zeta_th+"_wth"+w_th+".png")

#         fig = plt.figure()
#         plt.hist(np.abs(max_zeta), bins=30, edgecolor='black', alpha=0.5, color='b', label="max")
#         plt.hist(np.abs(mean_zeta), bins=30, edgecolor='black', alpha=0.3, color='r', label="mean")
#         plt.xlabel(r"$|\zeta|$ ($s^{-1}$)")
#         plt.ylabel("frequency")
#         plt.legend(loc='upper right')
#         plt.title(r"$\zeta_{th}=$" + zeta_th + "; $w_{th}=$" + w_th)
#         fig.savefig("meso_abszeta_zetath"+zeta_th+"_wth"+w_th+".png")

#         max_w = meso_supercell_variable_CSs(path, CS_days, "max_w", zeta_th, w_th)
#         mean_w = meso_supercell_variable_CSs(path, CS_days, "mean_w", zeta_th, w_th)

#         fig = plt.figure()
#         plt.hist(max_w, bins=30, edgecolor='black', alpha=0.5, color='b', label="max")
#         plt.hist(mean_w, bins=30, edgecolor='black', alpha=0.3, color='r', label="mean")
#         plt.xlabel(r"$w$ (m/s)")
#         plt.ylabel("frequency")
#         plt.legend(loc='upper right')
#         plt.title(r"$\zeta_{th}=$" + zeta_th + "; $w_{th}=$" + w_th)
#         fig.savefig("meso_w_zetath"+zeta_th+"_wth"+w_th+".png")

# # meso area, meso max hail
# cmap = plt.cm.tab10.colors
# fig = plt.figure()
# i = 0
# for zeta_th in zeta_ths:
#     for w_th in w_ths:
#         zeta_th, w_th = str(zeta_th), str(w_th)
#         meso_max_hail = meso_supercell_variable_CSs(path, CS_days, "meso_max_hail", zeta_th, w_th)
#         plt.hist(meso_max_hail, bins=30, alpha=0.5, range=(5,50), edgecolor=cmap[i], label=r"$\zeta_{th}=$" + zeta_th + "; $w_{th}=$" + w_th, histtype='step')
#         i += 1
# plt.xlabel("meso max hail diameter (mm)")
# plt.ylabel("frequency")
# plt.legend(loc='upper right', fontsize=7)
# fig.savefig("meso_max_hail.png")

# cell max hail
cmap = plt.cm.tab10.colors
fig = plt.figure()
i = 0
for zeta_th in zeta_ths:
    for w_th in w_ths:
        zeta_th, w_th = str(zeta_th), str(w_th)
        cell_max_rain = meso_supercell_variable_CSs(path, CS_days, "cell_max_rain", zeta_th, w_th)
        plt.hist(cell_max_rain, bins=30, alpha=0.5, edgecolor=cmap[i], label=r"$\zeta_{th}=$" + zeta_th + "; $w_{th}=$" + w_th, histtype='step')
        i += 1
plt.xlabel("cell max rain rate (mm/h)")
plt.ylabel("frequency")
plt.legend(loc='upper right', fontsize=7)
fig.savefig("cell_max_rain.png")

#================================================================================================================================
## DOMAIN MESOCYCLONES ##

# typ = "mean"
# years = ["2011", "2012", "2013", "2014", "2015", "2016", "2017", "2018", "2019", "2020", "2021"]
# skipped_days = ['20120604', '20140923', '20150725', '20160927', '20170725']

# for season in years:
    
#     IUH_values = IUH_meso_domain(season, typ, skipped_days)

#     # histogram
#     fig = plt.figure()
#     plt.hist(np.abs(IUH_values), bins=30, edgecolor='black', alpha=0.6)
#     plt.axvline(np.nanpercentile(np.abs(IUH_values), 20), color='m', linestyle='dashed', linewidth=1.3, label='20th percentile ~' + str(round(np.nanpercentile(np.abs(IUH_values), 20),1)))
#     plt.axvline(np.nanpercentile(np.abs(IUH_values), 30), color='y', linestyle='dashed', linewidth=1.3, label='30th percentile ~' + str(round(np.nanpercentile(np.abs(IUH_values), 30),1)))
#     plt.axvline(np.nanpercentile(np.abs(IUH_values), 40), color='g', linestyle='dashed', linewidth=1.3, label='40th percentile ~' + str(round(np.nanpercentile(np.abs(IUH_values), 40),1)))
#     plt.axvline(np.nanpercentile(np.abs(IUH_values), 50), color='r', linestyle='dashed', linewidth=1.3, label='50th percentile ~' + str(round(np.nanpercentile(np.abs(IUH_values), 50),1)))
#     plt.legend(loc='upper right')
#     plt.xlabel(r"|IUH| ($m^2/s^2$)")
#     plt.ylabel("frequency")
#     plt.title("Season " + season + " aggregated " + typ + " IUH")
#     fig.savefig("meso_" + typ + "_season" + season + "_aggregated_domain.png")
