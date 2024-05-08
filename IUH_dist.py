# This script aims at computing the IUH distribution accross whole domain during a convective day

import numpy as np
import pandas as pd
#import xarray as xr
import json
from CaseStudies import IUH
from matplotlib import pyplot as plt

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

    Parameters
    ----------
    CS_days : str
        day of the case study, "YYYYmmdd"
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
CS_days = ['20120630', '20130727', '20130729', '20140625', '20170801', '20190610', '20190611', '20190613', '20190614',
           '20190820', '20210620', '20210628', '20210629', '20210708', '20210712', '20210713']
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
