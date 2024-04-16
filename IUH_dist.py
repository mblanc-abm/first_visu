# This script aims at computing the IUH distribution accross whole domain during a convective day

import numpy as np
import pandas as pd
#import xarray as xr
from CaseStudies import IUH
from matplotlib import pyplot as plt

# FUNCTIONS
#================================================================================================================================

def hourly_IUH_hist(day):
    
    path_p = "/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_3D_plev/"
    path_s = "/project/pr133/velasque/cosmo_simulations/climate_simulations/RUN_2km_cosmo6_climate/4_lm_f/output/1h_2D/"
    
    hours = np.arange(24)    
    for h in hours:
        date = pd.to_datetime(day + str(h).zfill(2) + "0000")
        fname_p = path_p + "lffd" + day + str(h).zfill(2) + "0000p.nc"
        fname_s = path_s + "lffd" + day + str(h).zfill(2) + "0000.nc"
        
        iuh = abs(IUH(fname_p, fname_s))
        
        #print(np.min(iuh), np.max(iuh))
        plt.hist(iuh, log=True, range=(50,150))
        plt.title(date.strftime("%d/%m/%Y %H:%M:%S"))
        plt.xlabel(r"|IUH| ($m^2/s^2$)")
        plt.show()


def daily_IUH(day):
    
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

#================================================================================================================================
# MAIN
#================================================================================================================================

#day = "20190611"
CS_days = ['20120630', '20130727', '20130729', '20140625', '20170801', '20190610', '20210708', '20210712', '20210713']

for day in CS_days:
    iuh = daily_IUH(day)
    iuh = np.ndarray.flatten(abs(iuh))

    # histogram
    plt.hist(iuh, bins=20, log=True, edgecolor='black', alpha=0.6)
    plt.axvline(np.percentile(iuh, 99), color='m', linestyle='dashed', linewidth=1.3, label='99th percentile ~' + str(round(np.percentile(iuh, 99),1)))
    plt.xlabel(r"|IUH| ($m^2/s^2$)")
    plt.ylabel("frequency")
    plt.legend(loc='upper right')
    plt.title(pd.to_datetime(day).strftime("%d/%m/%Y"))
    plt.show()