from src.wrf_solar.read_solar import extract_one_day as solar_extract
from src.wrf_wind.read_wind import extract_one_day as wind_extract
import datetime 
import numpy as np 
from config.config import time_now
import xarray as xr 
from config.config import sd_wrf_outprefix
import os 

def convert_time(time_now:datetime.datetime):
    '''
    将当前时间分出3个时间段启动
    '''
    
    utc_time_now=time_now+datetime.timedelta(hours=-8)
    
    #
    shift_time_now1=utc_time_now+datetime.timedelta(hours=-8)
    shift_time_now2=utc_time_now+datetime.timedelta(hours=-7)
    shift_time_now3=utc_time_now+datetime.timedelta(hours=-9)
    
    #print(shift_time_now.hour)
    return [shift_time_now1,shift_time_now2,shift_time_now3]


def main(time_now):
    solar_file=solar_extract(time_now)
    wind_file=wind_extract(time_now)
    
    print(solar_file)
    solar_name=os.path.basename(solar_file)
    wind_name=os.path.basename(wind_file)
    
    ds_solar=xr.open_dataset(solar_file)
    ds_wind=xr.open_dataset(wind_file)
    
    total_ds=xr.merge([ds_solar,ds_wind])
    
    total_ds.to_netcdf(os.path.join(sd_wrf_outprefix,solar_name.replace('solar','')))
    print(os.path.join(sd_wrf_outprefix,solar_name.replace('solar',''))) 


if __name__=='__main__':
    
    t_list=convert_time(time_now)
    
    for itime in t_list:
        try:
            main(itime)
            
        except Exception as e :
            print('pass ')
