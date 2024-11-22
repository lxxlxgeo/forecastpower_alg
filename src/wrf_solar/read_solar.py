#%
import numpy as np
import re,os
from glob import glob
import datetime
from netCDF4 import Dataset
from wrf import (get_cartopy, latlon_coords, to_np, cartopy_xlim, cartopy_ylim,
                 getvar, ALL_TIMES)
from scipy.interpolate import griddata
from glob import glob
import pandas as pd
import xarray as xr
from config.config import wrf_extent,sd_solar_prefix,sd_solar_outprefix
#from src.aux import extract_tar_gz,delete_folder_contents
import metpy.calc as calc
import metpy.units as units
from tqdm import tqdm

def convert_xr(file):
    try:
        pattern = r'\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}'
        time_str=re.findall(pattern,file)[0]
        file_time=datetime.datetime.strptime(re.findall(pattern,file)[0],'%Y-%m-%d_%H:%M:%S')

    except:
        print('报错转移')
        pattern = r'\d{4}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2}'
        time_str=re.findall(pattern,file)[0]
        file_time=datetime.datetime.strptime(re.findall(pattern,file)[0],'%Y-%m-%d_%H:%M:%S')

  
    ok_time=pd.to_datetime(file_time)
    lon,lat,data_var=read_file(file)
    xr_ds=grid_data(lon,lat,data_var,ok_time)
    return xr_ds
#%%
def read_file(ncfile:str):
    '''
    读取单个的nc文件
    江苏省范围,全天候
    '''
    ds=Dataset(ncfile)
    #U10
    SDOWN=getvar(ds,'SWDNB',timeidx=ALL_TIMES).data
    #print(getvar(ds,'U10',timeidx=ALL_TIMES).data.shape)
    #print(U10.shape)
    #V10
    #V10=getvar(ds,'V10',timeidx=ALL_TIMES).data[0,:,:]
    

    XLAT=getvar(ds,'XLAT',timeidx=ALL_TIMES).data
    XLONG=getvar(ds,'XLONG',timeidx=ALL_TIMES).data
    #print(XLAT.shape)
    wl=np.where((XLONG>=wrf_extent[0]-0.1)&(XLONG<=wrf_extent[1]+0.1)&
                      (XLAT>=wrf_extent[2]-0.1)&(XLAT<=wrf_extent[3]+0.1))
    


    lon_line=XLONG[wl]
    lat_line=XLAT[wl]
    

    #U10
    SDOWN_line=SDOWN[wl]
    
    #V10
    #V10_line=V10[wl]
    print('step1')
    item_dict=dict()

    item_dict['SDOWN']=SDOWN_line

    
    
    #关闭文件
    ds.close()
    
    return lon_line,lat_line,item_dict

def grid_data(lon_line,lat_line,data_var,time):
    lat=np.arange(wrf_extent[2],wrf_extent[3],0.02)
    lon=np.arange(wrf_extent[0],wrf_extent[1],0.02)
    gridlon,gridlat=np.meshgrid(lon,lat)
    
    SDOWN=griddata((lon_line,lat_line),data_var['SDOWN'],(gridlon,gridlat),method='nearest',fill_value=-9999) #12
    #V10=griddata((lon_line,lat_line),data_var['V10'],(gridlon,gridlat),method='nearest',fill_value=-9999) #12
    

    
    SDOWN_da=xr.DataArray(SDOWN, coords={'latitude': lat, 'longitude': lon}, dims=('latitude', 'longitude')) 
    
    #V10_da=xr.DataArray(V10, coords={'lat': lat, 'lon': lon}, dims=('lat', 'lon')) 
    
    

    #将xarray.DataArray对象组合成一个xarray.Dataset对象，并指定变量名
    data_vars = {
        'SDOWN':SDOWN_da,
    }
    
    
    ds = xr.Dataset(data_vars)
    fillvalue=-9999
    #为每个变量设置编码属性，指定填充值
    for var_name in ds.variables:
        ds[var_name].encoding['_FillValue'] = fillvalue
    #为数据集添加时间维度和坐标
    ds['valid_time'] = xr.DataArray([time], dims=['valid_time'])
    ds.encoding['_FillValue'] = -9999 
    # 打印 Dataset 的信息
    return ds



def sort_files_by_time(file_list):
    # 导入os模块，用于获取文件的修改时间
    import os
    # 使用sorted函数对文件列表进行排序，使用lambda表达式作为排序的关键字
    # lambda表达式的作用是从文件名中提取出时间部分，并转换为datetime对象，以便比较
    # 例如，从'/nas/Datasets/WRF-RLDAS/henan/forecast-start12h/2023081812/wrfout_d02_2023-08-20_12:30:00'中提取出'2023-08-20_12:30:00'
    # 并使用datetime.strptime函数将其转换为datetime对象
    # datetime模块是Python内置的日期和时间处理模块
    from datetime import datetime
    try:
        sorted_file_list =file_list #sorted(file_list, key=lambda f: datetime.strptime(f[-19:], '%Y-%m-%d_%H:%M:%S'))
    except:
        print('报错转移')
        sorted_file_list = file_list # sorted(file_list, key=lambda f: datetime.strptime(f[-19:], '%Y-%m-%d_%H_%M_%S'))
    # 返回排序后的文件列表
    return sorted_file_list

def process_file(ifile):
    try:
        ds = convert_xr(ifile)
        print(os.path.basename(ifile))
        return ds
    except Exception as e:
        print(f'The error is {e}')
        return None

import concurrent.futures

def extract_one_day(forecast_time:datetime.datetime):
    '''
    执行提取一天的WRF辐照度数据.
    '''
    
    time_str=forecast_time.strftime('%Y%m%d%H0000')
    #print(time_str)
    input_files=glob(sd_solar_prefix+'/'+time_str+'/SOLAR_d01_*')
    print(sd_solar_prefix+'/'+time_str)
    
    if len(input_files)>0:
        
        # if time_str=='20231119120000':
        #     return 0
        files=sort_files_by_time(input_files)
        #print(files)
        out_prefix=sd_solar_outprefix
        # if not os.path.exists(out_prefix):
        #     os.mkdir(out_prefix)
        
        
        #out_time_str=forecast_time.strftime('%Y%m%d%H')
        outfilename='sd_wrf_solar_'+time_str+'.nc'
        outfile=os.path.join(out_prefix,outfilename)
        
        if os.path.exists(outfile):
            print(f'the out file is exists !!! >>>>> {outfile}')
            return outfile
            
        else:
            try:
                files=sorted(files)
                xr_list=[]
                # with concurrent.futures.ThreadPoolExecutor(2) as executor:
                #     # 并行处理文件
                #     results = list(executor.map(process_file, files))
                results=[process_file(file) for file in files]
                # 将所有有效的数据合并
                xr_list = [result for result in results if result is not None]
                ds_all = xr.concat(xr_list, dim='valid_time')

                # 按照时间维度排序
                ds_all = ds_all.sortby('valid_time')

                ds_all.to_netcdf(outfile)
                print(outfile+' is finish!!!')
                return outfile
            #delete_folder_contents(wrf_release_prefix_temp)
            #return files
            except Exception as e:
                print('程序出错')
                print(e)
    else:
        print('no file is find!!!')