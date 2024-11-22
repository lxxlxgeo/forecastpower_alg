import numpy as np
import pandas as pd
import datetime
import xarray as xr
from metpy.units import units
#from metpy import calc as mcalc
import metpy.calc as mp_calc
import os
from tqdm import tqdm
from joblib import Parallel, delayed
from config.config import csv_grid_path,sd_wrf_outprefix
import pandas as pd 

import warnings
warnings.filterwarnings('ignore')



import numpy as np

def calculate_relative_humidity(T2M, D2M):
    # 使用克劳修斯-克拉佩龙方程计算饱和水汽压
    e_s_T = 6.11 * np.exp(17.27 * (T2M - 273.15) / (T2M - 35.85))
    e_s_Td = 6.11 * np.exp(17.27 * (D2M - 273.15) / (D2M - 35.85))
    
    # 计算相对湿度
    RH = 100 * (e_s_Td / e_s_T)
    return RH


def convert_time(time_now:datetime.datetime):
    '''
    将当前时间分出3个时间段启动
    '''
    
    utc_time_now=time_now+datetime.timedelta(hours=-8)
    
    #
    shift_time_now1=utc_time_now+datetime.timedelta(hours=-8)

    
    #print(shift_time_now.hour)
    return shift_time_now1



def generate_time_list_wrf(time_bjt_now:datetime.datetime):
    
    year=time_bjt_now.year
    month=time_bjt_now.month
    day=time_bjt_now.day
    hours = time_bjt_now.hour
    
    date=datetime.datetime(year,month,day,hours)
    
    date=date+datetime.timedelta(hours=8) #当前时间往前+8

    #date=time_bjt_now
    
    start_time =date 

    # 设置结束时间为当天的23:59:59
    end_time = date+datetime.timedelta(hours=8) #小时的时间间隔
    #end_time = end_time + datetime.timedelta(days=1)  #days=6 为推送7天的数据

    # 设置时间间隔为1小时
    interval = datetime.timedelta(hours=1)

    # 生成时间列表
    current_time = start_time
    time_list = []

    while current_time <= end_time:
        time_list.append(current_time)
        current_time += interval
        
    #print(time_list)
    return time_list






def get_region_data(time_bjt_now, nwppath, lonmin, lonmax, latmin, latmax, dataType):
    """
    生成日期范围内的数据。

    参数：
    start_date (datetime.datetime): 起始日期。
    end_date (datetime.datetime): 结束日期。

    返回：
    aligned_radiation_values_list (list): 对齐的辐射数据列表。
    power_values_list (list): 功率数据列表。
    """
    #utc 文件时间
    file_utc_time = convert_time(time_bjt_now)
    #print(file_utc_time)

    #生成要预测的数据结果
    try:

        date_utc_list  = generate_time_list_wrf(file_utc_time)  #生成一个北京时间的列表

            
        #date_utc_list = [x+datetime.timedelta(hours=-8) for x in data_bjt_list] # 生成UTC时间的列表
    except Exception as e:
        print(e)
        return

    
    utc_file_timestr = file_utc_time.strftime('%Y%m%d%H')
    

    nwp_file=os.path.join(sd_wrf_outprefix,'sd_wrf__'+utc_file_timestr+'0000.nc')
    print(nwp_file)
    
    if (os.path.exists(nwp_file)):

        ds = xr.open_dataset(nwp_file) #打开文件
        #ds = ds.sel(longitude=slice(lonmin, lonmax),latitude=slice(latmax, latmin))
        ds = ds.sel(longitude=slice(lonmin, lonmax),latitude=slice(latmin, latmax))
        #print(ds)
        
        grid_df=pd.read_csv(csv_grid_path) #读取csv文件
        
        nwp_list=[]
        date_list=[]
        coor_value_list=[]
        
        for i in range(len(date_utc_list)):
            col_names=['SDOWN', 'U10','V10','WS100','T2M']
            #print(i)
            
            filter_ds=ds[col_names]
            try:
                #print(date_utc_list)
                filter_ds = filter_ds.sel(valid_time=date_utc_list[i],method='nearest')
            except Exception as e:
                print(f'select data wa : {e}')
                continue
            #print(filter_ds['ssrd'].mean())
            print(filter_ds)
            ssrd = filter_ds['SDOWN'].values
            #print(ssrd)
            #print(ssrd)
            #print(ssrd)
            #print(date_utc_list[i])
            
            ssrd[ssrd<0]=0.0
            #print(ssrd.mean())
            u10 =filter_ds['U10'].values
            v10 =filter_ds['V10'].values
            
            wind_speed_100=filter_ds['WS100'].values
            
            t2m=filter_ds['T2M'].values  #转换为摄氏度
            
            
            
            #以下是转出csv的写法:
            longitude = xr.DataArray(list(grid_df['lon']), dims="points")
            latitude = xr.DataArray(list(grid_df['lat']), dims="points")
            
            #print(filter_ds)

            ssrd_value=filter_ds['SDOWN'].sel(longitude=longitude,latitude=latitude,method='nearest').values
            u10_value=filter_ds['U10'].sel(longitude=longitude,latitude=latitude,method='nearest').values
            v10_value=filter_ds['V10'].sel(longitude=longitude,latitude=latitude,method='nearest').values
            t2m_value=filter_ds['T2M'].sel(longitude=longitude,latitude=latitude,method='nearest').values
            wind_speed_100_value=filter_ds['WS100'].sel(longitude=longitude,latitude=latitude,method='nearest').values 
            # print(d2m_value.shape)
            # print(filter_ds['d2m'][1].shape)
            
            #rh2_value=calculate_relative_humidity(t2m_value,d2m_value)
            t2m_c_value=t2m_value-273.15
            wind_speed_10_value=mp_calc.wind_speed(u10_value*units('m/s'),v10_value*units('m/s')).magnitude
            #wind_speed_100_value=mp_calc.wind_speed(u100_value*units('m/s'),v100_value*units('m/s')).magnitude
            
            df_item=dict()
            


            # 假设 longitude, latitude, ssrd_value 等都是 1D 的 numpy 数组
            longitude = np.array(longitude)
            latitude = np.array(latitude)
            ssrd_value = np.array(ssrd_value)
            ssrd_value[ssrd_value<0]=0.0
            wind_speed_10_value = np.array(wind_speed_10_value)
            wind_speed_100_value = np.array(wind_speed_100_value)
            t2m_c_value = np.array(t2m_c_value)
            t2m_c_value[t2m_c_value<0]=0.0
            # rh2_value = np.array(rh2_value)
            # rh2_value[rh2_value<0]=0.0

            # 使用 np.column_stack 按列堆叠数据
            ssrd_df = pd.DataFrame(np.column_stack([longitude, latitude, ssrd_value]), columns=['lon', 'lat', 'value'])
            wind_speed10_df = pd.DataFrame(np.column_stack([longitude, latitude, wind_speed_10_value]), columns=['lon', 'lat', 'value'])
            wind_speed100_df = pd.DataFrame(np.column_stack([longitude, latitude, wind_speed_100_value]), columns=['lon', 'lat', 'value'])
            t2m_df = pd.DataFrame(np.column_stack([longitude, latitude, t2m_c_value]), columns=['lon', 'lat', 'value'])
            #rh2_df = pd.DataFrame(np.column_stack([longitude, latitude, rh2_value]), columns=['lon', 'lat', 'value'])

            '''
            要返回的数据
            '''
            df_item['ssrd']=ssrd_df
            df_item['wind_speed10']=wind_speed10_df
            df_item['wind_speed100']=wind_speed100_df
            df_item['t2m']=t2m_df
            #df_item['rh2_df']=rh2_df
            
            
            
            #rh2=calculate_relative_humidity(t2m,d2m) #相对湿度
            
            
            
            
            wind_speed_10=mp_calc.wind_speed(u10*units('m/s'),v10*units('m/s')).magnitude
    
            
            
            t2m_c=t2m-273.15
            
            #print(df_item)
            
            ssrd[ssrd<0]=0.0
            t2m_c[t2m_c<0]=0.0
            

            ground_layer = np.stack((ssrd,wind_speed_10,wind_speed_100,t2m_c), axis=0)

            nwp_values = ground_layer
            #print(nwp_values)
            nwp_list.append(nwp_values)
            date_list.append(date_utc_list[i])
            coor_value_list.append(df_item)
            
        return nwp_list, date_list, file_utc_time,filter_ds['longitude'].values,filter_ds['latitude'].values,coor_value_list

    else:
        return None,None,None,None,None,None