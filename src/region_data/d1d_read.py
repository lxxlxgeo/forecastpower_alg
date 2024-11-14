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
from config.config import csv_grid_path
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



def convert_run_time(now_time):
    # 获取当前的日期，并将当前的日期转换为文件日期，文件日期为UTC时间
    print('the system time is {}'.format(now_time))
    year=now_time.year
    month=now_time.month
    day=now_time.day
    hour=now_time.hour

    if (hour>=4)&(hour<16):
        return datetime.datetime(year,month,day,12)+datetime.timedelta(days=-1)
    elif(hour>=0)&(hour<4):
        return datetime.datetime(year,month,day,0)+datetime.timedelta(days=-1)
    elif hour>=16:
        return datetime.datetime(year,month,day,0)


def generate_time_list(utc_file_date: datetime.datetime, time_bjt_now):

    hours = utc_file_date.hour

    if hours == 12:
        #如果是UTC时间的12时，则预测两天后的数据
        date = utc_file_date + datetime.timedelta(days=2)
    elif hours == 0:
        #如果UTC时间为00时，则预测一天后的数据
        date = utc_file_date + datetime.timedelta(days=1)
    else:
        print('set time is error,the hour must is 0 or 12,please check!')

    #设置生成的时间 生成的时间务必为当日的0时0分
    # start_time = date.replace(hour=0, minute=0, second=0)
    # if time_bjt_now.hour < 12:
    #     start_time = time_bjt_now.replace(hour=6, minute=0, second=0)
    # else:
    #     start_time = time_bjt_now.replace(hour=18, minute=0, second=0)
    start_time = date.replace(hour=0, minute=0, second=0)  # 包含超短期

    # 设置结束时间为当天的23:59:59
    end_time = date.replace(hour=23, minute=59, second=59)
    end_time = end_time + datetime.timedelta(days=1)  #days=6 为推送7天的数据

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

def generate_time_list_N9_N10(utc_file_date: datetime.datetime):

    hours = utc_file_date.hour

    if hours == 12:
        #如果是UTC时间的12时，则预测两天后的数据
        date = utc_file_date + datetime.timedelta(days=2)
    elif hours == 0:
        #如果UTC时间为00时，则预测一天后的数据
        date = utc_file_date + datetime.timedelta(days=1)
    else:
        print('set time is error,the hour must is 0 or 12,please check!')

    # 设置生成的时间 生成的时间务必为当日的0时0分
    start_time = date.replace(hour=0, minute=0, second=0)
    start_time = start_time + datetime.timedelta(days=8)  # 第九天

    # 设置结束时间为当天的23:59:59
    end_time = start_time.replace(hour=23, minute=59, second=59)
    end_time = end_time + datetime.timedelta(days=1)  #days=2 为推送3天的数据

    # 设置时间间隔为15分钟
    interval = datetime.timedelta(minutes=15)

    # 生成时间列表
    current_time = start_time
    time_list = []

    while current_time <= end_time:
        time_list.append(current_time)
        current_time += interval
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
    file_utc_time = convert_run_time(time_bjt_now)
    #print(file_utc_time)

    #生成要预测的数据结果
    try:
        if dataType == 'ECMWF':
            data_bjt_list = generate_time_list(file_utc_time, time_bjt_now)  #生成一个北京时间的列表
        elif dataType == 'GFS':
            data_bjt_list = generate_time_list_N9_N10(file_utc_time)  # 生成一个北京时间的列表
            
        date_utc_list = [x+datetime.timedelta(hours=-8) for x in data_bjt_list] # 生成UTC时间的列表
    except Exception as e:
        print(e)
        return

    utc_file_timestr = file_utc_time.strftime('%Y%m%d%H')

    # nwp_file=f'/data/Datasets/power_forecast/nwp_push/ECMWF/jiangsu/jiangsu_d1d/jiangsu_{utc_file_timestr}.nc'
    if dataType == 'ECMWF':
        nwp_file = os.path.join(nwppath, f'shandong_{utc_file_timestr}.nc')
        # 如果这一报的NWP没有按时到达，则读取上一报进行预测
        if not os.path.exists(nwp_file):
            file_utc_time2 = file_utc_time+datetime.timedelta(hours=-12)
            utc_file_timestr2 = file_utc_time2.strftime('%Y%m%d%H')
            # nwp_file = f'/data/Datasets/power_forecast/nwp_push/ECMWF/jiangsu/jiangsu_d1d/jiangsu_{utc_file_timestr2}.nc'
            nwp_file = os.path.join(nwppath, f'shandong_{utc_file_timestr2}.nc')
    elif dataType == 'GFS':
        nwp_file = os.path.join(nwppath, f'jiangsu_wind_{utc_file_timestr}.nc')
        # 如果这一报的NWP没有按时到达，则读取上一报进行预测
        if not os.path.exists(nwp_file):
            file_utc_time2 = file_utc_time+datetime.timedelta(hours=-12)
            utc_file_timestr2 = file_utc_time2.strftime('%Y%m%d%H')
            # nwp_file = f'/data/Datasets/power_forecast/nwp_push/ECMWF/jiangsu/jiangsu_d1d/jiangsu_{utc_file_timestr2}.nc'
            nwp_file = os.path.join(nwppath, f'jiangsu_wind_{utc_file_timestr2}.nc')
    
    if (os.path.exists(nwp_file)):

        ds = xr.open_dataset(nwp_file) #打开文件
        ds = ds.sel(longitude=slice(lonmin, lonmax),latitude=slice(latmax, latmin))
        #print(ds)
        
        grid_df=pd.read_csv(csv_grid_path) #读取csv文件
        
        nwp_list=[]
        date_list=[]
        coor_value_list=[]
        
        for i in range(len(date_utc_list)):
            col_names=['ssrd', 'u10','v10','u100','v100','t2m','d2m']
            
            ################################### 差分
            sfc_rad = [
                'ssrd',
            ]  #辐射变量提取
            # 合并所选择的数据集 U,V 19个气压层的数据

            #瞬时变量
            sfc_name = ['u10','v10','u100','v100','t2m','d2m']

            ds_sfc = ds[sfc_name]  #气象变量，瞬时值

            ds_rad = ds[sfc_rad]  #辐射变量，累计值

            ds_rad_diff = ds_rad.diff(dim='valid_time') / (3600 / 4)  # 不用管这个转换是否有问题，直接差分出辐射变量的瞬时值

            merge_ds_all = xr.merge([ds_sfc, ds_rad_diff])
            #print(merge_ds_all['ssrd'].mean())
            
            # 合并所选择的数据集 U,V 19个气压层的数据
            merge_ds_all = merge_ds_all[col_names]
            #print(merge_ds_all['valid_time'])
            try:
                filter_ds = merge_ds_all.sel(valid_time=slice(date_utc_list[i]+datetime.timedelta(minutes=(-15*1)),date_utc_list[i]+datetime.timedelta(minutes=(15*1))))
                if filter_ds.dims["valid_time"] != 3:
                    diff = filter_ds.dims["valid_time"] - 3
                    filter_ds=merge_ds_all.sel(valid_time=slice(date_utc_list[i]+datetime.timedelta(minutes=(-15*1)),date_utc_list[i]+datetime.timedelta(minutes=(15*1-15*diff))))
                #根据实发数据的时间索引，匹配NWP
            except Exception as e:
                print(f'select data wa : {e}')
                continue
            #print(filter_ds['ssrd'].mean())
            ssrd = filter_ds['ssrd'].values
            #print(ssrd)
            #print(date_utc_list[i])
            
            ssrd[ssrd<0]=0.0
            #print(ssrd.mean())
            u10 =filter_ds['u10'].values
            v10 =filter_ds['v10'].values
            
            u100=filter_ds['u100'].values
            v100=filter_ds['v100'].values
            
            t2m=filter_ds['t2m'].values  #转换为摄氏度
            d2m=filter_ds['d2m'].values   #转换为摄氏度
            
            
            #以下是转出csv的写法:
            longitude = xr.DataArray(list(grid_df['lon']), dims="points")
            latitude = xr.DataArray(list(grid_df['lat']), dims="points")
            
            #print(filter_ds)

            ssrd_value=filter_ds['ssrd'][1].sel(longitude=longitude,latitude=latitude,method='nearest').values
            u10_value=filter_ds['u10'][1].sel(longitude=longitude,latitude=latitude,method='nearest').values
            v10_value=filter_ds['v10'][1].sel(longitude=longitude,latitude=latitude,method='nearest').values
            u100_value=filter_ds['u100'][1].sel(longitude=longitude,latitude=latitude,method='nearest').values
            v100_value=filter_ds['v100'][1].sel(longitude=longitude,latitude=latitude,method='nearest').values
            t2m_value=filter_ds['t2m'][1].sel(longitude=longitude,latitude=latitude,method='nearest').values
            d2m_value=filter_ds['d2m'][1].sel(longitude=longitude,latitude=latitude,method='nearest').values 
            # print(d2m_value.shape)
            # print(filter_ds['d2m'][1].shape)
            
            rh2_value=calculate_relative_humidity(t2m_value,d2m_value)
            t2m_c_value=t2m_value-273.15
            wind_speed_10_value=mp_calc.wind_speed(u10_value*units('m/s'),v10_value*units('m/s')).magnitude
            wind_speed_100_value=mp_calc.wind_speed(u100_value*units('m/s'),v100_value*units('m/s')).magnitude
            
            df_item=dict()
            
            # longitude=np.array(longitude)
            # latitude=np.array(latitude)
            # #汇总为df 
            # ssrd_df=pd.DataFrame([longitude,latitude,ssrd_value],columns=['lon','lat','value'])
            # wind_speed10_df=pd.DataFrame([longitude,latitude,wind_speed_10_value],columns=['lon','lat','value'])
            # wind_speed100_df=pd.DataFrame([longitude,latitude,wind_speed_100_value],columns=['lon','lat','value'])
            # t2m_df=pd.DataFrame([longitude,latitude,t2m_c_value],columns=['lon','lat','value'])
            # rh2_df=pd.DataFrame([longitude,latitude,rh2_value],columns=['lon','lat','value'])
            # import numpy as np
            # import pandas as pd

            # 假设 longitude, latitude, ssrd_value 等都是 1D 的 numpy 数组
            longitude = np.array(longitude)
            latitude = np.array(latitude)
            ssrd_value = np.array(ssrd_value)
            ssrd_value[ssrd_value<0]=0.0
            wind_speed_10_value = np.array(wind_speed_10_value)
            wind_speed_100_value = np.array(wind_speed_100_value)
            t2m_c_value = np.array(t2m_c_value)
            t2m_c_value[t2m_c_value<0]=0.0
            rh2_value = np.array(rh2_value)
            rh2_value[rh2_value<0]=0.0

            # 使用 np.column_stack 按列堆叠数据
            ssrd_df = pd.DataFrame(np.column_stack([longitude, latitude, ssrd_value]), columns=['lon', 'lat', 'value'])
            wind_speed10_df = pd.DataFrame(np.column_stack([longitude, latitude, wind_speed_10_value]), columns=['lon', 'lat', 'value'])
            wind_speed100_df = pd.DataFrame(np.column_stack([longitude, latitude, wind_speed_100_value]), columns=['lon', 'lat', 'value'])
            t2m_df = pd.DataFrame(np.column_stack([longitude, latitude, t2m_c_value]), columns=['lon', 'lat', 'value'])
            rh2_df = pd.DataFrame(np.column_stack([longitude, latitude, rh2_value]), columns=['lon', 'lat', 'value'])

            '''
            要返回的数据
            '''
            df_item['ssrd']=ssrd_df
            df_item['wind_speed10']=wind_speed10_df
            df_item['wind_speed100']=wind_speed100_df
            df_item['t2m']=t2m_df
            df_item['rh2_df']=rh2_df
            
            
            
            rh2=calculate_relative_humidity(t2m,d2m) #相对湿度
            
            
            
            
            wind_speed_10=mp_calc.wind_speed(u10*units('m/s'),v10*units('m/s')).magnitude
            wind_dir_10=mp_calc.wind_direction(u10*units('m/s'),v10*units('m/s')).magnitude
            
            wind_speed_100=mp_calc.wind_speed(u100*units('m/s'),v100*units('m/s')).magnitude
            wind_dir_100=mp_calc.wind_direction(u100*units('m/s'),v100*units('m/s')).magnitude
            
            t2m_c=t2m-273.15
            
            #print(df_item)
            
            ssrd[ssrd<0]=0.0
            t2m_c[t2m_c<0]=0.0
            rh2[rh2<0]=0.0

            ground_layer = np.stack((ssrd,wind_speed_10,wind_speed_100,t2m_c,rh2), axis=0)

            nwp_values = ground_layer

            if nwp_values.shape[1]==3:
                nwp_list.append(nwp_values[:,1,:,:])
                date_list.append(date_utc_list[i])
                coor_value_list.append(df_item)
            else:
                continue
        return nwp_list, date_list, file_utc_time,filter_ds['longitude'].values,filter_ds['latitude'].values,coor_value_list

    else:
        return None,None,None,None,None,None