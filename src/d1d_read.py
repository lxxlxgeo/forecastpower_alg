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


import warnings
warnings.filterwarnings('ignore')


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


def generate_time_list(utc_file_date: datetime.datetime, time_bjt_now,cdq_tips='DQ'):

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
    start_time = time_bjt_now.replace(hour=0, minute=0, second=0)  # 包含超短期

    # 设置结束时间为当天的23:59:59
    end_time = date.replace(hour=23, minute=59, second=59)
    if cdq_tips=='CDQ':
        end_time = end_time + datetime.timedelta(days=3)  #days=6 为推送7天的数据
    else:
        end_time = end_time + datetime.timedelta(days=9)  #days=6 为推送7天的数据

    # 设置时间间隔为15分钟
    interval = datetime.timedelta(minutes=15)

    # 生成时间列表
    current_time = start_time
    time_list = []

    while current_time <= end_time:
        time_list.append(current_time)
        current_time += interval
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

def get_data(time_bjt_now, nwppath, lonmin, lonmax, latmin, latmax, dataType,cdq_tips='DQ'):
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

    #生成要预测的数据结果
    try:
        if dataType == 'ECMWF':
            data_bjt_list = generate_time_list(file_utc_time, time_bjt_now,cdq_tips=cdq_tips)  #生成一个北京时间的列表
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
        nwp_list=[]
        date_list=[]
        for i in range(len(date_utc_list)):
            # col_names = ['u','v','u10','v10','u100','v100','t2m','sp','fg310']
            col_names = ['u','v','u10','v10','u100','v100','t2m','sp']

            # 合并所选择的数据集 U,V 19个气压层的数据
            ds = ds[col_names]
            ds = ds.sel(longitude=slice(lonmin, lonmax),latitude=slice(latmax, latmin))
            try:
                #print('b')
                if dataType == 'ECMWF':
                    filter_ds=ds.sel(valid_time=slice(date_utc_list[i]+datetime.timedelta(minutes=(-15*3)),date_utc_list[i]+datetime.timedelta(minutes=(15*3))))
                    if filter_ds.dims["valid_time"] != 7:
                        diff = filter_ds.dims["valid_time"] - 7
                        filter_ds=ds.sel(valid_time=slice(date_utc_list[i]+datetime.timedelta(minutes=(-15*3)),date_utc_list[i]+datetime.timedelta(minutes=(15*3-15*diff))))
                    
                elif dataType == 'GFS':
                    filter_ds=ds.sel(valid_time=slice(date_utc_list[i]+datetime.timedelta(minutes=(-15*8)),date_utc_list[i]+datetime.timedelta(minutes=(15*0))))
            except Exception as e:
                print(f'select data wa : {e}')
                continue

            # 气压层风场
            #print(filter_ds['u'].shape)
            win_u=filter_ds['u'].values #获取U向风
            win_v=filter_ds['v'].values #获取V向风
            #print(win_u.shape)
            
            # 100米风向风速
            win_u100=filter_ds['u100'].values
            win_v100=filter_ds['v100'].values
            
            wind_speed_100=mp_calc.wind_speed(win_u100*units('m/s'),win_v100*units('m/s')).magnitude
            wind_dir_100=mp_calc.wind_direction(win_u100*units('m/s'),win_v100*units('m/s')).magnitude
            
            # 10米风向风速
            wind_u10=filter_ds['u10'].values
            wind_v10=filter_ds['v10'].values
            
            wind_speed_10=mp_calc.wind_speed(wind_u10*units('m/s'),wind_v10*units('m/s')).magnitude
            wind_dir_10=mp_calc.wind_direction(wind_u10*units('m/s'),wind_v10*units('m/s')).magnitude
            
            wind_dir_10_sin=np.sin(np.deg2rad(wind_dir_10))
            wind_dir_100_sin=np.sin(np.deg2rad(wind_dir_100))
            
            wind_dir_10_cos=np.cos(np.deg2rad(wind_dir_10))
            wind_dir_100_cos=np.cos(np.deg2rad(wind_dir_100))

            t2m=filter_ds['t2m'].values
            sp=filter_ds['sp'].values

            ground_layer=np.stack((wind_speed_10, wind_speed_100, wind_dir_10_sin, wind_dir_100_sin, t2m, sp),axis=0)

            wind_speed_layer=mp_calc.wind_speed(win_u*units('m/s'),win_v*units('m/s')).magnitude  #19层风速
            wind_dir_layer=mp_calc.wind_direction(win_u*units('m/s'),win_v*units('m/s')).magnitude #19层风向

            wind_dir_layer_sin=np.sin(np.deg2rad(wind_dir_layer))
            wind_dir_layer_cos=np.cos(np.deg2rad(wind_dir_layer))

            wind_layer=np.concatenate([wind_speed_layer[:,0:4], wind_dir_layer_sin[:,0:4]],axis=1)
            wind_layer=np.transpose(wind_layer,(1,0,2,3))
            nwp_values=np.concatenate([ground_layer,wind_layer],axis=0)

            if nwp_values.shape[1]==7:
                nwp_list.append(nwp_values)
                date_list.append(date_utc_list[i])
            else:
                continue
        return nwp_list, date_list, file_utc_time

    else:
        return None,None,None


def get_data_solar(time_bjt_now, nwppath, lonmin, lonmax, latmin, latmax, dataType,cdq_tips='DQ'):
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

    #生成要预测的数据结果
    try:
        if dataType == 'ECMWF':
            data_bjt_list = generate_time_list(file_utc_time, time_bjt_now,cdq_tips=cdq_tips)  #生成一个北京时间的列表
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
        nwp_list=[]
        date_list=[]
        for i in range(len(date_utc_list)):
            col_names=['ssrd', 'strd', 'ssr', 'str', 'tsr', 'ttr', 'tsrc', 'ttrc', 'ssrc',\
                'strc', 'fdir', 'cdir', 'lcc', 'mcc', 'hcc', 'tcc', 't2m', 'sp']
            
            ################################### 差分
            sfc_rad = [
                'ssrd', 'strd', 'ssr', 'str', 'tsr', 'ttr', 'tsrc', 'ttrc', 'ssrc',
                'strc', 'fdir', 'cdir'
            ]  #辐射变量提取
            # 合并所选择的数据集 U,V 19个气压层的数据

            #瞬时变量
            sfc_name = ['lcc', 'mcc', 'hcc', 'tcc', 't2m', 'sp',]

            ds_sfc = ds[sfc_name]  #气象变量，瞬时值

            ds_rad = ds[sfc_rad]  #辐射变量，累计值

            ds_rad_diff = ds_rad.diff(dim='valid_time') / (3600 / 3)  # 不用管这个转换是否有问题，直接差分出辐射变量的瞬时值

            merge_ds_all = xr.merge([ds_sfc, ds_rad_diff])
            
            # 合并所选择的数据集 U,V 19个气压层的数据
            merge_ds_all = merge_ds_all[col_names]
            
            try:
                filter_ds = merge_ds_all.sel(valid_time=slice(date_utc_list[i]+datetime.timedelta(minutes=(-15*3)),date_utc_list[i]+datetime.timedelta(minutes=(15*3))))
                if filter_ds.dims["valid_time"] != 7:
                    diff = filter_ds.dims["valid_time"] - 7
                    filter_ds=ds.sel(valid_time=slice(date_utc_list[i]+datetime.timedelta(minutes=(-15*3)),date_utc_list[i]+datetime.timedelta(minutes=(15*3-15*diff))))
                #根据实发数据的时间索引，匹配NWP
            except Exception as e:
                print(f'select data wa : {e}')
                continue
            ssrd = filter_ds['ssrd'].values
            strd = filter_ds['strd'].values
            ssr = filter_ds['ssr'].values
            str = filter_ds['str'].values
            tsr = filter_ds['tsr'].values
            ttr = filter_ds['ttr'].values
            tsrc = filter_ds['tsrc'].values
            ttrc = filter_ds['ttrc'].values
            ssrc = filter_ds['ssrc'].values
            strc = filter_ds['strc'].values
            fdir = filter_ds['fdir'].values
            cdir = filter_ds['cdir'].values
            lcc = filter_ds['lcc'].values
            mcc = filter_ds['mcc'].values
            hcc = filter_ds['hcc'].values
            tcc = filter_ds['tcc'].values
            t2m = filter_ds['t2m'].values
            sp = filter_ds['sp'].values

            ground_layer = np.stack((ssrd,strd,ssr,str,tsr,ttr,tsrc,ttrc,ssrc,strc,fdir,cdir,lcc,mcc,hcc,tcc,t2m,sp), axis=0)

            nwp_values = ground_layer

            if nwp_values.shape[1]==7:
                nwp_list.append(nwp_values)
                date_list.append(date_utc_list[i])
            else:
                continue
        return nwp_list, date_list, file_utc_time

    else:
        return None,None,None