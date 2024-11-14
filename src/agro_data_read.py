import numpy as np
import datetime
import xarray as xr
from metpy.units import units
import metpy.calc as mp_calc
import os
import pandas as pd
import sys
from metpy.calc import relative_humidity_from_dewpoint
# from config.config import out_agro_prefix, ecmwf_nwp_prefix,stnFile
# from src.call_api2 import call_api

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


def get_nwp(time_bjt_now: datetime.datetime, file_point: str, ecmwf_nwp_prefix, stnFile, out_agro_prefix, df_pre_winds, ftime, windSpeedCol, dataType):

    #获取文件时间
    file_utc_time = convert_run_time(time_bjt_now)

    utc_file_timestr = file_utc_time.strftime('%Y%m%d%H')

    if dataType == 'ECMWF':
        nwp_file = os.path.join(ecmwf_nwp_prefix, 'jiangsu_{}.nc').format(utc_file_timestr)
        # 如果这一报的nwp缺失，则用上一报的进行预测
        if not os.path.exists(nwp_file):
            file_utc_time2 = file_utc_time + datetime.timedelta(hours=-12)
            utc_file_timestr2 = file_utc_time2.strftime('%Y%m%d%H')
            nwp_file = os.path.join(ecmwf_nwp_prefix, 'jiangsu_{}.nc').format(utc_file_timestr2)
    elif dataType == 'GFS':
        nwp_file = os.path.join(ecmwf_nwp_prefix, 'jiangsu_wind_{}.nc').format(utc_file_timestr)
        # 如果这一报的nwp缺失，则用上一报的进行预测
        if not os.path.exists(nwp_file):
            file_utc_time2 = file_utc_time + datetime.timedelta(hours=-12)
            utc_file_timestr2 = file_utc_time2.strftime('%Y%m%d%H')
            nwp_file = os.path.join(ecmwf_nwp_prefix, 'jiangsu_wind_{}.nc').format(utc_file_timestr2)
    print(nwp_file)

    try:
        ds = xr.open_dataset(nwp_file)

        # sfc_rad = ['ssr']  #辐射变量
        # sfc_name = ['vis', 'sp', 'tcc', 't2m', 'u10', 'v10', 'u100', 'v100']  #瞬时变量

        # sfc_rad = ['ssrd', 'strd', 'ssr', 'str', 'tsr', 'ttr', 'tsrc', 'ttrc', 'ssrc', 'strc', 'fdir', 'cdir']
        # sfc_rad = ['ssrd', 'fdir']
        sfc_name = ['t2m', 'sp', 'u10', 'v10', 'u100', 'v100', 'd2m']

        # pl_name = ['u', 'v']  #气压层变量

        #瞬时变量
        ds_sfc = ds[sfc_name]

        #计算U,V风

        # wind_speed_10 = mp_calc.wind_speed(ds_sfc['u10'].values * units('m/s'),
        #                                    ds_sfc['v10'].values *
        #                                    units('m/s')).magnitude
        wind_dir_10 = mp_calc.wind_direction(ds_sfc['u10'].values * units('m/s'), ds_sfc['v10'].values * units('m/s')).magnitude
        # wind_speed_100 = mp_calc.wind_speed(
        #     ds_sfc['u100'].values * units('m/s'),
        #     ds_sfc['v100'].values * units('m/s')).magnitude
        wind_dir_100 = mp_calc.wind_direction(ds_sfc['u100'].values * units('m/s'), ds_sfc['v100'].values * units('m/s')).magnitude
        rh=relative_humidity_from_dewpoint((ds_sfc['t2m'].values-273.15) * units.degC, (ds_sfc['d2m']-273.15) * units.degC).to('percent')

        # wind_speed_10_array = xr.DataArray(
        #     wind_speed_10,
        #     coords={
        #         'valid_time': ds_sfc['u10'].valid_time,
        #         'latitude': ds_sfc['u10'].latitude,
        #         'longitude': ds_sfc['u10'].longitude
        #     },
        #     dims=('valid_time', 'latitude', 'longitude'))

        wind_dir_10_array = xr.DataArray(
            wind_dir_10,
            coords={
                'valid_time': ds_sfc['u10'].valid_time,
                'latitude': ds_sfc['u10'].latitude,
                'longitude': ds_sfc['u10'].longitude
            },
            dims=('valid_time', 'latitude', 'longitude'))
        
        rh_array = xr.DataArray(
            rh,
            coords={
                'valid_time': ds_sfc['t2m'].valid_time,
                'latitude': ds_sfc['t2m'].latitude,
                'longitude': ds_sfc['t2m'].longitude
            },
            dims=('valid_time', 'latitude', 'longitude'))

        # wind_speed_100_array = xr.DataArray(wind_speed_100,
        #                                     coords={
        #                                         'valid_time':
        #                                         ds_sfc['u10'].valid_time,
        #                                         'latitude':
        #                                         ds_sfc['u10'].latitude,
        #                                         'longitude':
        #                                         ds_sfc['u10'].longitude
        #                                     },
        #                                     dims=('valid_time', 'latitude',
        #                                           'longitude'))

        wind_dir_100_array = xr.DataArray(
            wind_dir_100,
            coords={
                'valid_time': ds_sfc['u10'].valid_time,
                'latitude': ds_sfc['u10'].latitude,
                'longitude': ds_sfc['u10'].longitude
            },
            dims=('valid_time', 'latitude', 'longitude'))

        # ds_sfc['wins10'] = wind_speed_10_array
        ds_sfc['wind10'] = wind_dir_10_array

        # ds_sfc['wins100'] = wind_speed_100_array
        ds_sfc['wind100'] = wind_dir_100_array
        
        ds_sfc['rh'] = rh_array
        #辐射变量
        # ds_rad = ds[sfc_rad].diff(dim='valid_time') / (3600 / 3)

        # #气压层提取:
        # ds_level1000 = ds[pl_name].sel(isobaricInhPa=1000)

        # ds_level950 = ds[pl_name].sel(isobaricInhPa=950)

        # ds_level925 = ds[pl_name].sel(isobaricInhPa=925)

        # ds_level900 = ds[pl_name].sel(isobaricInhPa=900)

        # #重命名：
        # ds_level1000 = ds_level1000.rename({
        #     'u': 'u_1000hPa',
        #     'v': 'v_1000hPa'
        # })

        # ds_level950 = ds_level950.rename({'u': 'u_950hPa', 'v': 'v_950hPa'})

        # ds_level925 = ds_level925.rename({'u': 'u_925hPa', 'v': 'v_925hPa'})

        # ds_level900 = ds_level900.rename({'u': 'u_900hPa', 'v': 'v_900hPa'})

        #最后汇总的所有变量
        # ds_all = xr.merge([ds_sfc, ds_rad], compat='override')
        ds_all = ds_sfc
        #打开站点表
        # df_points = pd.read_csv(
        #     '/home/gpusr/product/henan_power_push/single_forecast/src/station_point.csv'
        # )
        df_points = pd.read_csv(stnFile)

        df_points.set_index('ST_ID', inplace=True)

        #推送部分：重点

        #变量

        #cols = ['temp', 'pressure', 'wins10','wind10','wins100','wind100','u1000','v1000','u950','v950','u925','v925','u900','v900']
        # cols = ['dsr','tcc','vis','wins10','temp']
        # all_col = [
        #     'vis', 'tcc', 't2m', 'sp', 'wins10', 'wind10', 'wins100',
        #     'wind100', 'ssr', 'u_1000hPa', 'v_1000hPa', 'u_950hPa', 'v_950hPa',
        #     'u_925hPa', 'v_925hPa', 'u_900hPa', 'v_900hPa'
        # ]

        # all_col = ['ssrd', 'strd', 'ssr', 'str', 'tsr', 'ttr', 'tsrc', 'ttrc', 'ssrc', 'strc', 'fdir', 'cdir', 'lcc', 'mcc', 'hcc', 'tcc', 't2m', 'sp', 'wins10', 'wind10', 'd2m']
        all_col = ['t2m', 'rh', 'sp', 'wind10']

        for station_id, idf in df_points.iterrows():
            #st_id
            #获取类型：
            station_class = idf['CLASS']

            df_i_station = ds_all.sel(longitude=idf['LON'], latitude=idf['LAT'], method='nearest').to_dataframe()
            df_i_station = df_i_station[all_col]

            # df_i_station.columns = [
            #     'vis', 'tcc', 'temp', 'pressure', 'wins10', 'wind10',
            #     'wins100', 'wind100', 'dsr', 'u1000', 'v1000', 'u950', 'v950',
            #     'u925', 'v925', 'u900', 'v900'
            # ]
            
            # df_i_station.columns = all_col

            # print(df_i_station)

            df_i_station.columns = ['temp', 'humidness', 'pressure', 'wind_direction']
            df_i_station.insert(0, 'DATETIME', df_i_station.index + pd.Timedelta(hours=8))
            df_i_station['temp'] = df_i_station['temp'] - 273.15

            # df_i_station.drop(columns=['valid_time'],inplace=True)
            # print(df_i_station.header)
            # solar_col = ['dsr', 'tcc', 'vis', 'wins10', 'temp']
            # wind_col = [
            #     'temp', 'pressure', 'wins10', 'wind10', 'wins100', 'wind100',
            #     'u1000', 'v1000', 'u950', 'v950', 'u925', 'v925', 'u900',
            #     'v900'
            # ]
            #设置时间
            #df_i_station.reset_index(inplace=True)
            # df_i_station['DATETIME'] = df_i_station.index + pd.Timedelta(hours=8)

            if station_class == 'wind':

                file_bjt_time: datetime.datetime = file_utc_time + datetime.timedelta(hours=8)
                file_bjt_time_str = file_bjt_time.strftime('%Y%m%d')
                df_i_station.reset_index(inplace=True)
                df_i_station = df_i_station.interpolate(method='linear', axis=0)

                # time_bjt_now_str = time_bjt_now.strftime('%Y%m%d_%H')

                # outpath_agro = os.path.join(out_agro_prefix, ftime[:8])  # 系统时间
                # os.makedirs(outpath_agro, exist_ok=True)

                # out_file = os.path.join(outpath_agro, f'wind_' + idf['NAME'] + '_{}.csv').format(ftime)
                df_i_station.drop(columns=["valid_time"], inplace=True)
                
                df_i_station = pd.merge(df_i_station, df_pre_winds, on="DATETIME", how='inner')
                df_i_station.rename(columns={windSpeedCol: 'wind_speed'}, inplace=True)
                df_i_station.insert(0, 'id', windSpeedCol)
                df_i_station.insert(2, "Elevation", 10)
                # df_i_station.to_csv(out_file, encoding='utf-8-sig', index=False)
                # out_file=os.path.join(outpath_agro,f'yl_allvalue_{station_class}_{station_id}_{time_bjt_now_str}.txt')
                # csv_to_txt2_agro(df_csv=df_i_station,id='#'+station_class,SITENAME=station_id,cols=wind_col,plantType='wind',outfile=out_file)
            return df_i_station
            
            # elif station_class == 'solar':
            #     file_bjt_time: datetime.datetime = file_utc_time + datetime.timedelta(hours=8)
            #     file_bjt_time_str = file_bjt_time.strftime('%Y%m%d')
            #     df_i_station.reset_index(inplace=True)
            #     df_i_station.loc[df_i_station['全波段水平面总辐射'] < 0, '全波段水平面总辐射'] = 0
            #     df_i_station.loc[df_i_station['可见光水平面总辐射'] < 0, '可见光水平面总辐射'] = 0
            #     df_i_station['气压'] = df_i_station['气压'] / 100
            #     df_i_station['环境温度'] = df_i_station['环境温度'] - 273.15

            #     # for colname in solar_col:
            #     df_i_station.drop(columns=['valid_time'], inplace=True)
            #     df_i_station = df_i_station.interpolate(method='bfill')
            #     df_i_station = df_i_station.apply(lambda x: round(x, 3))
            #     # df_i_station24['valid_time'] = pd.to_datetime(df_i_station24['valid_time'])
            #     starttime = time_bjt_now
            #     endtime24 = starttime + datetime.timedelta(hours=24)
            #     endtime72 = starttime + datetime.timedelta(hours=72)

            #     df_i_station24 = df_i_station.copy()
            #     df_i_station24 = df_i_station24[df_i_station24['时间'] >= starttime]
            #     # df_i_station24 = df_i_station24[(df_i_station24['时间'] >= starttime) & (df_i_station24['时间'] <= endtime24)]
            #     # df_i_station24['时间'] = [x.strftime('%Y/%m/%d %H:%M') for x in df_i_station24['时间']]

            #     df_i_station72 = df_i_station.copy()
            #     df_i_station72 = df_i_station72[df_i_station72['时间'] >= starttime]
            #     # df_i_station72 = df_i_station72[(df_i_station72['时间'] >= starttime) & (df_i_station72['时间'] <= endtime72)]
            #     # df_i_station72['时间'] = [x.strftime('%Y/%m/%d %H:%M') for x in df_i_station72['时间']]

            #     outpath_agro = os.path.join(out_agro_prefix, time_bjt_now.strftime('%Y%m%d'))
            #     os.makedirs(outpath_agro, exist_ok=True)

            #     ################################ 保存并调用接口
            #     # 9点文件
            #     # 计算调度时间
            #     today = datetime.datetime.now().date()  
            #     oclock = datetime.datetime.combine(today, datetime.time(9, 0))
            #     dispatchTime9 = oclock.strftime('%Y-%m-%d %H:%M:%S')
            #     outftime9 = oclock.strftime('%Y%m%d%H%M%S')
                
            #     out_file24_09 = os.path.join(outpath_agro, f'NWP24_solar_' + idf['NAME'] + '_{}.csv').format(outftime9)
            #     out_file72_09 = os.path.join(outpath_agro, f'NWP72_solar_' + idf['NAME'] + '_{}.csv').format(outftime9)

            #     df_i_station24.to_csv(out_file24_09, encoding='utf-8-sig', index=False)
            #     df_i_station72.to_csv(out_file72_09, encoding='utf-8-sig', index=False)

            #     # fbTime = file_utc_time + datetime.timedelta(hours=8)
            #     # fbTime = fbTime.strftime('%Y-%m-%d %H:%M:%S')

            #     # today = datetime.datetime.now().date()  
            #     # nine_oclock = datetime.datetime.combine(today, datetime.time(9, 0))
            #     # dispatchTime = nine_oclock.strftime('%Y-%m-%d %H:%M:%S')

            #     # 9点调度
            #     # call_api(electricFieldName=idf['NAME'], 
            #     #          fbTime=dispatchTime9,
            #     #          fbInterval='', 
            #     #          forecastType='0', 
            #     #          fbType='2',
            #     #          filePath=out_file24_09[27:], 
            #     #          fileName=os.path.basename(out_file24_09),
            #     #          dispatchTime=dispatchTime9)
            #     # call_api(electricFieldName=idf['NAME'], 
            #     #          fbTime=dispatchTime9, 
            #     #          fbInterval='', 
            #     #          forecastType='0', 
            #     #          fbType='3',
            #     #          filePath=out_file72_09[27:], 
            #     #          fileName=os.path.basename(out_file72_09),
            #     #          dispatchTime=dispatchTime9)
                
            #     # # 13点调度
            #     # oclock13 = datetime.datetime.combine(today, datetime.time(13, 0))
            #     # dispatchTime13 = oclock13.strftime('%Y-%m-%d %H:%M:%S')
            #     # outftime13 = oclock13.strftime('%Y%m%d%H%M%S')

            #     # out_file24_13 = os.path.join(outpath_agro, f'NWP24_solar_' + idf['NAME'] + '_{}.csv').format(outftime13)
            #     # out_file72_13 = os.path.join(outpath_agro, f'NWP72_solar_' + idf['NAME'] + '_{}.csv').format(outftime13)

            #     # df_i_station24.to_csv(out_file24_13, encoding='utf-8-sig', index=False)
            #     # df_i_station72.to_csv(out_file72_13, encoding='utf-8-sig', index=False)

            #     # # 13点调用接口
            #     # call_api(electricFieldName=idf['NAME'], 
            #     #          fbTime=dispatchTime13,
            #     #          fbInterval='', 
            #     #          forecastType='0', 
            #     #          fbType='2',
            #     #          filePath=out_file24_13[27:], 
            #     #          fileName=os.path.basename(out_file24_13),
            #     #          dispatchTime=dispatchTime13)
            #     # call_api(electricFieldName=idf['NAME'], 
            #     #          fbTime=dispatchTime13, 
            #     #          fbInterval='', 
            #     #          forecastType='0', 
            #     #          fbType='3',
            #     #          filePath=out_file72_13[27:], 
            #     #          fileName=os.path.basename(out_file72_13),
            #     #          dispatchTime=dispatchTime13)

            #     # if not os.path.exists(out_file24_09) or not os.path.exists(out_file72_09)\
            #     #     or not os.path.exists(out_file24_13) or not os.path.exists(out_file72_13):
            #     # if not os.path.exists(out_file24_09) or not os.path.exists(out_file72_09):
            #     #     sys.exit(1)


    except Exception as e:
        print(e)
        sys.exit(1)
