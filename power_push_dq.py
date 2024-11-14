from src.model_solar import WeatherCNN3D as model_solar
from src.model_wind import WeatherCNN3D as model_wind
import os,re
import pandas as pd
import datetime
import numpy as np
from config.config import *
import torch
from torch import nn
from src.d1d_read import get_data_solar, get_data
import pickle, shutil
from tqdm import tqdm
from src.logging_tool import Logger
from src.call_api2 import call_api
import warnings
warnings.filterwarnings('ignore')


def convert_time_save(time_now:datetime.datetime):
    hours=time_now.hour
    if hours>14:
        hours='20'
    else:
        hours='08'
    return time_now.strftime('%Y%m%d')+hours


# 数据库入库的部分

# 导入8张表
from src.orm_model import PowerWindPointCdqTable,PowerWindPointDqTable,PowerWindPointZqTable,PowerWindPointDlTable

from src.orm_model import PowerSolarPointCdqTable,PowerSolarPointZqTable,PowerSolarPointDqTable,PowerSolarPointDlTable
from src.orm_model.Engine import GetEngine,GetSession


#def commit_wins(session,df:pd.DataFrame,fb_time,st_id,height,orm_model:PowerWindPointDqTable):
#    '''
#    风场站提交数据
#   '''
#    for label,idf in df.iterrows():
#        entity=orm_model(
#           data_source_id=100,
#           st_id=st_id,
#            fb_time=fb_time,
#            data_time=idf['DATETIME'],
#            data_feature_value=idf['功率'],
#            wins=idf['轮毂高度风速'],
#            height=height,
#            create_time=datetime.datetime.now(),
#        )
#        session.merge(entity)
#    
#    session.commit()
    

#todo：风场站提交数据库
def commit_wins(session,df:pd.DataFrame,fb_time,st_id,height,orm_model, weatherName):
    '''
    风场站提交数据到数据库
    '''
    if weatherName == '辐照度':
        for label,idf in df.iterrows():
            entity=orm_model(
                data_source_id=100,
                site_id=st_id,
                fb_time=fb_time,
                data_time=idf['DATETIME'],
                data_feature_value=idf['功率'],
                irr=idf[weatherName],
                create_time=datetime.datetime.now(),
            )
            session.merge(entity)    
    else:
        for label,idf in df.iterrows():
            entity=orm_model(
                data_source_id=100,
                site_id=st_id,
                fb_time=fb_time,
                data_time=idf['DATETIME'],
                data_feature_value=idf['功率'],
                winds=idf[weatherName],
                height=height,
                create_time=datetime.datetime.now(),
            )
            session.merge(entity)
    
    session.commit()

def commit_solar(session,df:pd.DataFrame,fb_time,st_id,height,orm_model:PowerWindPointDqTable):
    '''
    风场站提交数据到数据库
    '''
    for label,idf in df.iterrows():
        entity=orm_model(
            data_source_id=100,
            site_id=st_id,
            fb_time=fb_time,
            data_time=idf['DATETIME'],
            data_feature_value=idf['功率'],
            irr=idf['辐照度'],
            height=height,
            create_time=datetime.datetime.now(),
        )
        session.merge(entity)
    
    session.commit()
        
    


def get_solar_n10(Session, WeatherCNN3D, modelFile, paramFile, extent, time_now, nwppath, outdir, ftime):
    """
    基于EC进行功率和辐照度
    """
    
    time_now_str_exec=convert_time_save(time_now)
    device = torch.device('cpu')
    model =WeatherCNN3D(mattention=0.0,mlp_drop=0.3,fc=1024,num_heads=64,at_fun='tanh')
    model=model.to(device)
    # model=nn.DataParallel(model)
    # model.load_state_dict(torch.load(model_file, map_location = torch.device('cpu')))
    model.load_state_dict(torch.load(modelFile, map_location=torch.device('cpu')))
    model=model.to(torch.device('cpu'))
    model.eval()  # 将模型设置为评估模式
    print('模型加载完成')

    lonmin, lonmax, latmin, latmax = extent[0], extent[1], extent[2], extent[3]
    # nwp_list, date_list, file_utc_time = get_data(time_now, nwppath, lonmin, lonmax, latmin, latmax, "ECMWF")
    nwp_list, date_list, file_utc_time = get_data_solar(time_now, nwppath, lonmin, lonmax, latmin, latmax, "ECMWF")
    print(len(nwp_list))

    with open(paramFile, 'rb') as f:
        train_config = pickle.load(f)
    nwp_mean, nwp_std, label_mean, label_std = train_config['nwp_mean'], train_config['nwp_std'], train_config['label_mean'], train_config['label_std']
    print('参数加载完成')

    item_dict=[]
    for (inwp, idate) in tqdm(zip(nwp_list, date_list)):

        item = dict()
        inwp = inwp[np.newaxis, :]
        inwp = (inwp - nwp_mean[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]) / nwp_std[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        #print(inwp.shape)
        input_data = torch.tensor(inwp, dtype=torch.float)
        with torch.no_grad():
            output = model(input_data.to(device))

        y_pred = output.detach().cpu().numpy()[0]
        y_pred = y_pred * label_std + label_mean

        item['DATETIME'] = idate + datetime.timedelta(hours=8)
        item['power1'] = y_pred[0]
        item['power2'] = y_pred[1]
        item['power3'] = y_pred[2]
        item['power4'] = y_pred[3]
        item['power5'] = y_pred[4]
        item['weather1'] = y_pred[5]
        item['weather2'] = y_pred[6]
        item['weather3'] = y_pred[7]
        item['weather4'] = y_pred[8]
        item['weather5'] = y_pred[9]
        # item['001'] = y_pred[0]
        # item['002'] = y_pred[1]
        # item['003'] = y_pred[2]
        # item['wind_001'] = y_pred[4]
        # item['wind_002'] = y_pred[5]
        # item['wind_003'] = y_pred[6]

        item_dict.append(item)

    df = pd.DataFrame(item_dict)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df['DATETIME'] = [x.astype('datetime64[s]') for x in df['DATETIME'].values]

    df[['power1', 'power2', 'power3', 'power4', 'power5', 'weather1', 'weather2', 'weather3', 'weather4', 'weather5']] = \
        df[['power1', 'power2', 'power3', 'power4', 'power5', 'weather1', 'weather2', 'weather3', 'weather4', 'weather5']].astype(float).round(3)
    df[df[['power1', 'power2', 'power3', 'power4', 'power5', 'weather1', 'weather2', 'weather3', 'weather4', 'weather5']] < 0] = 0
    
    power_col=['DATETIME','功率']
    agro_col=['DATETIME','辐照度']
    
    df1 = df[["DATETIME", "power1", "weather1"]]
    df1.columns = ["DATETIME", "功率", "辐照度"]
    
    df2 = df[["DATETIME", "power2", "weather2"]]
    df2.columns = ["DATETIME", "功率", "辐照度"]
    
    df3 = df[["DATETIME", "power3", "weather3"]]
    df3.columns = ["DATETIME", "功率", "辐照度"] 

    df4 = df[["DATETIME", "power4", "weather4"]]
    df4.columns = ["DATETIME", "功率", "辐照度"]   
    
    df5 = df[["DATETIME", "power5", "weather5"]]
    df5.columns = ["DATETIME", "功率", "辐照度"]  
    
    # objlog.logmessage("1-forecast power and weather")

    ########################## 保存
    # 短临 24小时
    start_time_24 = time_now.replace(hour=0, minute=0, second=0)
    end_time_24 = start_time_24 + datetime.timedelta(hours=24)
    df_24_1 = df1[(df1["DATETIME"] >= start_time_24) & (df1["DATETIME"] <= end_time_24)]
    df_24_2 = df2[(df2["DATETIME"] >= start_time_24) & (df2["DATETIME"] <= end_time_24)]
    df_24_3 = df3[(df3["DATETIME"] >= start_time_24) & (df3["DATETIME"] <= end_time_24)]
    df_24_4 = df4[(df4["DATETIME"] >= start_time_24) & (df4["DATETIME"] <= end_time_24)]
    df_24_5 = df5[(df5["DATETIME"] >= start_time_24) & (df5["DATETIME"] <= end_time_24)]
    
    # # 入库超短期CDQ_WIND_EC_
    # # 保存
    fileDL1 = os.path.join(outdir, 'CDQ_SOLAR_EC_' + time_now_str_exec + '_CZ_1.csv')
    fileDL2 = os.path.join(outdir, 'CDQ_SOLAR_EC_' + time_now_str_exec + '_CZ_2.csv')
    fileDL3 = os.path.join(outdir, 'CDQ_SOLAR_EC_' + time_now_str_exec + '_CZ_3.csv')
    fileDL4 = os.path.join(outdir, 'CDQ_SOLAR_EC_' + time_now_str_exec + '_CZ_4.csv')
    fileDL5 = os.path.join(outdir, 'CDQ_SOLAR_EC_' + time_now_str_exec + '_CZ_5.csv')
    # df_24_1[power_col].to_csv(fileDL1, index=False, encoding='utf-8-sig')
    # df_24_2[power_col].to_csv(fileDL2, index=False, encoding='utf-8-sig')
    # df_24_3[power_col].to_csv(fileDL3, index=False, encoding='utf-8-sig')
    # df_24_4[power_col].to_csv(fileDL4, index=False, encoding='utf-8-sig')
    # df_24_5[power_col].to_csv(fileDL5, index=False, encoding='utf-8-sig')
    
    # 中期（8天）
    start_time = time_now.replace(hour=0, minute=0, second=0) + datetime.timedelta(days=1)
    
    df11 = df1[df1["DATETIME"] >= start_time]
    df12 = df2[df2["DATETIME"] >= start_time]
    df13 = df3[df3["DATETIME"] >= start_time]
    df14 = df4[df4["DATETIME"] >= start_time]
    df15 = df5[df5["DATETIME"] >= start_time]
    
    # 保存短期
    file1 = os.path.join(outdir, 'DQ_SOLAR_EC_' + time_now_str_exec + '_CZ_1.csv')
    file2 = os.path.join(outdir, 'DQ_SOLAR_EC_' + time_now_str_exec + '_CZ_2.csv')
    file3 = os.path.join(outdir, 'DQ_SOLAR_EC_' + time_now_str_exec + '_CZ_3.csv')
    file4 = os.path.join(outdir, 'DQ_SOLAR_EC_' + time_now_str_exec + '_CZ_4.csv')
    file5 = os.path.join(outdir, 'DQ_SOLAR_EC_' + time_now_str_exec + '_CZ_5.csv')
    
    filejz=os.path.join(outdir, 'DQ_SOLAR_EC_' + time_now_str_exec + '_JZ_山东省.csv')
    
    # 功率入库
    df11[power_col].to_csv(file1, index=False, encoding='utf-8-sig') 
    df12[power_col].to_csv(file2, index=False, encoding='utf-8-sig')
    df13[power_col].to_csv(file3, index=False, encoding='utf-8-sig')
    df14[power_col].to_csv(file4, index=False, encoding='utf-8-sig')
    df15[power_col].to_csv(file5, index=False, encoding='utf-8-sig')
    # objlog.logmessage("2-save ZQ result")
    dfjz=df15[power_col]
    dfjz['功率']=dfjz['功率']*1000
    dfjz.to_csv(filejz,index=False,encoding='utf-8-sig')    


    file6 = os.path.join(outdir, 'QX_SOLAR_EC_' + time_now_str_exec + '_CZ_1.csv')
    file7 = os.path.join(outdir, 'QX_SOLAR_EC_' + time_now_str_exec + '_CZ_2.csv')
    file8 = os.path.join(outdir, 'QX_SOLAR_EC_' + time_now_str_exec + '_CZ_3.csv')
    file9 = os.path.join(outdir, 'QX_SOLAR_EC_' + time_now_str_exec + '_CZ_4.csv')
    file10 = os.path.join(outdir, 'QX_SOLAR_EC_' + time_now_str_exec + '_CZ_5.csv')      
    df11[agro_col].to_csv(file6, index=False, encoding='utf-8-sig')
    df12[agro_col].to_csv(file7, index=False, encoding='utf-8-sig')
    df13[agro_col].to_csv(file8, index=False, encoding='utf-8-sig')
    df14[agro_col].to_csv(file9, index=False, encoding='utf-8-sig')
    df15[agro_col].to_csv(file10, index=False, encoding='utf-8-sig')
    
    
    return fileDL1, fileDL2, fileDL3, fileDL4, fileDL5, file1, file2, file3, file4, file5, file6, file7, file8, file9, file10

def get_wind_n10(Session,WeatherCNN3D, modelFile, paramFile, extent, time_now, nwppath, outdir, ftime):
    """
    基于EC进行功率和风速预报
    """
    
    time_now_str_exec=convert_time_save(time_now)
    device = torch.device('cpu')
    model =WeatherCNN3D(mattention=0.0,mlp_drop=0.3,fc=1024,num_heads=64,at_fun='tanh')
    model=model.to(device)
    # model=nn.DataParallel(model)
    # model.load_state_dict(torch.load(model_file, map_location = torch.device('cpu')))
    model.load_state_dict(torch.load(modelFile, map_location=torch.device('cpu')))
    model.to(device=torch.device('cpu'))
    model.eval()  # 将模型设置为评估模式
    print('模型加载完成')

    lonmin, lonmax, latmin, latmax = extent[0], extent[1], extent[2], extent[3]
    nwp_list, date_list, file_utc_time = get_data(time_now, nwppath, lonmin, lonmax, latmin, latmax, "ECMWF")
    # nwp_list, date_list, file_utc_time = get_data_solar(time_now, nwppath, lonmin, lonmax, latmin, latmax, "ECMWF")
    print(len(nwp_list))

    with open(paramFile, 'rb') as f:
        train_config = pickle.load(f)
    nwp_mean, nwp_std, label_mean, label_std = train_config['nwp_mean'], train_config['nwp_std'], train_config['label_mean'], train_config['label_std']
    print('参数加载完成')

    item_dict=[]
    for (inwp, idate) in tqdm(zip(nwp_list, date_list)):

        item = dict()
        inwp = inwp[np.newaxis, :]
        inwp = (inwp - nwp_mean[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]) / nwp_std[np.newaxis, :, np.newaxis, np.newaxis, np.newaxis]
        #print(inwp.shape)
        input_data = torch.tensor(inwp, dtype=torch.float)
        with torch.no_grad():
            output = model(input_data.to(device))

        y_pred = output.detach().cpu().numpy()[0]
        y_pred = y_pred * label_std + label_mean

        item['DATETIME'] = idate + datetime.timedelta(hours=8)
        item['power1'] = y_pred[0]
        item['power2'] = y_pred[1]
        item['power3'] = y_pred[2]
        item['power4'] = y_pred[3]
        item['power5'] = y_pred[4]
        item['s10_1'] = y_pred[5]
        item['s10_2'] = y_pred[6]
        item['s10_3'] = y_pred[7]
        item['s10_4'] = y_pred[8]
        item['s10_5'] = y_pred[9]
        item['s70_1'] = y_pred[10]
        item['s70_2'] = y_pred[11]
        item['s70_3'] = y_pred[12]
        item['s70_4'] = y_pred[13]
        item['s70_5'] = y_pred[14]
        item['hub_1'] = y_pred[15]
        item['hub_2'] = y_pred[16]
        item['hub_3'] = y_pred[17]
        item['hub_4'] = y_pred[18]
        item['hub_5'] = y_pred[19]


        item_dict.append(item)

    df = pd.DataFrame(item_dict)
    df['DATETIME'] = pd.to_datetime(df['DATETIME'])
    df['DATETIME'] = [x.astype('datetime64[s]') for x in df['DATETIME'].values]

    df[['power1', 'power2', 'power3', 'power4', 'power5', 's10_1', 's10_2', 's10_3', 's10_4', 's10_5', \
            's70_1', 's70_2', 's70_3', 's70_4', 's70_5', 'hub_1', 'hub_2', 'hub_3', 'hub_4', 'hub_5']] = \
        df[['power1', 'power2', 'power3', 'power4', 'power5', 's10_1', 's10_2', 's10_3', 's10_4', 's10_5', \
                's70_1', 's70_2', 's70_3', 's70_4', 's70_5', 'hub_1', 'hub_2', 'hub_3', 'hub_4', 'hub_5']].astype(float).round(3)
    df[df[['power1', 'power2', 'power3', 'power4', 'power5', 's10_1', 's10_2', 's10_3', 's10_4', 's10_5', 
        's70_1', 's70_2', 's70_3', 's70_4', 's70_5', 'hub_1', 'hub_2', 'hub_3', 'hub_4', 'hub_5']] < 0] = 0

    # 功率字段
    power_col=['DATETIME','功率']
    
    # 气象字段
    agro_col=['DATETIME','10m风速','70m风速','轮毂高度风速']
    
    df1 = df[["DATETIME", "power1", "s10_1", "s70_1", "hub_1"]]
    df1.columns = ["DATETIME", "功率", "10m风速", "70m风速", "轮毂高度风速"]
    
    df2 = df[["DATETIME", "power2", "s10_2", "s70_2", "hub_2"]]
    df2.columns = ["DATETIME", "功率", "10m风速", "70m风速", "轮毂高度风速"]
    
    df3 = df[["DATETIME", "power3", "s10_3", "s70_3", "hub_3"]]
    df3.columns = ["DATETIME", "功率", "10m风速", "70m风速", "轮毂高度风速"]

    df4 = df[["DATETIME", "power4", "s10_4", "s70_4", "hub_4"]]
    df4.columns = ["DATETIME", "功率", "10m风速", "70m风速", "轮毂高度风速"]   
    
    df5 = df[["DATETIME", "power5", "s10_5", "s70_5", "hub_5"]]
    df5.columns = ["DATETIME", "功率", "10m风速", "70m风速", "轮毂高度风速"]
    
    # objlog.logmessage("1-forecast power and weather")

    ########################## 保存
    ### 短临 24小时
    start_time_24 = time_now.replace(hour=8, minute=0, second=0)
    end_time_24 = start_time_24 + datetime.timedelta(hours=24)
    df_24_1 = df1[(df1["DATETIME"] >= start_time_24) & (df1["DATETIME"] <= end_time_24)]
    df_24_2 = df2[(df2["DATETIME"] >= start_time_24) & (df2["DATETIME"] <= end_time_24)]
    df_24_3 = df3[(df3["DATETIME"] >= start_time_24) & (df3["DATETIME"] <= end_time_24)]
    df_24_4 = df4[(df4["DATETIME"] >= start_time_24) & (df4["DATETIME"] <= end_time_24)]
    df_24_5 = df5[(df5["DATETIME"] >= start_time_24) & (df5["DATETIME"] <= end_time_24)]


    '''
    	平替场站
    6	山东威海海上风电场
    7	青岛莱西风电场
    8	国华渤中海上风电场
    9	龙源平邑风电场
    10	华能安丘滨海风电场

    国信黄海风电项目
    核源睢宁风电
    华能灌云风电
    龙源李埝风电
    深能八义集风电

    
    '''

    # 保存csv
    fileDL1 = os.path.join(outdir, 'CDQ_WIND_EC_' + time_now_str_exec + '_CZ_8.csv')
    fileDL2 = os.path.join(outdir, 'CDQ_WIND_EC_' + time_now_str_exec + '_CZ_7.csv')
    fileDL3 = os.path.join(outdir, 'CDQ_WIND_EC_' + time_now_str_exec + '_CZ_10.csv')
    fileDL4 = os.path.join(outdir, 'CDQ_WIND_EC_' + time_now_str_exec + '_CZ_9.csv')
    fileDL5 = os.path.join(outdir, 'CDQ_WIND_EC_' + time_now_str_exec + '_CZ_6.csv')
    # df_24_1[power_col].to_csv(fileDL1, index=False, encoding='utf-8-sig') # 华能灌云风电
    # df_24_2[power_col].to_csv(fileDL2, index=False, encoding='utf-8-sig') # 核源睢宁风电
    # df_24_3[power_col].to_csv(fileDL3, index=False, encoding='utf-8-sig') # 深能八义风电
    # df_24_4[power_col].to_csv(fileDL4, index=False, encoding='utf-8-sig') # 龙源李埝风电
    # df_24_5[power_col].to_csv(fileDL5, index=False, encoding='utf-8-sig') # 国信黄海风电
    
    #### 中期（8天）
    start_time = time_now.replace(hour=0, minute=0, second=0) + datetime.timedelta(days=1)

    df11 = df1[df1["DATETIME"] >= start_time]
    df12 = df2[df2["DATETIME"] >= start_time]
    df13 = df3[df3["DATETIME"] >= start_time]
    df14 = df4[df4["DATETIME"] >= start_time]
    df15 = df5[df5["DATETIME"] >= start_time]
    

    
    # 保存
    file1 = os.path.join(outdir, 'DQ_WIND_EC_' + time_now_str_exec +'_CZ_8.csv') #8
    file2 = os.path.join(outdir, 'DQ_WIND_EC_' + time_now_str_exec +'_CZ_7.csv') #7
    file3 = os.path.join(outdir, 'DQ_WIND_EC_' + time_now_str_exec+'_CZ_10.csv') #10
    file4 = os.path.join(outdir, 'DQ_WIND_EC_' + time_now_str_exec +'_CZ_9.csv')  #9
    file5 = os.path.join(outdir, 'DQ_WIND_EC_' + time_now_str_exec +'_CZ_6.csv')  #6
    filejz= os.path.join(outdir, 'DQ_WIND_EC_' + time_now_str_exec +'_JZ_山东省.csv')  #6
    df11[power_col].to_csv(file1, index=False, encoding='utf-8-sig')
    df12[power_col].to_csv(file2, index=False, encoding='utf-8-sig')
    df13[power_col].to_csv(file3, index=False, encoding='utf-8-sig')
    df14[power_col].to_csv(file4, index=False, encoding='utf-8-sig')
    df15[power_col].to_csv(file5, index=False, encoding='utf-8-sig')
    # objlog.logmessage("2-save ZQ result")
    dfjz=df15[power_col]
    dfjz['功率']=dfjz['功率']*1000
    dfjz.to_csv(filejz,index=False,encoding='utf-8-sig')    

    file6 = os.path.join(outdir, 'QX_WIND_EC_' + time_now_str_exec + '_CZ_8.csv')
    file7 = os.path.join(outdir, 'QX_WIND_EC_' + time_now_str_exec + '_CZ_7.csv')
    file8 = os.path.join(outdir, 'QX_WIND_EC_' + time_now_str_exec + '_CZ_10.csv')
    file9 = os.path.join(outdir, 'QX_WIND_EC_' + time_now_str_exec + '_CZ_9.csv')
    file10 = os.path.join(outdir, 'QX_WIND_EC_' + time_now_str_exec + '_CZ_6.csv')   
    #气象   
    df11[agro_col].to_csv(file6, index=False, encoding='utf-8-sig')
    df12[agro_col].to_csv(file7, index=False, encoding='utf-8-sig')
    df13[agro_col].to_csv(file8, index=False, encoding='utf-8-sig')
    df14[agro_col].to_csv(file9, index=False, encoding='utf-8-sig')
    df15[agro_col].to_csv(file10, index=False, encoding='utf-8-sig')

    return fileDL1, fileDL2, fileDL3, fileDL4, fileDL5, file1, file2, file3, file4, file5, file6, file7, file8, file9, file10


if __name__ == '__main__':

    logfile = os.path.join(runlogPath, 'shandong_solar.log')
    objlog = Logger(logfile=logfile)

    outdir=os.path.join(outprefix, time_now_str[:4])
    outdir=os.path.join(outdir,time_now_str[0:6])
    outdir = os.path.join(outdir, time_now_str[:8])
    os.makedirs(outdir, exist_ok=True)
    
    # 获取数据库连接
    session=None
    print(outdir)
    
    # 所有短临、短期、超短期 数据一起处理
    DLSolarFile1, DLSolarFile2, DLSolarFile3, DLSolarFile4, DLSolarFile5, \
        DQSolarFile1, DQSolarFile2, DQSolarFile3, DQSolarFile4, DQSolarFile5, \
            ZQSolarFile1, ZQSolarFile2, ZQSolarFile3, ZQSolarFile4, ZQSolarFile5 = \
        get_solar_n10(session,model_solar, model_file, param_file, extent_solar, time_now, ec_path, outdir, time_now_str)   # 光伏场9天

    objlog.logmessage("1-solar push")

    # 所有短临
    DLWindFile1, DLWindFile2, DLWindFile3, DLWindFile4, DLWindFile5,\
        DQWindFile1, DQWindFile2, DQWindFile3, DQWindFile4, DQWindFile5, \
            ZQWindFile1, ZQWindFile2, ZQWindFile3, ZQWindFile4, ZQWindFile5 = \
        get_wind_n10(session,model_wind, model_file_wind, param_file_wind, extent_wind, time_now, ec_path, outdir, time_now_str)   # 风场9天
    objlog.logmessage("2-wind push")
