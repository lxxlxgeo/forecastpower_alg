
import os,re
import pandas as pd
import datetime
import numpy as np
# from config.config import *
from src.region_data.d1d_read import get_region_data
from config.config import time_now,time_now_str
import pickle, shutil
from tqdm import tqdm
from src.logging_tool import Logger
from src.call_api2 import call_api
import warnings
from config.vision_config import *
warnings.filterwarnings('ignore')
from data_version.draw_data import Plotter
from data_version.color_config import * 

# from src.orm_model.PGengine import GetSession,GetEngine
# from src.orm_model.pg_tables.qx_region import QxRegionTable

from src.orm_model.Engine import GetEngine,GetSession #数据库的连接配置
from src.orm_model.visual_tables.regionfile import QxRegionTable #数据表ORM映射
# from src.orm_model.region_table.region_file import QxRegionTable
# from src.orm_model.Engine import GetEngine,GetSession

from data_version.convert_to_txt import convert_source_totext
def draw_data_to_map(session,extent,nwppath):
    lonmin, lonmax, latmin, latmax = extent[0], extent[1], extent[2], extent[3]
    # nwp_list, date_list, file_utc_time = get_data(time_now, nwppath, lonmin, lonmax, latmin, latmax, "ECMWF")
    nwp_list, date_list, file_utc_time,lon,lat,coor_dflist= get_region_data(time_now, nwppath, lonmin, lonmax, latmin, latmax, "ECMWF")
    #其中,NWP list 数据表,lon为经度,lat为维度
    file_bjt_time=file_utc_time+datetime.timedelta(hours=8)
    
    solar_plotter= Plotter(
        variable_name='irr',
        variables=['temp','wins','pre','t2m','d2m'],
        ticks=None,
        forecast_path_str=file_bjt_time.strftime("%Y%m%d%H"),
        data_extent=visual_total,
        map_extent=visual_total,
        proj=None,
        forecast_type="solar"
    )

    wins_plotter= Plotter(
        variable_name='wins',
        variables=['temp','wins','pre','t2m','d2m'],
        ticks=None,
        forecast_path_str=file_bjt_time.strftime("%Y%m%d%H"),
        data_extent=visual_total,
        map_extent=visual_total,
        proj=None,
        forecast_type="wins"
    )
    
    
    t2m_plotter= Plotter(
        variable_name='t2m',
        variables=['temp','wins','pre','t2m'],
        ticks=None,
        forecast_path_str=file_bjt_time.strftime("%Y%m%d%H"),
        data_extent=visual_total,
        map_extent=visual_total,
        proj=None,
        forecast_type="t2m"
    )
    
    
    d2m_plotter= Plotter(
        variable_name='t2m',
        variables=['temp','wins','pre','t2m'],
        ticks=None,
        forecast_path_str=file_bjt_time.strftime("%Y%m%d%H"),
        data_extent=visual_total,
        map_extent=visual_total,
        proj=None,
        forecast_type="d2m"
    )
    
    
    subdirectories = solar_plotter.create_subdirectories(outprefix,file_bjt_time.strftime('%Y%m%d%H'))
    for subdirectory in subdirectories.values():
        os.makedirs(subdirectory, exist_ok=True)
    
    for inwp,idate,icoor in zip(nwp_list, date_list,coor_dflist):
        print(inwp.shape)
        #其中 INWP 的 形状为 辐照度、经度、纬度
        bjt_idate=idate+datetime.timedelta(hours=8)
        
        solar_pngname=f"solar_{bjt_idate.strftime('%Y%m%d%H')}_forecast" + '.png' #辐照度
        
        winds10_pngname=f"winds10_{bjt_idate.strftime('%Y%m%d%H')}_forecast" + '.png' #风速10

        winds100_pngname=f"winds100_{bjt_idate.strftime('%Y%m%d%H')}_forecast" + '.png' #风速100
        
        rh2_pngname=f"rh2_{bjt_idate.strftime('%Y%m%d%H')}_forecast" + '.png' #2米相对湿度
        
        t2m_pngname=f"t2_{bjt_idate.strftime('%Y%m%d%H')}_forecast" + '.png' #2米气温
        
        #以下是批量替换文件名的程序
        solar_png=solar_plotter.get_output_file_path(outprefix,solar_pngname,forecast_cycle=file_bjt_time.strftime('%Y%m%d%H'))
        solar_txt=solar_png.replace('.png','.txt')
        solar_png_s=solar_png.replace('.png','_s.png')
        solar_csv=solar_png.replace('.png','.csv')
        
        
        
        winds10_png=wins_plotter.get_output_file_path(outprefix,winds10_pngname,forecast_cycle=file_bjt_time.strftime('%Y%m%d%H')) 
        winds10_txt=winds10_png.replace('.png','.txt')
        winds10_png_s=winds10_png.replace('.png','_s.png')
        winds10_csv=winds10_png.replace('.png','.csv')
        
        winds100_png=wins_plotter.get_output_file_path(outprefix,winds100_pngname,forecast_cycle=file_bjt_time.strftime('%Y%m%d%H')) 
        winds100_txt=winds100_png.replace('.png','.txt')
        winds100_png_s=winds100_png.replace('.png','_s.png')
        winds100_csv=winds100_png.replace('.png','.csv')
        
        
        rh2_png=d2m_plotter.get_output_file_path(outprefix,rh2_pngname,forecast_cycle=file_bjt_time.strftime('%Y%m%d%H')) 
        rh2_txt=rh2_png.replace('.png','.txt')
        rh2_png_s=rh2_png.replace('.png','_s.png')
        rh2_csv=rh2_png.replace('.png','.csv')
        
        
        t2m_png=t2m_plotter.get_output_file_path(outprefix,t2m_pngname,forecast_cycle=file_bjt_time.strftime('%Y%m%d%H')) 
        t2m_txt=t2m_png.replace('.png','.txt')
        t2m_png_s=t2m_png.replace('.png','_s.png')
        t2m_csv=t2m_png.replace('.png','.csv')
        
        
        '''
        #出图的形式
        # ssrd,wind_speed_10,wind_speed_100,t2m,rh2 
        df_item['ssrd']=ssrd_df
        df_item['wind_speed10']=wind_speed10_df
        df_item['wind_speed100']=wind_speed100_df
        df_item['t2m']=t2m_df
        df_item['rh2_df']=rh2_df
        '''
        
        solar_plotter.plot_map_simple(lon,lat,inwp[0,:,:],solar_cmap,soalr_norm,solar_png)  #辐照度
        solar_plotter.plot_map_simple(lon,lat,inwp[0,:,:],solar_cmap,soalr_norm,solar_png_s,True)  #辐照度邮票图
        icoor['ssrd'].to_csv(solar_csv,index=False,encoding='utf-8-sig') #辐照度csv
        convert_source_totext(lon,lat,inwp[0,:,:],solar_txt) #

        wins_plotter.plot_map_simple(lon,lat,inwp[1,:,:],wins_cmap,wins_norm,winds10_png) #10米风速
        wins_plotter.plot_map_simple(lon,lat,inwp[1,:,:],wins_cmap,wins_norm,winds10_png_s,True) #10米风速
        icoor['wind_speed10'].to_csv(winds10_csv,index=False,encoding='utf-8-sig') #辐照度csv
        convert_source_totext(lon,lat,inwp[1,:,:],winds10_txt)
        
        wins_plotter.plot_map_simple(lon,lat,inwp[2,:,:],wins_cmap,wins_norm,winds100_png) #100米风速
        wins_plotter.plot_map_simple(lon,lat,inwp[2,:,:],wins_cmap,wins_norm,winds100_png_s,True) #100米风速
        icoor['wind_speed100'].to_csv(winds100_csv,index=False,encoding='utf-8-sig') #辐照度csv
        convert_source_totext(lon,lat,inwp[2,:,:],winds100_txt)
        
        d2m_plotter.plot_map_simple(lon,lat,inwp[4,:,:],rh_cmap,rh_norm,rh2_png) #2米相对湿度 
        d2m_plotter.plot_map_simple(lon,lat,inwp[4,:,:],rh_cmap,rh_norm,rh2_png_s,True) #2米相对湿度 
        icoor['rh2_df'].to_csv(rh2_csv,index=False,encoding='utf-8-sig') #辐照度csv
        convert_source_totext(lon,lat,inwp[4,:,:],rh2_txt)
        
        t2m_plotter.plot_map_simple(lon,lat,inwp[3,:,:],temperature_cmap,temp_norm,t2m_png) #2米气气温
        t2m_plotter.plot_map_simple(lon,lat,inwp[3,:,:],temperature_cmap,temp_norm,t2m_png_s,True) #2米气气温
        icoor['t2m'].to_csv(t2m_csv,index=False,encoding='utf-8-sig') #辐照度csv
        convert_source_totext(lon,lat,inwp[3,:,:],t2m_txt)
        # print(lon.min())
        # print(lon.max())
        # print(lat.min())
        # print(lat.max())
        
        # entity=QxRegionTable(
            
        # )
        '''
        yb_type = Column(SmallInteger, comment='预报类型')
        data_from_id = Column(Integer, comment='预报数据源')
        fb_time = Column(DateTime, comment='发报时间')
        data_time = Column(DateTime, comment='预报时间')
        solar_path = Column(String(128), comment='辐照度的图片文件路径')
        temp_path = Column(String(255), comment='温度图片路径')
        pre_path = Column(String(255), comment='湿度图片路径')
        wind10_path = Column(String(128), comment='10米风速的图片文件路径')
        wind100_path = Column(String(255), comment='100米风速的图片文件路径')
        create_time = Column(DateTime, comment='创建时间')
        update_time = Column(DateTime, comment='修改时间')
        '''
        entity=QxRegionTable(
            yb_type=1,
            data_from_id=1,
            fb_time=file_bjt_time,
            data_time=bjt_idate,
            solar_path=solar_png.replace(outprefix,'draw_img'),
            solar_s_path=solar_png_s.replace(outprefix,'draw_img'),
            temp_path=t2m_png.replace(outprefix,'draw_img'),
            temp_s_path=t2m_png_s.replace(outprefix,'draw_img'),
            pre_path=rh2_png.replace(outprefix,'draw_img'),
            pre_s_path=rh2_png_s.replace(outprefix,'draw_img'),
            wind10_path=winds10_png.replace(outprefix,'draw_img'),
            wind10_s_path=winds10_png_s.replace(outprefix,'draw_img'),
            wind100_path=winds100_png.replace(outprefix,'draw_img'),
            wind100_s_path=winds100_png_s.replace(outprefix,'draw_img'),
            create_time=datetime.datetime.now()
        )
        session.merge(entity)
        
        #temp_tif=temp_ploter.get_output_file_path('temp',1,txt_out_prefix,temp_tifname,forecast_cycle=forecast_cycle_str)
        # entity=QxRegionTable(
        #     data_source_id=100,
        #     fb_time=file_bjt_time,
        #     data_time=bjt_idate,
        #     solar_name='光伏',
        #     solar_path=solar_png.replace('draw_output',''),
        #     wind_name='风场',
        #     wind_path=winds_png.replace('draw_output',''),
        #     create_time=datetime.datetime.now()
        # )
        # session.merge(entity)
        
    session.commit()

if __name__=='__main__':
    
    import datetime
    import gc 
    # #自动运行
    # start_date=datetime.datetime(2024,10,20,6,0,0)
    # end_date=datetime.datetime(2024,10,29,6,0,0)
    
    # current_time=start_date
    
    # while(current_time<end_date):
    #     gc.collect()
    #     time_now=current_time
    #     #time_now = datetime.datetime(2024, 10, 18, 8, 0, 0)
    #     #time_now=datetime.datetime.utcnow()+datetime.timedelta(hours=8)
    #     time_now_str = time_now.strftime('%Y%m%d0800')
        
    session=GetSession()
    draw_data_to_map(session,visual_total,ec_path)
    session.close()
        #current_time=current_time+datetime.timedelta(days=1)
        #gc.collect()
    
'''

class QxRegionTable(Base):
    __tablename__ = 'qx_region_table'

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    data_source_id = Column(Integer, nullable=True, comment='数据源')
    fb_time = Column(DateTime, nullable=True, comment='发报时间')
    data_time = Column(DateTime, nullable=True, comment='预报时间')
    solar_name = Column(String(128), nullable=True, comment='辐照度文件名称')
    solar_path = Column(String(128), nullable=True, comment='辐照度的图片文件路径')
    wind_name = Column(String(128), nullable=True, comment='风速的文件名称')
    wind_path = Column(String(128), nullable=True, comment='风速的图片文件路径')
    create_time = Column(DateTime, nullable=True, comment='创建时间')
    update_time = Column(DateTime, nullable=True, comment='修改时间')

    __table_args__ = (
        Index('idx_normal', 'fb_time'),
        {"comment": "气象区域预报结果表-风速图片+光伏图片"}
    )

'''