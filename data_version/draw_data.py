# -*-coding: utf-8 -*-
"""
Created on 2023/8/14 5:42

@author: lxce
"""
# -*- coding: utf-8 -*-
"""
Created on 2023/8/12 6:06

@author: lxce
"""
# %%
# latS = 42.0
# latN = 55.0
# lonL = 120.0
# lonR = 135.5
from matplotlib.patches import PathPatch
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.path import Path
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature
import cartopy.feature as cfeature
import numpy as np
import geopandas as gpd
import os
from cartopy.mpl.patch import geos_to_path
from data_version.rldas_configl import map_extent
from data_version.color_config import temperature_cmap
from shapely.geometry import shape


from shapely.ops import unary_union
plt.rcParams["font.sans-serif"] = ["SimHei"]  # 设置字体
plt.rcParams["axes.unicode_minus"] = False


# %%
class Plotter:
    def __init__(self, variable_name, variables, ticks, forecast_path_str, proj, data_extent, map_extent,
                 forecast_type='short-term'):
        '''
        :param fpath: 没搞懂这是啥
        :param variable_name: 变量名称
        :param city_path: 地级市矢量
        :param province_path: 省级矢量
        :param variables: 变量名，这又是啥
        :param ticks: 坐标范围
        :param manual_time: 运行时间
        :param forecast_step: 预报步长
        :param end: 结束时间?
        :param forecast_type:预报类型
        '''
        # self.fpath = fpath
        self.variable_name = variable_name
        # self.manual_time = manual_time
        # self.forecast_step = forecast_step
        # self.end = end
        # self.city_path = city_path
        # self.province_path = province_path
        self.variables = variables
        self.forecast_type = forecast_type  #
        self.ticks = ticks
        self.forecast_path_str = forecast_path_str
        self.proj = proj
        self.data_extent = data_extent
        self.map_extent = map_extent

        # 行政区矢量位置

        self.city_path = 'data_version/share/sd_city.shp'
        self.province_path = 'data_version/share/shandong_shp.shp'

        # 河流矢量 path
        self.level1rivers = './share/river/heilongjiang_level1.shp'
        self.level23rivers = './share/river/heilongjiang_level23.shp'
        self.level4rivers = './share/river/heilongjiang_level4.shp'
        self.level5rivers = './share/river/heilongjiang_level5.shp'

    # todo:  完成文件夹的创建
    def get_output_file_path(self,output_prefix, file_name,forecast_cycle=None):
        output_path = os.path.join(output_prefix, forecast_cycle,file_name)
        return output_path

    def create_subdirectories(self, base_path,forecast_cycle=None):
        # 创建子文件夹
        path_dict = dict()
        # for variable in self.variables:
        #     # 将变量名中的空格转换为下划线
        #     # 按照预报类型创建文件夹:短期，中期，短临
        #     #
        #     variable_folder = variable.replace(' ', '_')
        subdirectory = os.path.join(base_path, forecast_cycle)
        #path_dict[variable] = subdirectory
        if not os.path.exists(subdirectory):
            os.makedirs(subdirectory)
        return path_dict

    def add_city_shp(self,ax):
        city=cfeature.ShapelyFeature(Reader(self.city_path).geometries(), ccrs.PlateCarree(),
                                               edgecolor='gray',
                                               facecolor='none', linewidth=0.5)
        # city = cfeature.ShapelyFeature(
        #         Reader(self.city_path).geometries(),
        #         ccrs.PlateCarree(),
        #         edgecolor='gray',       # 设置边界颜色为灰色
        #         facecolor='none',       # 无填充色
        #         linestyle='--',         # 设置线型为虚线
        #         linewidth=0.5           # 线宽
        #     )
        ax.add_feature(city)  #添加地级市的矢量
    def add_river_shp(self, ax):
        # 这里更改为 一个函数,将河流矢量添加到图像中
        river_level1 = cfeature.ShapelyFeature(Reader(self.level1rivers).geometries(), ccrs.PlateCarree(),
                                               edgecolor='r',
                                               facecolor='none', linewidth=1.0)
        river_level23 = cfeature.ShapelyFeature(Reader(self.level23rivers).geometries(), ccrs.PlateCarree(),
                                                edgecolor='r',
                                                facecolor='none', linewidth=0.8)
        river_level4 = cfeature.ShapelyFeature(Reader(self.level4rivers).geometries(), ccrs.PlateCarree(),
                                               edgecolor='b',
                                               facecolor='none', linewidth=0.5)
        river_level5 = cfeature.ShapelyFeature(Reader(self.level5rivers).geometries(), ccrs.PlateCarree(),
                                               edgecolor='b',
                                               facecolor='none', linewidth=0.5)
        print("已将矢量添加到绘图程序")

        # 将河流水系添加到地图中
        ax.add_feature(river_level1)  # 添加一级河流
        ax.add_feature(river_level23)  # 添加二三级河流
        ax.add_feature(river_level4)  # 添加四级河流
        ax.add_feature(river_level5)  # 添加五级河流

    # @staticmethod
    def add_map_features(self, ax):
        city_name = gpd.read_file(self.city_path, encoding='utf-8')
        for x, y, label in zip(city_name.representative_point().x, city_name.representative_point().y,
                               city_name['地名']):
            ax.text(x - 0.1, y, label, fontsize=10)

        ax.add_geometries(city_name.geometry, ccrs.PlateCarree(),
                          edgecolor='k', facecolor='none', linewidth=0.5)

        ax.set_xticks(np.arange(map_extent[0], map_extent[1], 2), crs=ccrs.PlateCarree())
        ax.set_yticks(np.arange(map_extent[2], map_extent[3], 2), crs=ccrs.PlateCarree())

        lon_formatter = LongitudeFormatter(zero_direction_label=False)
        lat_formatter = LatitudeFormatter()
        ax.xaxis.set_major_formatter(lon_formatter)
        ax.yaxis.set_major_formatter(lat_formatter)

    def plot_contour_map(self, lat, lon, data, variable_label, levels, colormap, title, output_file, tips='colors',
                         draw_river=False,snow=None):
        fig = plt.figure(figsize=(8, 8), dpi=400)
        ax = fig.add_axes([0.1, 0.1, 0.8, 0.8], projection=ccrs.PlateCarree())
        ax.set_extent(map_extent)

        # axs.format(lonlim=(self.map_extent[0],self.map_extent[1]), latlim=(self.map_extent[2],self.map_extent[3]))
        # 添加地图要素:行政区和行政区名称
        self.add_map_features(ax)
        if tips == 'colors':
            a = ax.contourf(lon, lat, data, levels=levels, colors=colormap)  # ,extend='max')
            if snow is not None:
                snow_levels=[0,2.5,5,10,20,30,50]
                snow_ticks=[2.5,5,10,20,30]
                snow_colors=("#CCCCCC","#A1A1A1","#707070","#464646","#7346E1","#500078")
                snow[snow<0.1]=np.nan
                a1=ax.contourf(lon,lat,snow,snow_levels,colors=snow_colors,extend='max')
                #边缘裁剪，白化
                #records = Reader (self.province_path).records ()
                records = Reader (self.province_path).records ()
                for record in records:
                    path = Path.make_compound_path (*geos_to_path ([record.geometry]))
                for collection in a1.collections:
                    collection.set_clip_path (path, transform=ax.transData)    
        elif tips == 'cmaps':
            a = ax.contourf(lon, lat, data, levels=levels, cmap=colormap)  # ,extend='max')

        province = cfeature.ShapelyFeature(Reader(self.province_path).geometries(), ccrs.PlateCarree(), edgecolor='k',
                                           facecolor='none')
        ax.add_feature(province)

        # 添加河流
        if draw_river == True:
            self.add_river_shp(ax)
        else:
            pass

        # 边缘裁剪，白化
        records = Reader(self.province_path).records()
        for record in records:
            path = Path.make_compound_path(*geos_to_path([record.geometry]))

        for collection in a.collections:
            collection.set_clip_path(path, transform=ax.transData)

        plt.title(title, fontsize=20, y=1.0)

        cax = fig.add_axes([ax.get_position().x1 + 0.01, ax.get_position().y0, 0.02, ax.get_position().height])
        cb = plt.colorbar(a, cax, label=variable_label)

        if self.ticks != None:
            cb.set_ticks(self.ticks)
        else:
            pass

        cb.update_ticks()
        
        if snow is not None:
            cax1 = fig.add_axes ([ax.get_position().x0 + 0.02, ax.get_position ().y0+0.01, 0.04, ax.get_position ().height*0.3])
            cb2=plt.colorbar(a1,cax1,label="降雪量/$mm$",extendrect=True)
            cb2.set_ticks(snow_ticks)
            cb2.update_ticks()
        plt.savefig(output_file, bbox_inches='tight', pad_inches=0.2, transparent=False, format='png')

        # plt.close()
        # plt.show()
        plt.close()


    def plot_map_simple(self,grid_lon,grid_lat, data,colormap,norm,outfile,draw_city=False):
        fig = plt.figure(111, dpi=600)
        
        #地图展示:墨卡托投影
        map_lonmin=grid_lon.min()
        map_lonmax=grid_lon.max()
        
        map_latmin=grid_lat.min()
        map_latmax=grid_lat.max()
        
        #出图的范围
        show_map_extent=[map_lonmin,map_lonmax,map_latmin,map_latmax]
        
        
        ax = plt.axes(projection=ccrs.Mercator())
        plt.axis('off') #关闭坐标轴
        ax.set_extent(map_extent)
        im=ax.imshow(data,cmap=colormap,extent=show_map_extent,norm=norm,transform=ccrs.PlateCarree())
        print(map_extent)
        
    
        # 边缘裁剪，白化
        records = Reader(self.province_path).records()
        geometries = [shape(record.geometry) for record in records]
        clip_path = Path.make_compound_path(*geos_to_path(geometries))

        # 创建 PathPatch 进行裁剪
        clip_patch = PathPatch(clip_path, transform=ccrs.PlateCarree(),edgecolor='none', facecolor='none', zorder=1)
        
        # 将 PathPatch 添加到图形中
        ax.add_patch(clip_patch)
        # 设置裁剪
        im.set_clip_path(clip_patch)
        # province = cfeature.ShapelyFeature(Reader(self.province_path).geometries(), ccrs.PlateCarree(), edgecolor='k',
        #                                    facecolor='none')
        # ax.add_feature(province)
        
        if draw_city:
            self.add_city_shp(ax)
        else:
            pass 

        plt.savefig(outfile,format='png', bbox_inches='tight', dpi=600, pad_inches=0.0, transparent=True)
        print(show_map_extent)
        print(outfile)
        plt.close() #关闭
    def plot_map_simple_snow(self,grid_lon,grid_lat, data,colormap,norm,outfile,snow=None,Sleet=None):
        
        snow_levels=[0,1.0,2.0,4.0,8.0,12.0,30.0]  #降雪的色带 3小时
        snow_ticks=[2.5,5,10,20,30]
        snow_colors=("#CCCCCC","#A1A1A1","#707070","#464646","#7346E1","#500078")
        color = ("#FFFFFF", "#A6F28F", "#3DBA3D", "#61BBFF", "#0000FF", "#FA00FA", "#800040")
        level = [0, 0.1, 1, 3, 10, 20, 50, 70]  # 短临降水拉伸
        fig = plt.figure(111, dpi=600)
        
        #地图展示:墨卡托投影
        map_lonmin=grid_lon.min()
        map_lonmax=grid_lon.max()
        
        map_latmin=grid_lat.min()
        map_latmax=grid_lat.max()
        
        
        #出图的范围
        show_map_extent=[map_lonmin,map_lonmax,map_latmin,map_latmax]
        ax = plt.axes(projection=ccrs.Mercator())
        plt.axis('off') #关闭坐标轴
        #ax.set_extent(map_extent) 
        a = ax.contourf(grid_lon,grid_lat, data, levels=level, colors=color,transform=ccrs.PlateCarree())  # ,extend='max')
        #ax.set_extent(map_extent)
        # 边缘裁剪，白化
        provinces = list(Reader(self.province_path).geometries())
        
        # 将所有省份合并为一个多边形
        merged_province = unary_union(provinces)
        
        # 将多边形转换为墨卡托投影下的坐标
        mercator_merged_province = ccrs.Mercator().project_geometry(merged_province, ccrs.PlateCarree())
        
        # 创建裁剪路径
        path = Path.make_compound_path(*geos_to_path([shape(mercator_merged_province)]))

        for collection in a.collections:
            collection.set_clip_path(path, transform=ax.transData)
        if snow is not None:
            snow_levels=[0,1.0,2.0,4.0,8.0,12.0,30.0]
            snow_ticks=[2.5,5,10,20,30]
            snow_colors=("#CCCCCC","#A1A1A1","#707070","#464646","#7346E1","#500078")
            snow[snow<0.1]=np.nan
            a1=ax.contourf(grid_lon,grid_lat,snow,snow_levels,colors=snow_colors,extend='max',transform=ccrs.PlateCarree())#,transform=ccrs.PlateCarree())
            ax.set_extent(map_extent)
            #边缘裁剪，白化
            #records = Reader (self.province_path).records ()
            # records = Reader (self.province_path).records ()
            # for record in records:
            #     path = Path.make_compound_path (*geos_to_path ([record.geometry]))
            for collection in a1.collections:
                collection.set_clip_path (path, transform=ax.transData)  
                
        ax.set_extent(map_extent) 
        plt.savefig(outfile,format='png', bbox_inches='tight', dpi=600, pad_inches=0.0, transparent=True)
        print(show_map_extent)
        print(outfile)
        plt.close() #关闭    
    def plot_map_simple_win(self,grid_lon,grid_lat, data,outfile):
        '''
        大风预警信息,
        '''
        
        #color = ("#2DA0D7", "#EEE33F", "#DF8E49", "#C13634") #对应8级大风，9级大风，10级大风，11级大风以上
        
        color=("#A5F08E","#3DC0FF","#003DFF","#EEE33F", "#DF8E49", "#C13634")  #对应5级一下，6级，7级，8级，9级，10级以上
        
        '''
        todo: 更改大风预警等级
            wins_level_0 = (data < 0.3) | (np.isnan(data))  # 无风
            wins_level_1 = (data >= 0.3) & (data < 1.6)  # 1级风
            wins_level_2 = (data >= 1.6) & (data < 3.4)  # 2级风
            wins_level_3 = (data >= 3.4) & (data < 5.5)  # 3级风
            wins_level_4 = (data >= 5.5) & (data < 8.0)  # 4级风
            wins_level_5 = (data >= 8.0) & (data < 10.8)  # 5级风
            wins_level_6 = (data >= 10.8) & (data < 13.9)  # 6级风
            wins_level_7 = (data >= 13.9) & (data < 17.2)  # 7级风
            wins_level_8 = (data >= 17.2) & (data < 20.8)  # 8级风
            wins_level_9 = (data >= 20.8)  # 9级风
        '''
        
        
        level = [0,10.8,13.9,17.2,20.8,24.5,40]  #对应的拉伸 5级、6级、7级、8级、9级、10级
        fig = plt.figure(111, dpi=600)
        
        #地图展示:墨卡托投影
        map_lonmin=grid_lon.min()
        map_lonmax=grid_lon.max()
        
        map_latmin=grid_lat.min()
        map_latmax=grid_lat.max()
        
        #出图的范围
        show_map_extent=[map_lonmin,map_lonmax,map_latmin,map_latmax]
        
        ax = plt.axes(projection=ccrs.Mercator())
        
        plt.axis('off') #关闭坐标轴
        #ax.set_extent(show_map_extent)
        a = ax.contourf(grid_lon,grid_lat, data, levels=level, colors=color,transform=ccrs.PlateCarree()) # ,extend='max')
        #ax.set_extent(show_map_extent)
        # 边缘裁剪，白化
        # records = Reader(self.province_path).records()
        # for record in records:
        #     path = Path.make_compound_path(*geos_to_path([record.geometry]))
        # for collection in a.collections:
        #     collection.set_clip_path(path, transform=ax.transData)
        provinces = list(Reader(self.province_path).geometries())
        
        # 将所有省份合并为一个多边形
        merged_province = unary_union(provinces)
        
        # 将多边形转换为墨卡托投影下的坐标
        mercator_merged_province = ccrs.Mercator().project_geometry(merged_province, ccrs.PlateCarree())
        
        # 创建裁剪路径
        path = Path.make_compound_path(*geos_to_path([shape(mercator_merged_province)]))

        # 应用裁剪
        for collection in a.collections:
            collection.set_clip_path(path, transform=ax.transData)
        
        
        ax.set_extent(map_extent) #设置显示的范围
        plt.savefig(outfile,format='png', bbox_inches='tight', dpi=600, pad_inches=0.0, transparent=True)
        print(show_map_extent)
        print(outfile)
        plt.close() #关闭    
    def plot_map_simple_storm(self,grid_lon,grid_lat, data,outfile):
        '''
        暴雨预警信息,
        '''
        
        #color = ("#2DA0D7", "#EEE33F", "#DF8E49", "#C13634") #对应8级大风，9级大风，10级大风，11级大风以上
        #color=("#A5F08E","#3DC0FF","#003DFF","#EEE33F", "#DF8E49", "#C13634")  #对应5级一下，6级，7级，8级，9级，10级以上
        color = ("#0000FF","#FA00FA", "#800040") #暴雨预警级别
        
        
        
        '''
        todo: 更改大风预警等级
            wins_level_0 = (data < 0.3) | (np.isnan(data))  # 无风
            wins_level_1 = (data >= 0.3) & (data < 1.6)  # 1级风
            wins_level_2 = (data >= 1.6) & (data < 3.4)  # 2级风
            wins_level_3 = (data >= 3.4) & (data < 5.5)  # 3级风
            wins_level_4 = (data >= 5.5) & (data < 8.0)  # 4级风
            wins_level_5 = (data >= 8.0) & (data < 10.8)  # 5级风
            wins_level_6 = (data >= 10.8) & (data < 13.9)  # 6级风
            wins_level_7 = (data >= 13.9) & (data < 17.2)  # 7级风
            wins_level_8 = (data >= 17.2) & (data < 20.8)  # 8级风
            wins_level_9 = (data >= 20.8)  # 9级风
        '''
        
        
        level = [10,20,50,250]  #对应的拉伸 5级、6级、7级、8级、9级、10级
        fig = plt.figure(111, dpi=600)
        
        #地图展示:墨卡托投影
        map_lonmin=grid_lon.min()
        map_lonmax=grid_lon.max()
        
        map_latmin=grid_lat.min()
        map_latmax=grid_lat.max()
        
        #出图的范围
        show_map_extent=[map_lonmin,map_lonmax,map_latmin,map_latmax]
        
        ax = plt.axes(projection=ccrs.Mercator())
        
        plt.axis('off') #关闭坐标轴
        #ax.set_extent(show_map_extent)
        a = ax.contourf(grid_lon,grid_lat, data, levels=level, colors=color,transform=ccrs.PlateCarree()) # ,extend='max')
        #ax.set_extent(show_map_extent)
        # 边缘裁剪，白化
        # records = Reader(self.province_path).records()
        # for record in records:
        #     path = Path.make_compound_path(*geos_to_path([record.geometry]))
        # for collection in a.collections:
        #     collection.set_clip_path(path, transform=ax.transData)
        provinces = list(Reader(self.province_path).geometries())
        
        # 将所有省份合并为一个多边形
        merged_province = unary_union(provinces)
        
        # 将多边形转换为墨卡托投影下的坐标
        mercator_merged_province = ccrs.Mercator().project_geometry(merged_province, ccrs.PlateCarree())
        
        # 创建裁剪路径
        path = Path.make_compound_path(*geos_to_path([shape(mercator_merged_province)]))

        # 应用裁剪
        for collection in a.collections:
            collection.set_clip_path(path, transform=ax.transData)
        
        
        ax.set_extent(map_extent) #设置显示的范围
        plt.savefig(outfile,format='png', bbox_inches='tight', dpi=600, pad_inches=0.0, transparent=True)
        print(map_extent)
        #print(show_map_extent)
        print(outfile)
        plt.close() #关闭    

