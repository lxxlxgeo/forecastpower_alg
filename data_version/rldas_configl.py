'''
Author: Liang 7434493+skybluechina@user.noreply.gitee.com
Date: 2023-11-06 18:05:22
LastEditors: Liang 7434493+skybluechina@user.noreply.gitee.com
LastEditTime: 2024-05-27 03:55:14
FilePath: /RLDAS_statis/un_config/rldas_configl.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# -*- coding: utf-8 -*-
"""
Created on 2023/8/12 15:34

@author: lxce
"""
import numpy as np

rldas_path = '/share/data/WRF-RLDAS-Heilongjiang'

map_extent = [114.5, 122.8, 34.0, 38.5]
#visual_total = [114, 123, 34, 39.0]  # 区域设置
output_prefix = '/share/data/pic_heilongjiang/RLDAS_WRF'

txt_out_prefix='/share/data/pic_heilongjiang/hlj_service_product/imminent'

#统计要素矢量
statis_level_1='share/statis_feature/level/level1/level_1.shp'
statis_level_3='share/statis_feature/level/level3/level3.shp'
statis_level_5='share/statis_feature/level/level5/level5.shp'
statis_level_2='share/statis_feature/level/level2/level2.shp'
statis_level_4='share/statis_feature/level/level4/level4.shp'

#塔杆号风险矢量
line_shp='share/line_shp/mudanjiang.shp'

city_shp='/home/groupwang/software/lxdir/hlj_forecast_product/RLDAS_statis/share/statis_feature/level/level3/level3.shp'
region_shp='/home/groupwang/software/lxdir/hlj_forecast_product/RLDAS_statis/share/statis_feature/level/level5_city/level5_cityname.shp'
hlj_province_all='/home/groupwang/software/lxdir/hlj_forecast_product/RLDAS_statis/share/statis_feature/level/level1/level_1.shp'
hlj_province_region='/home/groupwang/software/lxdir/hlj_forecast_product/RLDAS_statis/share/statis_feature/level/level3_city/level3.shp'



line_shp_hljst='/home/groupwang/software/lxdir/hlj_forecast_product/RLDAS_statis/share/line_statis_shp/融合后线路.shp'


#黑龙江所有线路统计

line_shp_hlj_all='share/hlj_all_line/黑龙江全省.shp'  #黑龙江全省的输电线路塔杆连接线


def generate_plot_config(step: int) -> dict:
    pre_mid_level = [0, 0.1, 10, 25, 50, 100, 250, 500]  # 短期或中期降水量拉伸
    pre_mid_ticks = [0.1, 10, 25, 50, 100, 250]  # 短期或中期降水刻度
    cape_mid_level = np.arange(200, 4001, 200)  # 短期或中期位能拉伸
    wins_mid_level = np.arange(0, 35, 2)  # 短期或中期风速拉伸

    pre_imminent_level = [0, 0.1, 1, 3, 10, 20, 50, 70]  # 短临降水拉伸
    pre_imminent_ticks = [0.1, 1, 3, 10, 20, 50]  # 短临降水颜色条刻度
    cape_imminent_level = np.arange(200, 4001, 200)  # 短临位能拉伸
    wins_imminent_level = np.arange(0, 35, 2)  # 短临风速拉伸

    mid_short_color_conf = dict()
    mid_short_color_conf['pre_level'] = pre_mid_level
    mid_short_color_conf['pre_ticks'] = pre_mid_ticks
    mid_short_color_conf['cape_level'] = cape_mid_level
    mid_short_color_conf['wins_level'] = wins_mid_level

    imminent_color_conf = dict()
    imminent_color_conf['pre_level'] = pre_imminent_level
    imminent_color_conf['pre_ticks'] = pre_imminent_ticks
    imminent_color_conf['cape_level'] = cape_imminent_level
    imminent_color_conf['wins_level'] = wins_imminent_level

    if step == 3:
        return imminent_color_conf
    elif step == 24:
        return mid_short_color_conf
