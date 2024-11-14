'''
Author: Liang 7434493+skybluechina@user.noreply.gitee.com
Date: 2023-11-16 10:14:36
LastEditors: Liang 7434493+skybluechina@user.noreply.gitee.com
LastEditTime: 2024-02-29 16:09:52
FilePath: /RLDAS_statis/un_config/color_config.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import colors
import cmaps
# 定义颜色映射
# temperature_cmap_data = {
#     'red':   [(0.0, 0.5, 0.5), (0.25, 0.239, 0.95), (0.5, 0.945, 1.0), (0.75, 0.988, 0.996), (1.0, 0.925, 0.925)],
#     'green': [(0.0, 0.0, 0.0), (0.25, 0.624, 1.0), (0.5, 1.0, 1.0), (0.75, 0.498, 0.522), (1.0, 0.357, 0.357)],
#     'blue':  [(0.0, 0.482, 0.482), (0.25, 0.929, 1.0), (0.5, 1.0, 0.608), (0.75, 0.608, 0.463), (1.0, 0.376, 0.376)]
# }

# 气温
temperature_cmap= cmaps.cmp_b2r #气温的色带 
temp_norm=colors.Normalize(vmin=-20,vmax=40) #气温的拉伸范围

solar_cmap= cmaps.cmocean_matter #辐照度的色带
soalr_norm=colors.Normalize(vmin=0,vmax=1000)



#降水
prec_cmap=cmaps.precip4_11lev #降水的色带
prec_norm=colors.Normalize(vmin=0,vmax=50) #降水的拉伸范围

#相对湿度
rh_cmap=cmaps.cmocean_algae #相对湿度的色带
rh_norm=colors.Normalize(vmin=0,vmax=100) #相对湿度的拉伸范围

#风速
wins_cmap=cmaps.wind_17lev #风速的色带
wins_norm=colors.Normalize(vmin=0,vmax=18) #风速的拉伸范围


#覆冰厚度
ice_cmap=cmaps.seaice_2 #覆冰厚度的色带
ice_norm=colors.Normalize(vmin=0,vmax=200) #覆冰厚度的拉伸范围

#雷暴
cape_cmaps=cmaps.WhBlGrYeRe #雷暴的色带
cape_norm=colors.Normalize(vmin=0,vmax=5000) #雷暴的拉伸范围



