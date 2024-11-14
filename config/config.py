

import datetime 
time_now=datetime.datetime.utcnow()+datetime.timedelta(hours=8)
#time_now=datetime.datetime(2024,11,14,6)

time_now_str=time_now.strftime('%Y%m%d%H')

##
# ECMWF D1D BZ path
ec_d1d_bz_path='/mnt/xxdata/wlzx_model/raw/babj/ecmwf/hres_9km/achn'

# 解压后的GRIB文件
ec_d1d_grib_path='/public4/dataset_yl/forecastpower/nwp_grib' 

#预报文件生成
outprefix = '/public4/dataset_yl/forecastpower/shandongforecast/shandong_power'

model_file = 'share/modefile/solar/model.pth'
param_file = 'share/modefile/solar/train_step7_config.pkl'

model_file_wind = 'share/modefile/wind/model_wind.pth'
param_file_wind = 'share/modefile/wind/train_step7_config.pkl'

# 山东区域精细化预报模式，适用短临24小时、短期3天和超短期0-4小时
shandong_heigh_prefix_wind='/data/Datasets/power_forecast/re_process/WRF/shandong/'

# 全球模式数据，适用于 3-10天
ec_path = '/public4/dataset_yl/forecastpower/nwp_push'

extent_solar = [117, 119, 32, 34]
extent_wind = [117.6, 119.6, 33, 35]

runlogPath = 'runlog'


###气象站点csv网格数据,station文件
csv_grid_path='src/data/grid_stations.csv'
