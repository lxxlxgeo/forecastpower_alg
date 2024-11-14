import cdsapi
import calendar
import os
import numpy as np
from datetime import datetime
c = cdsapi.Client()
now = datetime.now()
current_year = now.year
current_month = now.month
if current_month == 1:
    year = current_year - 1
    month = 12
else:
    year = current_year
    month = current_month - 1
data_path = '/public4/dataset_yl/EC_ERA5'

days_in_month = calendar.monthrange(year, month)[1]
days = [d for d in range(1, days_in_month + 1)]

for day in days:
    file_name = os.path.join(
        data_path, f"ERA5.solar_radiation_all.{year}{str(month).zfill(2)+str(day).zfill(2)}.nc")
    if os.path.exists(file_name):
        print(f"文件 {file_name} 已存在。跳过。")
        continue
    try:
        c.retrieve(
            'reanalysis-era5-single-levels',
            {
                'product_type': 'reanalysis',
                'format': 'netcdf',
                'variable': [
                    'Surface solar radiation downwards', 'mean_surface_downward_short_wave_radiation_flux', 'mean_surface_downward_long_wave_radiation_flux','Forecast albedo','Total cloud cover', 'Low cloud cover','High cloud cover','Medium cloud cover','total_column_cloud_ice_water','total_column_cloud_liquid_water','total_column_water_vapour','Total column ozone',
                ],
                "year": str(year),
                "month": str(month),
                "day": str(day),
                'time': [
                    '00:00', '01:00', '02:00',
                    '03:00', '04:00', '05:00',
                    '06:00', '07:00', '08:00',
                    '09:00', '10:00', '11:00',
                    '12:00', '13:00', '14:00',
                    '15:00', '16:00', '17:00',
                    '18:00', '19:00', '20:00',
                    '21:00', '22:00', '23:00',
                ],
                'area': [
                    53, 73, 18,
                    136,
                ],
                'grid' : [0.25, 0.25],
            },
            file_name,)
        print(f"已下载：{file_name}")
    except Exception as e:
        print(f"下载 {file_name} 时出错：{e}")