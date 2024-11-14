from src.bz2 import decompress_files
from config.config import ec_d1d_bz_path,ec_d1d_grib_path,time_now,time_now_str
import datetime 
import os 

if __name__=='__main__':
    

    now_time=time_now
    year=now_time.year
    month=now_time.month
    day=now_time.day
    hour=now_time.hour
    
    if hour>=16:
        finally_hour=0
        forecast_time=datetime.datetime(year,month,day,finally_hour)
        bz_time='00'
    elif (hour>=0)&(hour<5):
        finally_hour=0
        forecast_time=datetime.datetime(year,month,day,finally_hour)+datetime.timedelta(days=-1)
        bz_time='00'
    elif (hour>=5)&(hour<16):
        finally_hour=12
        forecast_time=datetime.datetime(year,month,day,finally_hour)+datetime.timedelta(days=-1)
        bz_time='12'
        
    forecast_time=forecast_time
    forecast_time_str=forecast_time.strftime('%Y%m%d%H') # UTC 时间
    forecast_time_str_bjt=(forecast_time+datetime.timedelta(hours=8)).strftime('%Y%m%d%H') #北京时间
    
    file_path=os.path.join(ec_d1d_grib_path,forecast_time_str)
    use_bz_path=os.path.join(os.path.join(ec_d1d_bz_path,forecast_time.strftime('%Y%m%d')),bz_time)
    print(file_path)
    print(use_bz_path)
    decompress_files(use_bz_path,file_path)
