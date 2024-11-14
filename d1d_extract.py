import xarray as xr
from glob import glob
import datetime
import os,re
from joblib import Parallel, delayed
from config.config import time_now
import warnings
import sys
from config.config import ec_path
warnings.filterwarnings('ignore')


class EC_extract(object):
    def __init__(self,datetime_n) -> None:
        self.datetime_n=datetime_n
        self.input_prefix = '/public4/dataset_yl/forecastpower/nwp_grib/' # EC的根目录
        # F:\D1D-2022 2022年6月份开始
        self.outprefix = ec_path  #输出的
        #self.extent = [117, 119.6, 32, 35]   # 风+光伏
        self.extent= [113, 124, 30, 40]
        self.__get_run_info()
    def __get_run_info(self):
        #
        now_time=self.datetime_n
        year=now_time.year
        month=now_time.month
        day=now_time.day
        hour=now_time.hour
        if hour>=16:
            finally_hour=0
            forecast_time=datetime.datetime(year,month,day,finally_hour)
        elif (hour>=0)&(hour<5):
            finally_hour=0
            forecast_time=datetime.datetime(year,month,day,finally_hour)+datetime.timedelta(days=-1)
        elif (hour>=5)&(hour<16):
            finally_hour=12
            forecast_time=datetime.datetime(year,month,day,finally_hour)+datetime.timedelta(days=-1)
        self.forecast_time=forecast_time
        self.forecast_time_str=forecast_time.strftime('%Y%m%d%H')
        self.forecast_time_str_bjt=(forecast_time+datetime.timedelta(hours=8)).strftime('%Y%m%d%H')
        self.file_path=os.path.join(self.input_prefix,self.forecast_time_str)
        print(self.file_path)
        sfc_files=glob(self.file_path+'/*.grib2')
        #pl_files=glob(self.file_path+'/pl*.grib1')
        if (len(sfc_files)==0):
            print('the current time is no file')
        else:
            
            # sfc_files=sorted(sfc_files, key=lambda x: int(re.search(r'D1D(\d+)\-ACHN.grib2$', x).group(1)))
            # sfc_files=sfc_files
            self.sfc_files=sfc_files

    def get_datasets(self):
        
        # 判断如果不存在nc文件，再进行提取
        outfile_time_str=self.forecast_time_str
        outfile=os.path.join(self.outprefix,'shandong_'+outfile_time_str+'.nc')
        print(outfile)
        if not os.path.exists(outfile):

            # 合并后的地面层,气压层文件
            target_ds_list = []
            
            # ['ssrd','strd','ssr','str','tsr','ttr','tsrc','ttrc','ssrc','strc','fdir','cdir','lcc','mcc','hcc','tcc','fal','10u','10v','2t','sp','vis']
            # sfc_varname = ['10u', '10v', '100u', '100v',
            #                '2t', 'sp','tcw','tcwv','sst',
            #                'ssrd','strd','ssr','str','tsr',
            #                'ttr','tsrc','ttrc','ssrc','strc',
            #                'fdir','cdir','lcc','mcc','hcc','tcc','fal','vis']   # 地面层数据
            pl_varname = ['u', 'v', 'q'] #气压层的数据
            sfc_varname = ['ssrd','strd','ssr','str','tsr','ttr','tsrc','ttrc','ssrc','strc','fdir','cdir','lcc','mcc','hcc','tcc','2t','2d','sp','10u','10v', '100u', '100v']
            
            def read_data(isfc):
                
                try:
                    sfc_ds = xr.merge([xr.open_dataset(isfc, engine='cfgrib', filter_by_keys={'dataType': 'fc', 'typeOfLevel': 'surface', 'shortName': f'{i}'}, backend_kwargs={'indexpath': None}) for i in sfc_varname])
                    pl_ds=xr.merge([xr.open_dataset(isfc,engine='cfgrib', filter_by_keys={'typeOfLevel': 'isobaricInhPa', 'shortName':f'{i}'},backend_kwargs={'indexpath': None}) for i in pl_varname])
                    # 缩减范围
                    sfc_ds_sub = sfc_ds.sel(longitude=slice(self.extent[0], self.extent[1]), latitude=slice(self.extent[3], self.extent[2]))
                    pl_ds_sub = pl_ds.sel(longitude=slice(self.extent[0]-0.3, self.extent[1]+0.3), latitude=slice(self.extent[3]+0.3, self.extent[2]-0.3))
                    
                    # 气压层数据插值到地面层大小
                    i_lon = sfc_ds_sub['longitude'].values
                    i_lat = sfc_ds_sub['latitude'].values
                    pl_ds_sub = pl_ds_sub.interp(longitude=i_lon, latitude=i_lat, method='nearest')
                    ds_merge = xr.merge([pl_ds_sub, sfc_ds_sub])
                    # ds_merge = sfc_ds_sub
                    sfc_ds.close()
                    print(isfc)
                    return ds_merge
                except:
                    return None 
            target_ds_list = Parallel(n_jobs=4)(delayed(read_data)(x) for x in self.sfc_files)
            target_ds_list = [x for x in target_ds_list if x is not None]
            #target_ds_list=[read_data(x) for x in self.sfc_files]
            #target_ds_list=[x for x in target_ds_list if x is not None ]
            targer_ds=xr.concat(target_ds_list,dim='valid_time')
            targer_ds=targer_ds.sortby('valid_time')
            target_ds_res=targer_ds.resample({'valid_time':'15min'}).interpolate("cubic") #最终要的数据为3次线性的15分钟间隔数据
            # outfile_time_str=self.forecast_time_str
            # outfile=os.path.join(self.outprefix,'Henan_'+outfile_time_str+'.nc')
            #return target_ds_res
            #return targer_ds
            # target_ds_res.to_netcdf(outfile)
            if len(target_ds_res['valid_time']) >= 900:  # 如果nas中的ec全部到达，这里valid_time的长度是961
                target_ds_res.to_netcdf(outfile)
            else:
                sys.exit(1)


def process_date(itime):
    try:
        ec_run = EC_extract(itime)
        ec_run.get_datasets()
        print(itime)
        print('ok')
    except Exception as e:
        print(e)
        sys.exit(1)


if __name__ == '__main__':
    #time1=datetime.datetime.utcnow()
    time1=datetime.datetime(2024,9,10,6,30)
    # exec_time=datetime.datetime(2024,5,23,6,30)   #启动时间,手动设置
    exec_time=time_now
    
    process_date(exec_time)
    time2=datetime.datetime.utcnow()
    print(time2-time1)


