import io
import numpy as np

#fro un_config.rldas_configl import map_extent
from scipy.interpolate import griddata
import numpy as np
from osgeo import gdal,osr



class Resample(object):

    #def __init__(self):
    def __init__(self,lons:np.ndarray,lats:np.ndarray,data:np.ndarray,res=0.1):
        '''
        :param lons: RLDAS 经度
        :param lats: RLDAS 纬度
        :param data: RLDAS 要素,数据
        :param res: 分辨率,默认0.04
        :return:
        '''

        self.lons=lons
        self.lats=lats
        self.data=data
        self.res=res   #分辨率
        self.get_lonlat_extent()

        self.shape=lons.shape


    def get_lonlat_extent(self):

        lon_min=self.lons.min()
        lon_max=self.lons.max()

        lat_min=self.lats.min()
        lat_max=self.lats.max()
        extent=[lon_min,lon_max,lat_min,lat_max]
        self.extext=extent

    def generate_grid_lonlat(self):
        proj_extent=self.extext

        lon_line=np.arange(proj_extent[0],proj_extent[1],self.res)
        lat_line=np.arange(proj_extent[2],proj_extent[3],self.res)

        grid_lon,grid_lat=np.meshgrid(lon_line,lat_line)

        del lon_line,lat_line
        return grid_lon,grid_lat
    def resample_data(self):

        # 将 lambet 投影转换为等经纬投影
        grid_lon,grid_lat=self.generate_grid_lonlat()

        source_lon=self.lons.reshape(self.shape[0]*self.shape[1])
        source_lat=self.lats.reshape(self.shape[0]*self.shape[1])
        source_data=self.data.reshape(self.shape[0]*self.shape[1])
        source_data[np.isnan(source_data)]=0


        grid_data=griddata((source_lon,source_lat),source_data,(grid_lon,grid_lat),method='linear')

        return grid_data,grid_lon,grid_lat
    @staticmethod
    def write_tiff(outfile, data, extents, res):
        driver = gdal.GetDriverByName('GTiff')
        # out_tif_name=os.path.join(out_dir,str(year)+str(month).zfill(2)+'_XCO2_average.tif')
        target_tif_name = outfile
        im_height, im_width = data.shape
        out_tif = driver.Create(target_tif_name, im_width,
                                im_height, 1, gdal.GDT_Float32)
        # 设置影像显示区域
        LonMin, LatMax = extents[0], extents[3]
        geotransform = (LonMin, res, 0, LatMax, 0, -res)
        out_tif.SetGeoTransform(geotransform)
        # 设置地理信息，选取所需的坐标系统
        srs = osr.SpatialReference()
        srs.ImportFromEPSG(4326)  # 定义输出的坐标为WGS84，Authority['EPSG','4326']
        out_tif.SetProjection(srs.ExportToWkt())  # 新建图层投影
        # 数据写出
        out_tif.GetRasterBand(1).WriteArray(data)  # 数据写入内存，此时未写入硬盘
        out_tif.FlushCache()  # 数据写入硬盘
        out_tif = None  # 关闭tif文件bu

def write_text(data, txtfile):
    bio = io.BytesIO()
    data = np.round(data, decimals=4)
    datastr = data.astype(np.str_)
    np.savetxt(bio, datastr, fmt="%s", delimiter=' ')
    mystr = bio.getvalue().decode('utf-8')
    mystr = mystr.replace('nan', 'n')
    gzipData(mystr, txtfile)
    

def gzipData(content, fileName):
    '''
    :param content:
    :param fileName:
    :return:
    '''
    import gzip  # compress the content and write to a file
    with gzip.open(fileName, 'wb') as f:
        f.write(content.encode('utf-8'))

def convert_source_totext(lon:np.ndarray,lat:np.ndarray,data:np.ndarray,outfile:str):
    #resample = Resample(lon, lat, data)
    #resample.get_lonlat_extent()
    #grid_data,grid_lon,grid_lat=resample.resample_data()
    # print(grid_lat.max())
    # print(grid_lat.min())
    # print(grid_lon.max())
    # print(grid_lon.min())
    write_text(data,outfile)
    
    #return np.flipud(grid_data),grid_lon,grid_lat