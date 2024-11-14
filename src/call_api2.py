import requests
import os

# url='http://47.104.225.193:10035/cloud/powerForecastFileTable/insertFilesData'

def post_request(json_string):
    # url='http://47.104.225.193:10035/cloud/powerForecastFileTable/insertFilesData'
    url='xxx' 
    parms=json_string
    respose=requests.get(url,params=parms)
    context=respose.json()
    print(context)

def call_api(electricFieldName=None, fbTime=None, fbInterval=None, forecastType=None, fbType=None, filePath=None, fileName=None, dispatchTime=None):
    """调用接口入库

    Args:
        electricFieldName (_type_): 场站名
        fbTime (_type_): 发报时间
        fbDuration (_type_): 发报时长
        fbInterval (_type_): 发报间隔
        forecastType (_type_): 
        fbType: 0超短期, 1短期, 2鲁山未来24h, 3鲁山未来72小时, 4粤电气象
        filePath (_type_): 文件路径
        fileName (_type_): 文件名
    """

    params={
        "electricFieldId": "",
        "electricFieldName": electricFieldName,  #"aaaa",
        "stationId": '',
        "fbTime": fbTime,   #"2024-05-03 20:00:00"
        "fbDuration": '',
        "fbInterval": fbInterval,    #''
        "forecastType": forecastType,  #'',
        "fbType": fbType,
        "filePath": filePath,
        "fileName": fileName,
        "isProcessed": "",
        "failureCount": "",
        "dispatchTime":dispatchTime
    }

    post_request(params)

if __name__ == '__main__':
    # call_api(electricFieldName='malou', 
    #             fbTime="2024-05-03 20:00:00", 
    #             fbInterval='', 
    #             forecastType='0', 
    #             fbType='1',
    #             filePath='/share/data/power_foreacst/henan/henan_power/Power_push/Liangwa/CDQ/20240505/CDQ_solar_Liangwa_20240505153000.csv', 
    #             fileName='CDQ_solar_Liangwa_20240505153000.csv')

    file_CDQ = "/share/data/power_foreacst/guangdong/zhanjiang/wailuo/2024071716/CDQ_power_zhanjiang_2024071716.csv"
    file_DQ = "/share/data/power_foreacst/guangdong/zhanjiang/wailuo/2024071716/DQ_power_zhanjiang_2024071716.csv"
    file_wheather = "/share/data/power_foreacst/guangdong/zhanjiang/wailuo/2024071716/wind_zhanjiang_2024071716.csv"

    filelist = [file_DQ, file_CDQ, file_wheather]
    fbType_dict = {
        "CDQ": 0,
        "DQ": 1,
        "wind": 4,
    }

    file = file_CDQ
    for file in filelist:

        fbTime = "2024-07-17 16:00:00"
        forecastType = 1

        strtype = os.path.basename(file).split("_")[0]
        fbType = fbType_dict[strtype]

        call_api(electricFieldName="zhanjiang_wailuo",
                    fbTime = fbTime,
                    fbInterval ='', 
                    forecastType = 1, 
                    fbType =fbType,
                    filePath = file[27:], 
                    fileName = os.path.basename(file),
                    dispatchTime = fbTime)


