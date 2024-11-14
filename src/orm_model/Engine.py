'''
Author: Liang 7434493+skybluechina@user.noreply.gitee.com
Date: 2023-11-06 18:05:22
LastEditors: Liang 7434493+skybluechina@user.noreply.gitee.com
LastEditTime: 2023-12-12 17:38:25
FilePath: /RLDAS_statis/model/Engine.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
import urllib.parse
import pymysql
'''导入模型'''
#
# from Data_Reader.control_framework_orm.models import Weather_station_his,Weather_station_his_check
#%%
'''
连接字符串

'''


sql_info = {'host': '10.76.36.56',#连接地址
            'port': '3307',#端口号
            'schema': 'kys_power',#数据库名
            'user': 'root',#用户名
            'pw': urllib.parse.quote_plus("geovis@123,.")  #数据库的密码
            }
'''
创建数据库引擎
'''
class Create_Engine(object):
    def __init__(self):
        self.conn=self.conn_to_sql()
    def conn_to_sql(self):
        pymysql.install_as_MySQLdb()
        cmd = f"mysql://{sql_info['user']}:{sql_info['pw']}@{sql_info['host']}:{sql_info['port']}/{sql_info['schema']}"
        engine=create_engine(cmd)
        return engine

def GetEngine():
    Instance=Create_Engine()
    return Instance.conn


def GetSession():
    engine=Create_Engine().conn
    sessionmakers=sessionmaker(bind=engine,autoflush=False)
    session=sessionmakers()
    return session
