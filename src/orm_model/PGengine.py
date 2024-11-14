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
import psycopg2 
'''导入模型'''
#
# from Data_Reader.control_framework_orm.models import Weather_station_his,Weather_station_his_check
#%%
'''
连接字符串

'''
import urllib.parse

# 替换以下变量为你的数据库信息
username = 'postgres'
password = urllib.parse.quote('pg@123')
host = '120.27.111.50'  # 通常是 'localhost' 或者数据库的 IP 地址
port = '5433'  # PostgreSQL 默认端口
database = 'kys_power_yd'

# 创建连接字符串
connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}'

'''
创建数据库引擎
'''
class Create_Engine(object):
    def __init__(self):
        self.conn=self.conn_to_sql()
    def conn_to_sql(self):
        
        #pymysql.install_as_MySQLdb()
        connection_string = f'postgresql://{username}:{password}@{host}:{port}/{database}'
        engine=create_engine(connection_string)
        return engine

def GetEngine():
    Instance=Create_Engine()
    return Instance.conn


def GetSession():
    engine=Create_Engine()
    engine=engine.conn
    sessionmakers=sessionmaker(bind=engine,autoflush=False)
    session=sessionmakers()
    return session