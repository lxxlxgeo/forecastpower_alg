from sqlalchemy import create_engine, Column, Integer, SmallInteger, String, DateTime, BigInteger, PrimaryKeyConstraint, Index
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

Base = declarative_base()

class QxRegionTable(Base):
    __tablename__ = 'qx_region_table'
    __table_args__ = {
        'extend_existing': True
    }

    yb_type = Column(SmallInteger, comment='预报类型')  # 使用 TINYINT 代替 SmallInteger
    data_from_id = Column(Integer, comment='预报数据源')
    fb_time = Column(DateTime, comment='发报时间')
    data_time = Column(DateTime, comment='预报时间')
    solar_path = Column(String(128), comment='辐照度的图片文件路径')
    solar_s_path=Column(String(128),comment='辐照度的邮票图路径')
    temp_path = Column(String(255), comment='温度图片路径')
    temp_s_path=Column(String(255),comment='温度的邮票图路径')
    pre_path = Column(String(255), comment='湿度图片路径')
    pre_s_path=Column(String(255),comment='湿度的邮票图路径')
    wind10_path = Column(String(128), comment='10米风速的图片文件路径')
    wind10_s_path=Column(String(128),comment='10米风速的图片文件路径')
    wind100_path = Column(String(255), comment='100米风速的图片文件路径')
    wind100_s_path=Column(String(255),comment='100米风速图片文件路径')
    create_time = Column(DateTime, comment='创建时间')
    update_time = Column(DateTime, comment='修改时间')

    __table_args__ = (
        PrimaryKeyConstraint('yb_type', 'data_from_id', 'fb_time', 'data_time', name='qx_region_table_pkey'),
    )