from sqlalchemy import Column, BigInteger, Integer, DateTime, String, Index
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class QxRegionTable(Base):
    __tablename__ = 'qx_region_table'

    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    data_source_id = Column(Integer, nullable=True, comment='数据源')
    fb_time = Column(DateTime, nullable=True, comment='发报时间')
    data_time = Column(DateTime, nullable=True, comment='预报时间')
    solar_name = Column(String(128), nullable=True, comment='辐照度文件名称')
    solar_path = Column(String(128), nullable=True, comment='辐照度的图片文件路径')
    wind_name = Column(String(128), nullable=True, comment='风速的文件名称')
    wind_path = Column(String(128), nullable=True, comment='风速的图片文件路径')
    create_time = Column(DateTime, nullable=True, comment='创建时间')
    update_time = Column(DateTime, nullable=True, comment='修改时间')

    __table_args__ = (
        Index('idx_normal', 'fb_time'),
        {"comment": "气象区域预报结果表-风速图片+光伏图片"}
    )