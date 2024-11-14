from sqlalchemy import Column, BigInteger, Integer, DateTime,Index,Float
from sqlalchemy.dialects.mysql import TINYINT
from ..Base import Base

class PowerWindPointDlTable(Base):
    __tablename__ = 'power_wind_point_dl_table'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    data_source_id = Column(TINYINT, nullable=True, default=None, comment='数据源')
    site_id = Column(Integer, nullable=True, default=None, comment='站点id')
    fb_time = Column(DateTime, nullable=True, default=None, comment='发报时间')
    data_time = Column(DateTime, nullable=True, default=None, comment='预报时间')
    data_feature_value = Column(Float, nullable=True, default=None, comment='预报要素值')
    winds=Column(Float, nullable=True, default=None, comment='场站气象要素')
    height = Column(Float, nullable=True, default=None, comment='高度')
    create_time = Column(DateTime, nullable=True, default=None, comment='创建时间')
    update_time = Column(DateTime, nullable=True, default=None, comment='修改时间')

    __table_args__ = (
        Index('idx_normal', fb_time),
        {'comment': '单点预报结果表-短临风速', 'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )
