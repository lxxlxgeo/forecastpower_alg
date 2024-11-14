from sqlalchemy import Column, BigInteger, Integer, DateTime, Float, Index
from sqlalchemy.dialects.mysql import TINYINT
from ..Base import Base
class PowerSolarPointCdqTable(Base):
    __tablename__ = 'power_solar_point_cdq_table'
    
    id = Column(BigInteger, primary_key=True, autoincrement=True, comment='主键id')
    data_source_id = Column(TINYINT, nullable=True, default=None, comment='数据源')
    site_id = Column(Integer, nullable=True, default=None, comment='站点id')
    fb_time = Column(DateTime, nullable=True, default=None, comment='发报时间')
    data_time = Column(DateTime, nullable=True, default=None, comment='预报时间')
    data_feature_value = Column(Float, nullable=True, default=None, comment='预报要素值')
    irr=Column(Float, nullable=True, default=None, comment='场站气象要素')
    create_time = Column(DateTime, nullable=True, default=None, comment='创建时间')
    update_time = Column(DateTime, nullable=True, default=None, comment='修改时间')

    __table_args__ = (
        Index('idx_normal', site_id, fb_time),
        {'comment': '单点预报结果表-超短期辐照度', 'mysql_engine': 'InnoDB', 'mysql_charset': 'utf8mb4'}
    )
