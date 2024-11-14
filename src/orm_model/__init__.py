from src.orm_model.solar_fcst_table.powerSolarPoint_cdq import PowerSolarPointCdqTable
from src.orm_model.solar_fcst_table.powerSolarPoint_dq import PowerSolarPointDqTable
from src.orm_model.solar_fcst_table.powerSolarPoint_zq import PowerSolarPointZqTable
from src.orm_model.solar_fcst_table.powerSolarPoint_dl import PowerSolarPointDlTable



from src.orm_model.wins_fcst_table.powerWindPoint_cdq import PowerWindPointCdqTable
from src.orm_model.wins_fcst_table.powerWindPoint_dl import PowerWindPointDlTable
from src.orm_model.wins_fcst_table.powerWindPoint_zq import PowerWindPointZqTable
from src.orm_model.wins_fcst_table.powerWindPoint_dq import PowerWindPointDqTable


'''
整合表
'''

__all__=['PowerSolarPointCdqTable',
         'PowerSolarPointDlTable',
         'PowerSolarPointDqTable',
         'PowerSolarPointZqTable',
         'PowerWindPointCdqTable',
         'PowerWindPointDlTable',
         'PowerWindPointZqTable',
         'PowerWindPointDqTable']