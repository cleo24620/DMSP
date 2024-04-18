# -*- coding: utf-8 -*-
# @Author  : cleo
# @Software: PyCharm

import time
import pandas as pd
from netCDF4 import Dataset

### 代码备份
# only ut file has the quality control-- idm and rpa-- for velocity of ions.
# idm_bool = (data['idm_flag_ut'] == 1)
# rpa_bool = (data['rpa_flag_ut'] == 1)
# flag_bool = idm_bool * rpa_bool

# # ut数据和1s数据映射到相同时间点列表
# d1, d2 = read_ut(dp_ut, fn_ut)
# for_bool = list(d2['timestamps'])
# data['bool_to_ut'] = np.zeros(len(data['timestamps']))
# for i, t in enumerate(data['timestamps']):
#     data['bool_to_ut'][i] = (t in for_bool)  # 创建新列然后根据索引重新对列元素赋值, 不能直接赋值
# data_1s_to_ut = pd.DataFrame()
# for v in ['timestamps', 'bx', 'by', 'bz', 'diff_bx', 'diff_by', 'diff_bz','mbx', 'mby', 'mbz','ne']:
#     data_1s_to_ut[v] = data[v][data['bool_to_ut']]
# data_1s_to_ut = data_1s_to_ut.reset_index(drop=True)

# 少数离子数量密度数据为NaN，用插值进行填充

# basic parameter
# (mu0,r_mo,r_mh,r_mhe,NA) = (1.25663706212e-6,15.999,1.008,4.0026,6.02214076e23)  # r_mo 相对原子质量
# mo = r_mo / (1000 * NA)  # kg 国际标准单位
###