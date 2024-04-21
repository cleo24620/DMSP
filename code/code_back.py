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

# def need_data(d_ut_flag,d_1s_to_ut):
#     needed_d = d_ut_flag.copy()
#     needed_d[['bx','by','bz','diff_bx','diff_by','diff_bz','mbx','mby','mbz','ne']] = d_1s_to_ut[['bx','by','bz','diff_bx','diff_by','diff_bz','mbx','mby','mbz','ne']]
#     # needed_d['time_intervals'] = np.zeros(len(needed_d))
#     ls = np.zeros(len(needed_d))
#     for i in range(0,len(needed_d)-1):
#         ls[i] = needed_d['timestamps'].iloc[i+1] - needed_d['timestamps'].iloc[i]
#     needed_d['time_intervals'] = ls
#     needed_d = needed_d.interpolate()  # 少数离子数量密度数据为NaN，此处使用插值进行填充
#     # 求 va
#     (mu0,r_mo,r_mh,r_mhe,NA) = (1.25663706212e-6,15.999,1.008,4.0026,6.02214076e23)
#     nh = needed_d['ph+'] * needed_d['ne']
#     nhe = needed_d['phe+'] * needed_d['ne']
#     no = needed_d['po+'] * needed_d['ne']
#     mo = r_mo / (1000 * NA)  # kg 国际标准单位
#     mh = r_mh / (1000 * NA)
#     mhe = r_mhe / (1000 * NA)
#     rho = sum([no * mo, nh * mh, nhe * mhe])
#     _ = np.sqrt(mu0 * rho)
#     # 使用观测磁场，则求得的va过大，远远超过第一宇宙速度，所以使用扰动磁场（但是注意：va并不是离子运动的速度）
#     vax = needed_d['mbx'] / _  # km/s
#     vay = needed_d['mby'] / _
#     vaz = needed_d['mbz'] / _
#     needed_d['vax'] = vax
#     needed_d['vay'] = vay
#     needed_d['vaz'] = vaz
#     return needed_d