#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 20:46
# @Author  : cleo
# @Software: PyCharm

import time
import numpy as np
import pandas as pd
from netCDF4 import Dataset


def read_ut(dp,fn):
    fp = dp+fn
    nc_obj = Dataset(fp)
    data = pd.DataFrame()
    nc_variables = ["timestamps","gdlat","glon","gdalt",
                    "ni","ph+","phe+","po+","ion_v_sat_for","ion_v_sat_left","vert_ion_v",
                    "idm_flag_ut","rpa_flag_ut"]
    col_names = ["timestamps","gdlat","glon","gdalt",
                    "ni","ph+","phe+","po+","vx","vy","vz",
                    "idm_flag_ut","rpa_flag_ut"]
    for v,cn in zip(nc_variables,col_names):
        data[cn] = nc_obj.variables[v][:]
    # add time_str col, i.e. turning unix time to str time. And the time is UTC.
    timestamps = data['timestamps']
    time_str = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) for t in timestamps]
    data.insert(0, 'time', time_str)
    # use flag to shift vx,vy,vz data
    idm_bool = (data['idm_flag_ut'] == 1)
    rpa_bool = (data['rpa_flag_ut'] == 1)
    flag_bool = idm_bool * rpa_bool
    timestamps,vx,vy,vz,ni,ph,phe,po=(data['timestamps'][flag_bool],data['vx'][flag_bool],data['vy'][flag_bool],
                                      data['vz'][flag_bool],data['ni'][flag_bool],data['ph+'][flag_bool],
                                      data['phe+'][flag_bool],data['po+'][flag_bool])
    time_str = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) for t in timestamps]
    data_dic = {'time':time_str,'timestamps':timestamps,'vx':vx,'vy':vy,'vz':vz,'ni':ni,'ph+':ph,'phe+':phe,'po+':po}
    data_flag = pd.DataFrame(data_dic).reset_index(drop=True)
    return data,data_flag


def read_1s(dp,fn,dp_ut,fn_ut):
    fp = dp + fn
    nc_obj = Dataset(fp)
    data = pd.DataFrame()
    nc_variables = ['timestamps','b_forward','b_perp','bd','diff_b_for','diff_b_perp','diff_bd','ne']
    col_names = ['timestamps','bx','by','bz','diff_bx','diff_by','diff_bz','ne']
    for v,cn in zip(nc_variables,col_names):
        data[cn] = nc_obj.variables[v][:]
    data['mbx'] = data['bx'] - data['diff_bx']
    data['mby'] = data['by'] - data['diff_by']
    data['mbz'] = data['bz'] - data['diff_bz']
    # ut数据和1s数据映射到相同时间点列表
    d1, d2 = read_ut(dp_ut, fn_ut)
    for_bool = list(d2['timestamps'])
    data['bool_to_ut'] = np.zeros(len(data['timestamps']))
    for i, t in enumerate(data['timestamps']):
        data['bool_to_ut'][i] = (t in for_bool)  # 创建新列然后根据索引重新对列元素赋值, 不能直接赋值
    data_1s_to_ut = pd.DataFrame()
    for v in ['timestamps', 'bx', 'by', 'bz', 'diff_bx', 'diff_by', 'diff_bz','mbx', 'mby', 'mbz','ne']:
        data_1s_to_ut[v] = data[v][data['bool_to_ut']]
    data_1s_to_ut = data_1s_to_ut.reset_index(drop=True)
    return data,data_1s_to_ut


def need_data(d_ut_flag,d_1s_to_ut):
    needed_d = d_ut_flag.copy()
    needed_d[['bx','by','bz','diff_bx','diff_by','diff_bz','mbx','mby','mbz','ne']] = d_1s_to_ut[['bx','by','bz','diff_bx','diff_by','diff_bz','mbx','mby','mbz','ne']]
    # needed_d['time_intervals'] = np.zeros(len(needed_d))
    ls = np.zeros(len(needed_d))
    for i in range(0,len(needed_d)-1):
        ls[i] = needed_d['timestamps'].iloc[i+1] - needed_d['timestamps'].iloc[i]
    needed_d['time_intervals'] = ls
    needed_d = needed_d.interpolate()  # 少数离子数量密度数据为NaN，此处使用插值进行填充
    # 求 va
    (mu0,r_mo,r_mh,r_mhe,NA) = (1.25663706212e-6,15.999,1.008,4.0026,6.02214076e23)
    nh = needed_d['ph+'] * needed_d['ne']
    nhe = needed_d['phe+'] * needed_d['ne']
    no = needed_d['po+'] * needed_d['ne']
    mo = r_mo / (1000 * NA)  # kg 国际标准单位
    mh = r_mh / (1000 * NA)
    mhe = r_mhe / (1000 * NA)
    rho = sum([no * mo, nh * mh, nhe * mhe])
    _ = np.sqrt(mu0 * rho)
    # 使用观测磁场，则求得的va过大，远远超过第一宇宙速度，所以使用扰动磁场（但是注意：va并不是离子运动的速度）
    vax = needed_d['mbx'] / _  # km/s
    vay = needed_d['mby'] / _
    vaz = needed_d['mbz'] / _
    needed_d['vax'] = vax
    needed_d['vay'] = vay
    needed_d['vaz'] = vaz
    return needed_d


dp_ut="../data/f15ut/2011/"
fp_ut="dms_ut_20110101_15.002.nc"
dp_1s="../data/f15s1/2011/"
fp_1s="dms_20110101_15s1.001.nc"
d_ut,d_ut_flag = read_ut(dp_ut,fp_ut)
d_1s,d_1s_to_ut = read_1s(dp_1s,fp_1s,dp_ut,fp_ut)
need_d = need_data(d_ut_flag,d_1s_to_ut)

# # 绘图
# datetime_ = pd.to_datetime(seconds,unit='s',
#                origin=pd.Timestamp('2011-09-09'))
# data_dic = {'datetime':datetime_, 'm_bz':m_bz}
# data_pd = pd.DataFrame(data_dic,index=datetime_)
# data_pd['m_bz'].plot()
#
# # 绘制多个函数
# data_ssm_bool[['m_bx','m_by','m_bz']].plot(rot=20,figsize=(15,5))
#
# 标记x轴坐标值
# fig,ax = plt.subplots()
# data_f15_ut.plot(ax=ax,x='timestamps',y='vz',rot=20)
# for_xlabels = [d[-8:] for d in data_f15_ut['time'][[0,2698,5396,8094,10792]]]
# ax.set_xticks(data_f15_ut['timestamps'][[0,2698,5396,8094,10792]],labels=for_xlabels)