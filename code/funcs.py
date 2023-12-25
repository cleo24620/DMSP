#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2023/11/9 20:46
# @Author  : cleo
# @Software: PyCharm

import time
import os

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
    timestamps = nc_obj.variables['timestamps'][:]
    time_str = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) for t in timestamps]
    data.insert(0, 'time', time_str)
    # use flag to shift vx,vy,vz data
    idm_bool = (data['idm_flag_ut'] == 1)
    rpa_bool = (data['rpa_flag_ut'] == 1)
    flag_bool = idm_bool * rpa_bool
    timestamps = data['timestamps'][flag_bool]
    time_str = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) for t in timestamps]
    vx = data['vx'][flag_bool]
    vy = data['vy'][flag_bool]
    vz = data['vz'][flag_bool]
    ni = data['ni'][flag_bool]
    ph = data['ph+'][flag_bool]
    phe = data['phe+'][flag_bool]
    po = data['po+'][flag_bool]
    data_dic = {'time':time_str,'timestamps':timestamps,'vx':vx,'vy':vy,'vz':vz,'ni':ni,'ph+':ph,'phe+':phe,'po+':po}
    data_flag = pd.DataFrame(data_dic)
    # # interp (np.interp)
    # # x
    # nts = np.arange(data_flag['timestamps'].iloc[0],data_flag['timestamps'].iloc[-1],4)
    # time_str = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) for t in nts]
    # # interp y
    # vx_interp = np.interp(nts,data_flag['timestamps'],data_flag['vx (m/s)'])
    # vy_interp = np.interp(nts,data_flag['timestamps'],data_flag['vy'])
    # vz_interp = np.interp(nts,data_flag['timestamps'],data_flag['vz'])
    # data_dic = {'Time': time_str, 'timestamps': nts,
    #             'vx (m/s)': vx_interp, 'vy': vy_interp, 'vz': vz_interp}
    # data_flag_interp = pd.DataFrame(data_dic)
    # scipy
    # x
    # nts = np.arange(data['timestamps'].iloc[0], data['timestamps'].iloc[-1], 4)
    # time_str = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) for t in nts]
    # f = interp1d(data_vxyz_flag['timestamps'],data_vxyz_flag['vx (m/s)'],kind='quadratic')
    # vx_interp = f(nts)
    # f = interp1d(data_vxyz_flag['timestamps'],data_vxyz_flag['vy'],kind='quadratic')
    # vy_interp = f(nts)
    # f = interp1d(data_vxyz_flag['timestamps'],data_vxyz_flag['vz'],kind='quadratic')
    # vz_interp = f(nts)
    # # 处理nan ni ph...
    # ni_nan_bool = data_pd['Ni'].isna()
    # ph_nan_bool = data_pd['ph+'].isna()
    # phe_nan_bool = data_pd['phe+'].isna()
    # po_nan_bool = data_pd['po+'].isna()
    # ni_nan_idx = data_pd['timestamps'][ni_nan_bool]
    # ph_nan_idx = data_pd['timestamps'][ph_nan_bool]
    # phe_nan_idx = data_pd['timestamps'][phe_nan_bool]
    # po_nan_idx = data_pd['timestamps'][po_nan_bool]
    # # interp x
    # x = data_pd['timestamps'][~ni_nan_bool]
    # y = data_pd['Ni'][~ni_nan_bool]
    # f = interp1d(x,y,kind='quadratic')
    # ni_nan_intepr =
    # data['Ni'].interpolate(inplace=True)
    # data['ph+'].interpolate(inplace=True)
    # data['phe+'].interpolate(inplace=True)
    # data['po+'].interpolate(inplace=True)
    # nts = np.arange(data['timestamps'].iloc[0], data['timestamps'].iloc[-1], 4)
    # time_str = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) for t in nts]
    # f = interp1d(data['timestamps'], data['Ni'], kind='linear')
    # ni_interp = f(nts)
    # f = interp1d(data['timestamps'], data['ph+'], kind='linear')
    # ph_interp = f(nts)
    # f = interp1d(data['timestamps'], data['phe+'], kind='linear')
    # phe_interp = f(nts)
    # f = interp1d(data['timestamps'], data['po+'], kind='linear')
    # po_interp = f(nts)
    # data_dic = {'Time': time_str, 'timestamps': nts,
    #             'vx (m/s)': vx_interp, 'vy': vy_interp, 'vz': vz_interp,
    #             'Ni':ni_interp,'ph+':ph_interp,'phe+':phe_interp,'po+':po_interp}
    # data_interp = pd.DataFrame(data_dic)
    data_flag = data_flag.reset_index(drop=True)
    return data,data_flag


def read_1s(dp,fn,dp_ut,fn_ut):
    fp = dp + fn
    nc_obj = Dataset(fp)
    data = pd.DataFrame()
    data['timestamps'] = nc_obj.variables['timestamps'][:]
    data['bx'] = nc_obj.variables['b_forward'][:] * 1e9  # T -> nT
    data['by'] = nc_obj.variables['b_perp'][:] * 1e9
    data['bz'] = nc_obj.variables['bd'][:] * 1e9
    data['diff_bx'] = nc_obj.variables['diff_b_for'][:] * 1e9
    data['diff_by'] = nc_obj.variables['diff_b_perp'][:] * 1e9
    data['diff_bz'] = nc_obj.variables['diff_bd'][:] * 1e9
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
    for v in ['timestamps', 'bx', 'by', 'bz', 'diff_bx', 'diff_by', 'diff_bz','mbx', 'mby', 'mbz']:
        data_1s_to_ut[v] = data[v][data['bool_to_ut']]
    data_1s_to_ut = data_1s_to_ut.reset_index(drop=True)
    return data,data_1s_to_ut


def dHTA(V, B, time):
    r = len(time)
    Kijm = np.empty((3, 3, r))
    K1ijm = np.empty((3, 3, r))
    K2ijm = np.empty((3, 3, r))
    for m in range(r):
        BB = B[m, 0]**2 + B[m, 1]**2 + B[m, 2]**2
        for i in range(3):
            for j in range(3):
                if i == j:
                    Kijm[i, j, m] = BB * (1 - B[m, i] * B[m, j] / BB)
                else:
                    Kijm[i, j, m] = -B[m, i] * B[m, j]
        K1ijm[:, :, m] = Kijm[:, :, m] * time[m]
        K2ijm[:, :, m] = Kijm[:, :, m] * (time[m] ** 2)
    K0 = np.sum(Kijm, 2) / r
    K1 = np.sum(K1ijm, 2) / r
    K2 = np.sum(K2ijm, 2) / r
    Rb = np.empty((3, r))
    Rb2 = np.empty((3, r))
    for m in range(r):
        Rb[:, m] = np.dot(Kijm[:, :, m], np.array([V[m, 0], V[m, 1], V[m, 2]]))
        Rb2[:, m] = Rb[:, m] * time[m]
    VHT = np.dot(np.linalg.inv(K0), (np.sum(Rb, axis=1) / r))
    RHS1 = np.sum(Rb, 1) / r
    RHS2 = np.sum(Rb2, 1) / r
    # VHT0aHT = np.linalg.solve(np.array([[K0,K1],[K1,K2]]),np.array([[RHS1],[RHS2]]))
    VHT0aHT = np.linalg.solve(
        np.hstack(
            (np.vstack(
                (K0, K1)), np.vstack(
                (K1, K2)))), np.hstack(
                    (RHS1, RHS2)))
    VHT0 = VHT0aHT[:3]
    aHT = VHT0aHT[3:6]
    Vht = VHT
    Vht0 = VHT0
    aVht = aHT
    return Vht, Vht0, aVht


def need_data(d_ut_flag,d_1s_to_ut,year,month,day,root="../data/csv/"):
    """
    返回分析所需的数据，并将数据存储为csv文件。
    """
    needed_d = d_ut_flag.copy()
    needed_d[['vx','vy','vz']] = needed_d[['vx','vy','vz']] / 1e3
    needed_d[['bx','by','bz','diff_bx','diff_by','diff_bz','mbx','mby','mbz']] = d_1s_to_ut[['bx','by','bz','diff_bx','diff_by','diff_bz','mbx','mby','mbz']]
    # needed_d['time_intervals'] = np.zeros(len(needed_d))
    ls = np.zeros(len(needed_d))
    for i in range(0,len(needed_d)-1):
        ls[i] = needed_d['timestamps'].iloc[i+1] - needed_d['timestamps'].iloc[i]
    needed_d['time_intervals'] = ls
    needed_d = needed_d.interpolate()  # 少数离子数量密度数据为NaN，此处使用插值进行填充
    # HT Frame
    V = np.array([needed_d['vx'], needed_d['vy'], needed_d['vz']]).T
    B = np.array([needed_d['bx'], needed_d['by'], needed_d['bz']]).T
    time = np.array(needed_d['time_intervals'])
    Vht, Vht0, aVht = dHTA(V, B, time)
    ht_frame = pd.DataFrame({'Vht':Vht,'Vht0':Vht0,'aVht':aVht})
    # 引入 HT Frame 后的 need_data
    needed_d['vx_prime'] = needed_d['vx'] - ht_frame['Vht'][0]
    needed_d['vy_prime'] = needed_d['vy'] - ht_frame['Vht'][1]
    needed_d['vz_prime'] = needed_d['vz'] - ht_frame['Vht'][2]
    # 求 va
    (mu0,r_mo,r_mh,r_mhe,NA) = (1.25663706212e-6,15.999,1.008,4.0026,6.02214076e23)
    nh = needed_d['ph+'] * needed_d['ni']
    nhe = needed_d['phe+'] * needed_d['ni']
    no = needed_d['po+'] * needed_d['ni']
    mo = r_mo / (1000 * NA)  # kg 国际标准单位
    mh = r_mh / (1000 * NA)
    mhe = r_mhe / (1000 * NA)
    rho = sum([no * mo, nh * mh, nhe * mhe])
    _ = np.sqrt(mu0 * rho)
    # 使用观测磁场，则求得的va过大，远远超过第一宇宙速度，所以使用扰动磁场（但是注意：va并不是离子运动的速度）
    vax = needed_d['mbx'] / (_ * 1e9 * 1e3)  # km/s
    vay = needed_d['mby'] / (_ * 1e9 * 1e3)
    vaz = needed_d['mbz'] / (_ * 1e9 * 1e3)
    needed_d['vax'] = vax
    needed_d['vay'] = vay
    needed_d['vaz'] = vaz
    # 存储数据
    if not os.path.exists(root):
        os.makedirs(root)
        needed_d.to_csv(root+f"needed_data_{year}-{month}-{day}.csv", index=False)
    needed_d.to_csv(root+f"needed_data_{year}-{month}-{day}.csv", index=False)
    ht_frame.to_csv(root+f"HT_Frame_{year}-{month}-{day}.csv", index=False)
    return needed_d


def ssm(dp,fn,data_ut_flag):
    fp = dp+fn
    data = np.loadtxt(fp, skiprows=6, usecols=[0, 1, 4, 5, 6, 9, 10, 11])
    fts = [str(int(t)) for t in data[:, 0]]
    ssm_uts = [time.mktime(time.strptime(t, '%Y%j%H%M%S')) + 8 * 60 * 60 for t in fts]  # pay attention to time_zone
    # seconds = data[:, 1]
    # seconds = [int(s) for s in seconds]
    geo_lat = data[:, 2]
    geo_lon = data[:, 3]
    alt = data[:, 4]
    m_bx = data[:, 5]
    m_by = data[:, 6]
    m_bz = data[:, 7]
    data_dic = {'ssm_uts': ssm_uts,
                'm_bx': m_bx, 'm_by': m_by, 'm_bz': m_bz}
    data_index = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) for t in ssm_uts]
    data_ssm = pd.DataFrame(data_dic, index=data_index)
    for_compare_t = data_ut_flag['timestamps']
    data_ssm['bool_to_ut'] = data_ssm['ssm_uts']
    data_ssm['bool_to_ut'][0] = 1
    for idx, t_ssm in enumerate(data_ssm['ssm_uts']):
        data_ssm['bool_to_ut'][idx] = (t_ssm in list(data_ut_flag['timestamps']))
        # 此处要用list, 如果使用`pd.series() or pd.dataframe`, 则会返回含索引列的对应数据对象, 返回的结果一定是`false`
    ssm_uts_bool = data_ssm['ssm_uts'][data_ssm['bool_to_ut']]
    m_bx_bool = data_ssm['m_bx'][data_ssm['bool_to_ut']]
    m_by_bool = data_ssm['m_by'][data_ssm['bool_to_ut']]
    m_bz_bool = data_ssm['m_bz'][data_ssm['bool_to_ut']]
    data = {'ssm_uts': ssm_uts_bool, 'm_bx': m_bx_bool, 'm_by': m_by_bool, 'm_bz': m_bz_bool}
    data_index = data_ssm.index
    data_index = data_index[data_ssm['bool_for_ut']]
    data_ssm_to_ut = pd.DataFrame(data, data_index)
    return data,data_ssm_to_ut