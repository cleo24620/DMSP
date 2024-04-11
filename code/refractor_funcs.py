# -*- coding: utf-8 -*-
# @Author  : cleo
# @Software: PyCharm

import time
import numpy as np
import pandas as pd
from netCDF4 import Dataset

def read_nc_file(file_path):
    """
    读取 NC 格式文件并返回数据变量的名字和说明。
    参数:
        file_path (str): NC 文件的路径。
    返回:
        dict: 包含变量名字和说明的字典。
    """
    # 使用 Dataset 打开文件
    with Dataset(file_path, 'r') as nc:
        variables_info = {}
        # 遍历文件中的变量
        for name, variable in nc.variables.items():
            # 获取变量的说明 and units（如果有的话）
            description = variable.description if 'description' in variable.ncattrs() else 'No description'
            units = variable.units if 'units' in variable.ncattrs() else 'No units'
            # the name of the variable serves as the key of the dictionary, and the description and unit of
            # the variable form a dictionary as the value of the dictionary.
            # we can add more metedata of the variable to the value of the dictionary.
            variables_info[name] = {'description': description, 'units': units}
    return variables_info


def get_data(file_path, variables):
    """return data as a DataFrame"""
    nc_obj = Dataset(file_path)
    data = pd.DataFrame()
    # col name is the v_name
    for v in variables:
        data[v] = nc_obj.variables[v][:]
    # add time_str col, i.e. turning unix time to str time. And the time is UTC.
    timestamps = data['timestamps']
    time_str = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) for t in timestamps]
    data.insert(0, 'time', time_str)
    return data


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
