import libs
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from netCDF4 import Dataset
import time
import re


class Walen:
    def __init__(self, file_path_s1, file_path_ut):
        self.file_path_s1 = file_path_s1
        self.file_path_ut = file_path_ut
        self.data_details_s1 = self.read_nc_details(is_s1=True)
        self.data_details_ut = self.read_nc_details(is_ut=True)
        self.variables_s1 = self.data_details_s1.keys()
        self.variables_ut = self.data_details_ut.keys()

    def read_nc_details(self, is_s1=False, is_ut=False):
        if is_s1:
            file_path = self.file_path_s1
        elif is_ut:
            file_path = self.file_path_ut
        else:
            print("please choose 1 type in ut or s1")
            return
        # 使用 Dataset 打开文件
        with Dataset(file_path, 'r') as nc:
            data_details = {}
            # 遍历文件中的变量
            for name, variable in nc.variables.items():
                # 获取变量数据
                data = variable[:]
                # 提取变量的所有属性为字典
                attrs = {attr_name: variable.getncattr(attr_name) for attr_name in variable.ncattrs()}
                # 保存变量数据和属性
                data_details[name] = {
                    'data': data,
                    'attributes': attrs
                }
        return data_details

    def get_data(self, is_s1=False, is_ut=False):
        if is_s1:
            file_path = self.file_path_s1
            variables = self.variables_s1
        elif is_ut:
            file_path = self.file_path_ut
            variables = self.variables_ut
        else:
            print("please choose 1 type in ut or s1")
            return
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


file_path_s1 = r"G:\0_postgraduate\DMSP\data\2011\15s1\dms_20110101_15s1.001.nc"
file_path_ut = r"G:\0_postgraduate\DMSP\data\2011\ut\dms_ut_20110101_15.002.nc"
walen = Walen(file_path_s1,file_path_ut)
data_details_s1 = walen.read_nc_details(is_s1=True)
data_s1 = walen.get_data(is_s1=True)
data_details_ut = walen.read_nc_details(is_ut=True)
data_ut = walen.get_data(is_ut=True)

(mu0,r_mo,r_mh,r_mhe,NA) = (1.25663706212e-6,15.999,1.008,4.0026,6.02214076e23)  # r_mo 相对原子质量
mo = r_mo / (1000 * NA)  # kg 国际标准单位
mhe = r_mhe / (1000 * NA)  # kg 国际标准单位
mh = r_mh / (1000 * NA)  # kg 国际标准单位

ni = data_ut['ni']
po = data_ut['po+']
phe = data_ut['phe+']
ph = data_ut['ph+']
no = ni*po
nhe = ni*phe
nh = ni*ph

rho = no*mo + nhe*mhe + nh*mh
# t = data_ut['timestamps'][1]
# rho = rho[1]
#
# timestamps_s1 = data_s1['timestamps']
# idx = timestamps_s1.index[timestamps_s1.eq(t)]
# bd = data_s1['bd'][idx]
# diff_bd = data_s1['diff_bd'][idx]
# bd_model = bd - diff_bd
#
# va = bd_model / np.sqrt(rho*mu0)