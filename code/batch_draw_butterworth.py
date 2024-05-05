# -*- coding: utf-8 -*-
# @Author  : cleo
# @Software: PyCharm

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re
from scipy.signal import savgol_filter,butter, filtfilt

import pyDMSP

# ssies3

satellite = '18'
if satellite == '17':
    is_f17 = True
else:
    is_f17 = False

ssies3_info = pd.DataFrame()
ssies3_info['fp'] = pyDMSP.return_suffix_file_paths(rf"G:\0_postgraduate\DMSP\data\2014\f{satellite}\ssies3")
dhm = []
for fp in ssies3_info['fp']:
    # note: in python string, '\' 是转义符号，所以在pattern赋值时需要添加r
    # 同时需要注意'/'符号也可能导致报错？
    pattern = rf"dmsp-f{satellite}_ssies-3_thermal-plasma_\d{{4}}(\d{{8}})_v\d{{2}}\.cdf"
    match = re.search(pattern,fp)
    dhm.append(match.group(1))
ssies3_info['dhm'] = dhm
ssies3_info['day'] = [_[:4] for _ in dhm]
# ssm
ssm_fps = pyDMSP.return_suffix_file_paths(rf"G:\0_postgraduate\DMSP\data\2014\f{satellite}\ssm")
ssm_info = pd.DataFrame()
ssm_info['fp'] = ssm_fps
days = []
for fp in ssm_info['fp']:
    pattern = rf"dmsp-f{satellite}_ssm_magnetometer_\d{{4}}(\d{{4}})"
    match = re.search(pattern,fp)
    days.append(match.group(1))
ssm_info['day'] = days

# path
fig_save_path = rf"G:\0_postgraduate\DMSP\fig\ssies3_ssm_v_delta_b_butter_worth\f{satellite}"
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

# draw
for ssies3_fp,ssies3_dhm,ssies3_day in zip(ssies3_info['fp'], ssies3_info['dhm'], ssies3_info['day']):
    for ssm_fp,day_day in zip(ssm_info['fp'], ssm_info['day']):
        if ssies3_day == day_day:
            try:
                # # if file exits, jump draw.
                # save_fig_path = os.path.join(fig_save_path, f"ssies3_ssm_v_b_{dhm_ssies3}.png")
                # if os.path.isfile(save_fig_path):
                #     print(f"ssies3_ssm_v_b_{dhm_ssies3}.png exits, jump draw.")
                #     continue
                # Object CDFFile

                ssies3_data_SC, ssm_data_clip = pyDMSP.data_for_draw(ssies3_fp, ssm_fp, is_f17=is_f17)

                # preprocess
                ssies3_data_SC['V_SC_x_fillna'] = ssies3_data_SC['V_SC_x'].fillna(method='ffill').fillna(method='bfill')
                ssies3_data_SC['V_SC_y_fillna'] = ssies3_data_SC['V_SC_y'].fillna(method='ffill').fillna(method='bfill')
                ssies3_data_SC['V_SC_z_fillna'] = ssies3_data_SC['V_SC_z'].fillna(method='ffill').fillna(method='bfill')

                ssm_data_clip['DELTA_B_SC_x_fillna'] = ssm_data_clip['DELTA_B_SC_x'].fillna(method='ffill').fillna(
                    method='bfill')
                ssm_data_clip['DELTA_B_SC_y_fillna'] = ssm_data_clip['DELTA_B_SC_y'].fillna(method='ffill').fillna(
                    method='bfill')
                ssm_data_clip['DELTA_B_SC_z_fillna'] = ssm_data_clip['DELTA_B_SC_z'].fillna(method='ffill').fillna(
                    method='bfill')

                # 设定滤波器参数
                fs = 1  # 采样频率
                cutoff = 0.01  # 截止频率（Hz）
                nyq = 0.5 * fs  # 奈奎斯特频率
                normal_cutoff = cutoff / nyq  # 归一化截止频率

                # 设计Butterworth低通滤波器
                b = butter(N=5, Wn=normal_cutoff, btype='low', analog=False)[0]
                a = butter(N=5, Wn=normal_cutoff, btype='low', analog=False)[1]

                ssies3_data_SC['V_SC_x_fillna_butterworth'] = filtfilt(b, a, ssies3_data_SC['V_SC_x_fillna'])
                ssies3_data_SC['V_SC_y_fillna_butterworth'] = filtfilt(b, a, ssies3_data_SC['V_SC_y_fillna'])
                ssies3_data_SC['V_SC_z_fillna_butterworth'] = filtfilt(b, a, ssies3_data_SC['V_SC_z_fillna'])

                ssm_data_clip['DELTA_B_SC_x_fillna_butterworth'] = filtfilt(b, a, ssm_data_clip['DELTA_B_SC_x_fillna'])
                ssm_data_clip['DELTA_B_SC_y_fillna_butterworth'] = filtfilt(b, a, ssm_data_clip['DELTA_B_SC_y_fillna'])
                ssm_data_clip['DELTA_B_SC_z_fillna_butterworth'] = filtfilt(b, a, ssm_data_clip['DELTA_B_SC_z_fillna'])

                ssies3_v_strs = ['V_SC_x_fillna_butterworth','V_SC_y_fillna_butterworth','V_SC_z_fillna_butterworth']
                ssm_v_strs = ['DELTA_B_SC_x_fillna_butterworth','DELTA_B_SC_y_fillna_butterworth',
                              'DELTA_B_SC_z_fillna_butterworth']

                # draw
                fig = pyDMSP.draw_ssies3_ssm(ssies3_data_SC, ssm_data_clip, ssies3_v_strs,ssm_v_strs,
                                             left_unit='m/s', right_unit='nT', dhm=ssies3_dhm,
                                             title_part=f'ssies3_ssm_v_delta_b_butterworth_{cutoff}', satellite=satellite,
                                             is_save=True,fig_save_path=fig_save_path)
                if fig != None:
                    plt.close(fig)
            except Exception as e:
                print("------")
                print(f"{ssies3_fp} is a file that cross 2 day.")
                print(e)
                print("------")