# -*- coding: utf-8 -*-
# @Author  : cleo
# @Software: PyCharm

import os
from datetime import datetime

from netCDF4 import Dataset
import spacepy.pycdf
import h5py
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re

import libs_cleo

# ssies3
ssies3_infos = pd.DataFrame()
ssies3_infos['fp'] = libs_cleo.return_suffix_file_paths(r"G:\0_postgraduate\DMSP\data\2014\f16\ssies3")
dhm = []
for fp in ssies3_infos['fp']:
    pattern = r"G:\\0_postgraduate\\DMSP\\data\\2014\\f16\\ssies3\\dmsp-f16_ssies-3_thermal-plasma_2014(\d{8})"
    match = re.search(pattern,fp)
    dhm.append(match.group(1))
ssies3_infos['dhm'] = dhm
ssies3_infos['day'] = [_[:4] for _ in dhm]
# ssm
ssm_fps = libs_cleo.return_suffix_file_paths(r"G:\0_postgraduate\DMSP\data\2014\f16\ssm")
ssm_infos = pd.DataFrame()
ssm_infos['fp'] = ssm_fps
days = []
for fp in ssm_infos['fp']:
    pattern = r"G:\\0_postgraduate\\DMSP\\data\\2014\\f16\\ssm\\dmsp-f16_ssm_magnetometer_2014(\d{4})"
    match = re.search(pattern,fp)
    days.append(match.group(1))
ssm_infos['day'] = days

fig_save_path = r"G:\0_postgraduate\DMSP\fig\ssies3_ssm_v_b"
if not os.path.exists(fig_save_path):
    os.makedirs(fig_save_path)

for fp_ssies3,dhm_ssies3,day_ssies3 in zip(ssies3_infos['fp'],ssies3_infos['dhm'],ssies3_infos['day']):
    for fp_ssm,day_ssm in zip(ssm_infos['fp'],ssm_infos['day']):
        if day_ssies3 == day_ssm:
            try:
                # Object CDFFile
                ssies3 = libs_cleo.CDFFile(fp_ssies3)
                ssm = libs_cleo.CDFFile(fp_ssm)
                # data
                data_ssies3 = ssies3.ssies3_data()
                data_ssm = ssm.ssm_data()
                # v set nan
                vx_set_nan = ssies3.vx_set_nan()
                vy_set_nan = ssies3.v_yz_set_nan(v_str='vy')
                vz_set_nan = ssies3.v_yz_set_nan(v_str='vz')
                # change the coordinate system of v
                data_dic = {'Epoch':vx_set_nan['Epoch'], 'V_SC_x':-vz_set_nan['vz'], 'V_SC_y':vx_set_nan['vx'], 'V_SC_z':-vy_set_nan['vy']}
                ssies3_data_SC = pd.DataFrame(data_dic)
                # ssm clip
                st = data_ssies3['timestamps'].iloc[0]
                et = data_ssies3['timestamps'].iloc[-1]
                s_idx = np.where(data_ssm['timestamps']==st)
                e_idx = np.where(data_ssm['timestamps']==et)
                data_ssm_1 = data_ssm.iloc[s_idx[0][0]:e_idx[0][0]+1].reset_index(drop=True)
                # draw
                fig,axs = plt.subplots(3, 1,figsize=(20,10*3))
                ax1 = axs[0]
                ax2 = axs[1]
                ax3 = axs[2]
                # fig title
                fig.suptitle(f'{dhm_ssies3} ssies3_ssm_v_b')
                # x
                ax1.plot(ssies3_data_SC['Epoch'],ssies3_data_SC['V_SC_x'],label='vx')
                ax1.plot(data_ssm_1['Epoch'],data_ssm_1['DELTA_B_SC_x'],label='delta_bx')
                # ax.plot(data_ssm_1['Epoch'],data_ssm_1['DELTA_B_SC_ORIG_y'],label='delta_b_orig_x')
                ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))  # 只在某些时刻显示刻度
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 格式化显示格式
                # 在x=0处添加一条红色虚线
                ax1.axhline(y=0, color='red', linestyle='--', label='y = 0')
                ax1.legend()
                ax1.set_title('x direction')
                # y
                ax2.plot(ssies3_data_SC['Epoch'],ssies3_data_SC['V_SC_y'],label='vy')
                ax2.plot(data_ssm_1['Epoch'],data_ssm_1['DELTA_B_SC_y'],label='delta_by')
                # ax.plot(data_ssm_1['Epoch'],data_ssm_1['DELTA_B_SC_ORIG_z'],label='delta_b_orig_y')
                ax2.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))  # 只在某些时刻显示刻度
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 格式化显示格式
                # 在x=0处添加一条红色虚线
                ax2.axhline(y=0, color='red', linestyle='--', label='y = 0')
                ax2.legend()
                ax2.set_title('y direction')
                # z
                ax3.plot(ssies3_data_SC['Epoch'],ssies3_data_SC['V_SC_z'],label='vz')
                ax3.plot(data_ssm_1['Epoch'],data_ssm_1['DELTA_B_SC_z'],label='delta_bz')
                # ax.plot(data_ssm_1['Epoch'],data_ssm_1['DELTA_B_SC_ORIG_x'],label='delta_b_orig_z')
                ax3.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))  # 只在某些时刻显示刻度
                ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 格式化显示格式
                # 在x=0处添加一条红色虚线
                ax3.axhline(y=0, color='red', linestyle='--', label='y = 0')
                ax3.legend()
                ax3.set_title('z direction')
                # save
                save_fig_path = os.path.join(fig_save_path,f"ssies3_ssm_v_b_{dhm_ssies3}.png")
                plt.savefig(save_fig_path)
                plt.close(fig)
                print(f"ssies3_ssm_v_b_{dhm_ssies3}.png already save.")
            except Exception as e:
                print("------")
                print(f"{fp_ssies3} is a file that cross 2 day.")
                print(e)
                print("------")