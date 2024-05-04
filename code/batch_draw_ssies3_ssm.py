# -*- coding: utf-8 -*-
# @Author  : cleo
# @Software: PyCharm

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re

import pyDMSP

# ssies3
ssies3_info = pd.DataFrame()
ssies3_info['fp'] = pyDMSP.return_suffix_file_paths(r"G:\0_postgraduate\DMSP\data\2014\f16\ssies3")
dhm = []
for fp in ssies3_info['fp']:
    pattern = r"G:\\0_postgraduate\\DMSP\\data\\2014\\f16\\ssies3\\dmsp-f16_ssies-3_thermal-plasma_2014(\d{8})"
    match = re.search(pattern,fp)
    dhm.append(match.group(1))
ssies3_info['dhm'] = dhm
ssies3_info['day'] = [_[:4] for _ in dhm]
# ssm
ssm_fps = pyDMSP.return_suffix_file_paths(r"G:\0_postgraduate\DMSP\data\2014\f16\ssm")
ssm_info = pd.DataFrame()
ssm_info['fp'] = ssm_fps
days = []
for fp in ssm_info['fp']:
    pattern = r"G:\\0_postgraduate\\DMSP\\data\\2014\\f16\\ssm\\dmsp-f16_ssm_magnetometer_2014(\d{4})"
    match = re.search(pattern,fp)
    days.append(match.group(1))
ssm_info['day'] = days

# path
fig_save_path = r"G:\0_postgraduate\DMSP\fig\ssies3_ssm_v_delta_b"
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
                ssies3_data, ssm_data = pyDMSP.data_for_draw(ssies3_fp, ssm_fp)
                fig = pyDMSP.draw_ssies3_ssm(ssies3_data,ssm_data,ssies3_v_str='V_SC',ssm_v_str='DELTA_B_SC',
                                             ssies3_unit='m/s',ssm_unit='nT',dhm=ssies3_dhm,
                                             title_part='ssies3_ssm_v_delta_b',is_save=True,
                                             fig_save_path=fig_save_path)
                plt.close(fig)
            except Exception as e:
                print("------")
                print(f"{ssies3_fp} is a file that cross 2 day.")
                print(e)
                print("------")