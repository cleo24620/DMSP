# -*- coding: utf-8 -*-
# @Author  : cleo
# @Software: PyCharm

import os
from datetime import datetime
from typing import Optional

from netCDF4 import Dataset
import spacepy.pycdf
import h5py
import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import re



class NcFile:
    def __init__(self,file_path,s1_or_ut='s1'):
        """
        :param file_path: absolute path
        """
        self.file_path = file_path
        self.directory = os.path.dirname(self.file_path)
        self.filename = os.path.basename(self.file_path)
        self.data_details = self.return_data_details()
        self.variables = self.data_details.keys()
        self.original_data = self.return_original_data()
        # 正则表达式匹配四位数字年份
        match = re.search(r'\b\d{4}\b', self.directory)
        if match:
            self.year = match.group(0)
        else:
            print("No year found in the path.")
        if s1_or_ut == 's1':
            # 提取文件名中的日期
            self.date = os.path.basename(self.filename).split('_')[1]
        elif s1_or_ut == 'ut':
            # 提取文件名中的日期
            self.date = os.path.basename(self.filename).split('_')[2]

    def return_data_details(self) ->dict:
        """数据变量名及其描述"""
        # 使用 Dataset 打开文件
        with Dataset(self.file_path, 'r') as nc:
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

    def return_original_data(self) ->pd.DataFrame:
        """这个NC文件的数据包含timestamps。返回原始数据。同时通过时间戳获取时间字符串，并将其作为DataFrame的第1列。"""
        nc_obj = Dataset(self.file_path)
        data = pd.DataFrame()
        # col name is the v_name
        for v in self.variables:
            data[v] = nc_obj.variables[v][:]
        # add time_str col, i.e. turning unix time to str time. And the time is UTC.
        try:
            timestamps = data['timestamps']
        except:
            raise Exception("数据没有“timestapms”列")
        # time_str = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) for t in timestamps]
        time_datetime = pd.to_datetime(timestamps,unit='s')
        data.insert(0, 'Epoch', time_datetime)
        return data

    def preprocess_data1(self):
        """处理经度180度和-180度的突变；去掉经度的变化率非常大的部分。最后得到分割好的纬度、经度、磁扰。
        这个函数可用于绘制MagneticDisturbanceFigure类"""
        latitude = self.original_data['gdlat']
        longitude = self.original_data['glon']
        # 磁场数据（T转nT）
        diff_bt = np.sqrt(
            self.original_data['diff_b_for'] ** 2 + self.original_data['diff_b_perp'] ** 2 + self.original_data['diff_bd'] ** 2) * 1e9
        diff_bt_ff_bf = diff_bt.fillna(method='ffill').fillna(method='bfill')
        # 检测并处理经度(180,-180)附近的突变点
        threshold = 180  # 设置一个阈值来检测突变点
        segments = []
        start = 0
        num_data = len(latitude)
        for i in range(1, num_data):
            # 检测经度突变
            if np.abs(longitude[i] - longitude[i - 1]) > threshold:
                # 存储当前段的数据
                segments.append((latitude[start:i], longitude[start:i], diff_bt_ff_bf[start:i]))
                start = i  # 更新下一段的起点

        # 添加最后一段
        segments.append((latitude[start:], longitude[start:], diff_bt_ff_bf[start:]))
        preprocessed_segments = []
        for segment in segments:
            longitude = segment[1]
            latitude = segment[0]
            diff_bt_ff_bf = segment[2]
            differences = np.diff(longitude)
            sign_changes = np.sign(differences[:-1]) != np.sign(differences[1:])
            change_indices = np.where(sign_changes)[0] + 1
            # 分割 Series：如果有多个变号点，选择第一个进行分割
            if len(change_indices) > 0:
                first_change_index = change_indices[0]
                longitude1 = longitude[:first_change_index + 1]  # 包括变号点在第一个 Series 中
                longitude2 = longitude[first_change_index + 1:]
                latitude1 = latitude[:first_change_index + 1]
                latitude2 = latitude[first_change_index + 1:]
                diff_bt_ff_bf1 = diff_bt_ff_bf[:first_change_index + 1]
                diff_bt_ff_bf2 = diff_bt_ff_bf[first_change_index + 1:]
                # 计算两个 Series 的一阶差分的平均绝对值
                mean_abs_diff1 = np.mean(np.abs(np.diff(longitude1)))
                mean_abs_diff2 = np.mean(np.abs(np.diff(longitude2)))

                # 选择平均绝对值较小的 Series
                selected_longitude = longitude1 if mean_abs_diff1 < mean_abs_diff2 else longitude2
                selected_latitude = latitude1 if mean_abs_diff1 < mean_abs_diff2 else latitude2
                selected_diff_bt_ff_bf = diff_bt_ff_bf1 if mean_abs_diff1 < mean_abs_diff2 else diff_bt_ff_bf2
            else:
                selected_longitude = longitude
                selected_latitude = latitude
                selected_diff_bt_ff_bf = diff_bt_ff_bf
            # renew segment
            preprocessed_segments.append((selected_latitude, selected_longitude, selected_diff_bt_ff_bf))

        return preprocessed_segments

    def preprocess_data2(self):
        """将preprocess_data1()预处理的数据分成北半球和南半球2个数据且按照时间顺序，逐元素对应，先北后南。得到的数据可用于绘制沿轨迹磁扰图像
        （每轨）。这个函数可用于绘制MagneticDisturbanceFigure类。"""
        preprocessed_segments = self.preprocess_data1()
        merge_segs = []
        latitude = []
        longitude = []
        diff_bt = []
        for segment in preprocessed_segments:
            latitude.extend(segment[0])
            longitude.extend(segment[1])
            diff_bt.extend(segment[2])
        # 计算相邻元素之间的差的符号，检测变号点
        sign_changes = np.sign(latitude[:-1]) != np.sign(latitude[1:])
        # 寻找变号的索引
        change_indices = np.where(sign_changes)[0] + 1  # +1 是因为变号点在第二个元素开始
        # 使用变号点索引分割数组
        latitude_segs = np.split(latitude, change_indices)
        longitude_segs = np.split(longitude, change_indices)
        diff_bt_segs = np.split(diff_bt, change_indices)
        if len(latitude_segs) > 0:
            if latitude_segs[0][0] < 0:
                latitude_segs.insert(0, np.array([]))
                longitude_segs.insert(0, np.array([]))
                diff_bt_segs.insert(0, np.array([]))
            if latitude_segs[-1][0] > 0:
                latitude_segs.append(np.array([]))
                longitude_segs.append(np.array([]))
                diff_bt_segs.append(np.array([]))
        for lat_seg, lon_seg, diff_bt_seg in zip(latitude_segs, longitude_segs, diff_bt_segs):
            merge_segs.append([lat_seg, lon_seg, diff_bt_seg])
        north_segments = merge_segs[::2]
        south_segments = merge_segs[1::2]
        return north_segments, south_segments


class CDFFile:
    def __init__(self,file_path,is_f17:bool,is_ssies3=False,is_ssm=False):
        """
        :param file_path: absolute path
        """
        self.file_path = file_path
        self.directory = os.path.dirname(self.file_path)
        self.filename = os.path.basename(self.file_path)
        self.cdf = spacepy.pycdf.CDF(self.file_path)
        self.is_f17 = is_f17
        self.is_ssies3 = is_ssies3
        self.is_ssm = is_ssm
        # match = re.search(r"(\d{12})",self.filename)
        # if match:
        #     self.start_time_str = match.group(1)  # 提取匹配的部分
        #     self.start_time_datetime = pd.to_datetime(self.start_time_str)
        # else:
        #     print("cannot find timestamp in filename.")

    def ssies3_data(self, is_drop=False):
        """
        read cdaweb ssies3 data. Don't guarantee can read ssies3 data from other resource.
        all variables data of ssies3 data from cdaweb are one dimension and their lengths are same.
        :param is_drop:
        :return:
        """
        data = pd.DataFrame()
        # get cdf variable names
        var_names = self.cdf.keys()
        # preview the variables. get the data without drop variables or with all variables.
        if is_drop:
            disgard_vars = ['corrpaqual','ebm','rpainfo','nmbot','rpaground','pot','corvelx','corvely','corvelz']
            for var_name in var_names:
                if var_name in disgard_vars:
                    continue
                var = self.cdf[var_name][...]
                data[var_name] = var
        else:
            for var_name in var_names:
                var = self.cdf[var_name][...]
                data[var_name] = var
        # add 'timestamps' column to data, note that the unit of timestamps is second, not n second. The same unit
        # attention is when I want to change unix timestamp to pandas datetime object, the default unit is n second.
        data['timestamps'] = data['Epoch'].astype('int64') // 1e9

        return data

    def ssm_data(self):
        """
        read cdaweb ssies3 data. Don't guarantee can read ssies3 data from other resource.
        ssm data is different from ssies3 data.
        some variable data have 3 dim. the label variable data have 1 dim but their length are 3.
        If I want to get more information,
        :return:
        """
        # remove label variable data, get the name and shape of the rest variable data.
        var_shape = {}
        for var in self.cdf:
            if len(self.cdf[var].shape) == 1 and self.cdf[var].shape[0] == 3:
                continue
            var_shape[var] = self.cdf[var].shape
        # 加载数据
        data = pd.DataFrame()
        for var, shape in var_shape.items():
            # add if statement to resolve the 3 dim variable data to x,y,z coordinate.
            if (len(shape) == 2) and (shape[1] == 3):
                data[f'{var}_x'] = self.cdf[var][...][:, 0]
                data[f'{var}_y'] = self.cdf[var][...][:, 1]
                data[f'{var}_z'] = self.cdf[var][...][:, 2]
            else:
                data[var] = self.cdf[var][...]
        # add 'timestamps' column to data
        data['timestamps'] = data['Epoch'].astype('int64') // 1e9
        return data

    def vx_set_nan(self,is_raw=False) ->pd.DataFrame:
        """
        质量控制和缺失时刻的数据点填充为nan
        """
        # prepare the data needed to be processed.
        data = self.ssies3_data()
        vxqual = data['vxqual']
        if is_raw:
            vx = data['vxraw']
            vx_set_nan = data['vxraw'].copy()
        else:
            vx = data['vx']
            vx_set_nan = data['vx'].copy()
        # find the time of data needed to be set nan.
        # for my fft analysis, I need the complete time series data without nan. So in the end, I need to process these
        # nan.
        # 注意要用括号将条件括起来，不然会报错：
        # ValueError: The truth value of a Series is ambiguous. Use a.empty, a.bool(), a.item(), a.any() or a.all().
        filter = (~(vxqual == 1) | ((vx > 2000) | (vx < -2000)))
        vx_set_nan[filter] = np.nan
        vx_df = pd.DataFrame()
        vx_df['timestamps'] = data['timestamps'].copy()
        vx_df['vx'] = vx_set_nan
        # lost time data set nan
        differences = np.diff(vx_df['timestamps'].to_numpy())
        if np.any(differences != 1):  # note the time resolution
            vx_df = return_full_time_df(vx_df)
        vx_df['Epoch'] = pd.to_datetime(vx_df['timestamps'],unit='s')
        return vx_df

    def v_yz_set_nan(self,v_str) ->pd.DataFrame:
        """
        质量控制和缺失时刻的数据点填充为nan.
        For vy and vz, the quality flag is same idm flag. (I'm not sure that the loss time data is the same for the
        2 variables)
        Note: the quality is level 2 product, before use it in search, I should check the data.
        Reference the ssies3 flag doc.
        """
        data = self.ssies3_data()
        if not all(data['vyqual']==data['vzqual']):
            raise Exception("vy quality not equal vz quality!!! Please check it.")
        vqual = data['vyqual']
        if v_str == 'vy':
            v = data['vy']
            v_set_nan = data['vy'].copy()
        else:
            v = data['vz']
            v_set_nan = data['vz'].copy()
        if self.is_f17:
            filter = ~(((vqual == 1) | (vqual == 6)) & (v < 2000) & (v > -2000))
        else:
            filter = ~((vqual == 1) & (v < 2000) & (v > -2000))
        v_set_nan[filter] = np.nan
        v_df = pd.DataFrame()
        v_df['timestamps'] = data['timestamps'].copy()
        if v_str == 'vy':
            v_df['vy'] = v_set_nan
        else:
            v_df['vz'] = v_set_nan
        # 将数据缺失时刻的值设置为Nan
        differences = np.diff(v_df['timestamps'].to_numpy())
        if np.any(differences != 1):
            v_df = return_full_time_df(v_df)
        # 添加Epoch列
        v_df['Epoch'] = pd.to_datetime(v_df['timestamps'],unit='s')
        return v_df

    def return_vx_parts(self, max_nan_len=60):
        """
        单轨101min，连续nan最大值为101min的1/10，设置为60s，对应max_nan_len=60
        :return: 列表格式，元素也是列表，由连续索引数据组成
        """
        vx_set_nan = self.vx_set_nan()  # index是连续的
        # 找出所有NaN的位置
        is_nan = vx_set_nan.isna()
        # 通过cumsum()创建分组，连续的NaN会有相同的组号
        grouped = is_nan.ne(is_nan.shift()).cumsum()
        # 统计每组NaN的数量
        nan_counts = is_nan.groupby(grouped).sum()
        # 找出连续NaN数量超过max_nan_len的组
        large_nan_groups = nan_counts[nan_counts > max_nan_len].index
        # 标记需要保留的数据
        mask = ~grouped.isin(large_nan_groups) | ~is_nan
        # 应用mask，删除连续NaN超过10的部分
        filtered_series = vx_set_nan[mask]
        # 分割
        # 计算索引的不连续处
        diffs = filtered_series.index.to_series().diff() > 1
        breaks = filtered_series.index[diffs].tolist()  # 获得所有不连续索引的位置
        index0 = filtered_series.index[0]
        index_end = filtered_series.index[-1]
        # 添加起始和结束索引以确保完整分割
        if index0 not in breaks:
            breaks.insert(0, index0)
        if index_end not in breaks:
            breaks.append(index_end + 1)  # 添加最后一个索引加一，以确保包括最后一个元素
        # 根据不连续的索引分割Series
        parts = [filtered_series.loc[breaks[i]:breaks[i + 1] - 1] for i in range(len(breaks) - 1)]
        return parts

    def return_start_end_idx(self):
        """
        for data parts
        :return:
        """
        parts = self.return_vx_parts()
        longest_part = max(parts, key=len)
        start_idx = longest_part.index[0]
        end_idx = longest_part.index[-1]
        return start_idx,end_idx

class MagneticDisturbanceFigure(NcFile):
    """NcFile实例化时传入的默认时s1类型数据"""
    def draw_1d(self):
        """所有轨迹的磁扰,分成南北半球"""
        preprocessed_segments = self.preprocess_data1()
        # 创建 figure 和两个子图，每个子图使用极坐标
        fig = plt.figure(figsize=(12, 6))
        ax_north = fig.add_subplot(121, polar=True)
        ax_south = fig.add_subplot(122, polar=True)

        # 循环处理每个数据段，分别在两个子图中绘制
        for segment in preprocessed_segments:
            # nor
            # 将经度转换为弧度
            longitude_rad = np.radians(segment[1])
            # 将纬度转换为从北极点开始的距离
            latitude_from_pole_north = 90 - segment[0]
            # 北极散点图
            sc_north = ax_north.scatter(longitude_rad, latitude_from_pole_north, c=segment[2], cmap='coolwarm', s=10)

            # sou
            # 将纬度转换为从南极点开始的距离
            latitude_from_pole_south = 90 - (-segment[0])  # 对称的南纬
            # 南极散点图
            sc_south = ax_south.scatter(longitude_rad, latitude_from_pole_south, c=segment[2], cmap='coolwarm', s=10)

        # 为两个子图设置相同的纬度范围和标签
        for ax in [ax_north, ax_south]:
            ax.set_ylim(0, 90)
            ax.set_yticks(range(10, 100, 10))
            degree_sign = u'\N{DEGREE SIGN}'
            ax.set_yticklabels(
                [f"{90 - y}{degree_sign}" if (90 - y in [0, 30, 60]) else '' for y in range(10, 100, 10)])
            ax.set_xticks(np.radians(range(0, 360, 30)))
            ax.set_xticklabels([str(int(x / 30)) for x in range(0, 360, 30)])

        ax_north.set_title('Northern Hemisphere')
        ax_south.set_title('Southern Hemisphere')
        # 添加颜色条
        plt.colorbar(sc_north, ax=[ax_north, ax_south], shrink=0.5, aspect=5)
        # 设置标题
        plt.suptitle(f'Magnetic Field Disturbance {self.date}', x=0.42)
        return fig

    def draw_1d_limit_lat(self, threshold=60):
        """所有轨迹的磁扰，分成南北半球展示，限制纬度范围"""
        preprocessed_segments = self.preprocess_data1()
        # 创建 figure 和两个子图，每个子图使用极坐标
        fig = plt.figure(figsize=(12, 6))
        ax1 = fig.add_subplot(121, polar=True)
        ax2 = fig.add_subplot(122, polar=True)

        # 循环处理每个数据段，分别在两个子图中绘制
        for segment in preprocessed_segments:
            nor_filter = segment[0] > threshold
            sou_filter = segment[0] < -threshold
            # nor
            # 将经度转换为弧度
            longitude_rad = np.radians(segment[1])[nor_filter]
            # 将纬度转换为从北极点开始的距离
            latitude_from_pole_north = 90 - segment[0][nor_filter]
            # 北极散点图
            sc_north = ax1.scatter(longitude_rad, latitude_from_pole_north, c=segment[2][nor_filter], cmap='coolwarm',
                                   s=10)

            # sou
            longitude_rad = np.radians(segment[1])[sou_filter]
            # 将纬度转换为从南极点开始的距离
            latitude_from_pole_south = 90 - (-segment[0])[sou_filter]  # 对称的南纬
            # 南极散点图
            sc_south = ax2.scatter(longitude_rad, latitude_from_pole_south, c=segment[2][sou_filter], cmap='coolwarm',
                                   s=10)

        # 为两个子图设置相同的纬度范围和标签
        for ax in [ax1, ax2]:
            ax.set_ylim(0, 90 - threshold)  # 设置极坐标图的纬度范围
            ax.set_yticks(range(10, 100 - threshold, 10))
            degree_sign = u'\N{DEGREE SIGN}'
            ax.set_yticklabels([f"{90 - y}{degree_sign}" for y in range(10, 100 - threshold, 10)])
            ax.set_xticks(np.radians(range(0, 360, 30)))
            ax.set_xticklabels([str(int(x / 30)) for x in range(0, 360, 30)])

        ax1.set_title('Northern Hemisphere')
        ax2.set_title('Southern Hemisphere')
        # 添加颜色条
        plt.colorbar(sc_north, ax=[ax1, ax2], shrink=0.5, aspect=5)
        plt.suptitle(f'Magnetic Field Disturbance {self.date} in ({threshold},90)', x=0.42)

        return fig

    def draw_orbits(self):
        """按轨绘制磁扰"""
        north_segments, south_segments = self.preprocess_data2()
        north_segments = north_segments
        south_segments = south_segments
        num_segments = len(north_segments)
        # 创建 figure，为每个 segment 创建两个子图（北半球和南半球）
        fig, axes = plt.subplots(num_segments, 2, figsize=(12, 6 * num_segments), subplot_kw={'polar': True})
        i = 0
        for north_segment, south_segment in zip(north_segments, south_segments):
            # nor
            # 将经度转换为弧度
            longitude_rad = np.radians(north_segment[1])
            # 将纬度转换为从极点开始的距离
            latitude_from_pole_north = 90 - north_segment[0]
            # 获取当前行的子图
            ax_north = axes[i][0]
            # 绘制北半球散点图
            sc_north = ax_north.scatter(longitude_rad, latitude_from_pole_north, c=north_segment[2], cmap='coolwarm',
                                        s=10)
            if len(latitude_from_pole_north) > 1:
                # 添加 "start" 和 "end" 标注
                ax_north.annotate('start', (longitude_rad[0], latitude_from_pole_north[0]),
                                  textcoords="offset points", xytext=(-20, -10), ha='center', color='blue')
                ax_north.annotate('end', (longitude_rad[-1], latitude_from_pole_north[-1]),
                                  textcoords="offset points", xytext=(20, 10), ha='center', color='red')

            # sou
            # 将经度转换为弧度
            longitude_rad = np.radians(south_segment[1])
            # 将纬度转换为从极点开始的距离
            latitude_from_pole_south = 90 - (-south_segment[0])
            # 获取当前行的子图
            ax_south = axes[i][1]
            # 绘制南半球散点图
            sc_south = ax_south.scatter(longitude_rad, latitude_from_pole_south, c=south_segment[2], cmap='coolwarm',
                                        s=10)
            if len(latitude_from_pole_south) > 1:
                # 添加 "start" 和 "end" 标注
                ax_south.annotate('start', (longitude_rad[0], latitude_from_pole_south[0]),
                                  textcoords="offset points", xytext=(-20, -10), ha='center', color='blue')
                ax_south.annotate('end', (longitude_rad[-1], latitude_from_pole_south[-1]),
                                  textcoords="offset points", xytext=(20, 10), ha='center', color='red')

            # 设置子图的标题
            if i == 0:
                ax_north.set_title('Northern Hemisphere')
                ax_south.set_title('Southern Hemisphere')

            # 设置纬度范围和标签
            for ax in [ax_north, ax_south]:
                ax.set_ylim(0, 90)
                ax.set_yticks(range(10, 100, 10))
                degree_sign = u'\N{DEGREE SIGN}'
                ax.set_yticklabels(
                    [f"{90 - y}{degree_sign}" if (90 - y in [0, 30, 60]) else '' for y in range(10, 100, 10)])
                ax.set_xticks(np.radians(range(0, 360, 30)))
                ax.set_xticklabels([str(int(x / 30)) for x in range(0, 360, 30)])
            i += 1
        # 添加水平放置的颜色条在图的顶部
        plt.colorbar(sc_north, ax=axes.ravel().tolist(), shrink=0.5, aspect=5, orientation='horizontal', location='top')
        plt.suptitle(f'Magnetic Field Disturbance {self.date} per orbit', y=0.8)
        return fig

    def draw_orbits_limit_lat(self, threshold=60):
        """按轨绘制磁扰，限制纬度范围"""
        north_segments, south_segments = self.preprocess_data2()
        north_segments = north_segments
        south_segments = south_segments
        num_segments = len(north_segments)
        # 创建 figure，为每个 segment 创建两个子图（北半球和南半球）
        fig, axes = plt.subplots(num_segments, 2, figsize=(12, 6 * num_segments), subplot_kw={'polar': True})
        i = 0
        for north_segment, south_segment in zip(north_segments, south_segments):
            nor_filter = north_segment[0] > threshold
            sou_filter = south_segment[0] < -threshold
            # nor
            # 将经度转换为弧度
            longitude_rad = np.radians(north_segment[1])[nor_filter]
            # 将纬度转换为从极点开始的距离
            latitude_from_pole_north = 90 - north_segment[0][nor_filter]
            # 获取当前行的子图
            ax_north = axes[i][0]
            # 绘制北半球散点图
            sc_north = ax_north.scatter(longitude_rad, latitude_from_pole_north, c=north_segment[2][nor_filter],
                                        cmap='coolwarm', s=10)
            if len(latitude_from_pole_north) > 1:
                # 添加 "start" 和 "end" 标注
                ax_north.annotate('start', (longitude_rad[0], latitude_from_pole_north[0]),
                                  textcoords="offset points", xytext=(-20, -10), ha='center', color='blue')
                ax_north.annotate('end', (longitude_rad[-1], latitude_from_pole_north[-1]),
                                  textcoords="offset points", xytext=(20, 10), ha='center', color='red')

            # sou
            # 将经度转换为弧度
            longitude_rad = np.radians(south_segment[1])[sou_filter]
            # 将纬度转换为从极点开始的距离
            latitude_from_pole_south = 90 - (-south_segment[0])[sou_filter]
            # 获取当前行的子图
            ax_south = axes[i][1]
            # 绘制南半球散点图
            sc_south = ax_south.scatter(longitude_rad, latitude_from_pole_south, c=south_segment[2][sou_filter],
                                        cmap='coolwarm', s=10)
            if len(latitude_from_pole_south) > 1:
                # 添加 "start" 和 "end" 标注
                ax_south.annotate('start', (longitude_rad[0], latitude_from_pole_south[0]),
                                  textcoords="offset points", xytext=(-20, -10), ha='center', color='blue')
                ax_south.annotate('end', (longitude_rad[-1], latitude_from_pole_south[-1]),
                                  textcoords="offset points", xytext=(20, 10), ha='center', color='red')

            # 设置子图的标题
            if i == 0:
                ax_north.set_title('Northern Hemisphere')
                ax_south.set_title('Southern Hemisphere')

            # 设置纬度范围和标签
            for ax in [ax_north, ax_south]:
                ax.set_ylim(0, 90 - threshold)  # 设置极坐标图的纬度范围
                ax.set_yticks(range(10, 100 - threshold, 10))
                degree_sign = u'\N{DEGREE SIGN}'
                ax.set_yticklabels([f"{90 - y}{degree_sign}" for y in range(10, 100 - threshold, 10)])
                ax.set_xticks(np.radians(range(0, 360, 30)))
                ax.set_xticklabels([str(int(x / 30)) for x in range(0, 360, 30)])
            i += 1
        # 添加水平放置的颜色条在图的顶部
        plt.colorbar(sc_north, ax=axes.ravel().tolist(), shrink=0.5, aspect=5, orientation='horizontal', location='top')
        plt.suptitle(f'Magnetic Field Disturbance {self.date} per orbit in ({threshold},90)', y=0.8)
        return fig

    def save_1d(self):
        fig_save_path = f'G:/0_postgraduate/DMSP/fig/1d/{self.year}/'
        # 确保保存图像的路径存在，如果不存在则创建
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        fig = self.draw_1d()
        # 构建保存图像的完整路径
        save_fig_path = os.path.join(fig_save_path, f'magnetic_disturbance_1d_{self.date}.png')
        # 保存图像
        fig.savefig(save_fig_path)
        # 关闭图像以释放内存
        plt.close(fig)
        print(f"magnetic_disturbance_1d_{self.date}.png 已保存。")

    def save_1d_limit_lat(self):
        fig_save_path = f'G:/0_postgraduate/DMSP/fig/1d_limit_lat/{self.year}/'
        # 确保保存图像的路径存在，如果不存在则创建
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        fig = self.draw_1d_limit_lat()
        # 构建保存图像的完整路径
        save_fig_path = os.path.join(fig_save_path, f'magnetic_disturbance_1d_limit_lat_{self.date}.png')
        # 保存图像
        fig.savefig(save_fig_path)
        # 关闭图像以释放内存
        plt.close(fig)
        print(f"magnetic_disturbance_1d_limit_lat_{self.date}.png 已保存。")

    def save_orbits(self):
        fig_save_path = f'G:/0_postgraduate/DMSP/fig/orbits/{self.year}/'
        # 确保保存图像的路径存在，如果不存在则创建
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        fig = self.draw_orbits()
        # 构建保存图像的完整路径
        save_fig_path = os.path.join(fig_save_path, f'magnetic_disturbance_orbits_{self.date}.png')
        # 保存图像
        fig.savefig(save_fig_path)
        # 关闭图像以释放内存
        plt.close(fig)
        print(f'magnetic_disturbance_orbits_{self.date}.png 已保存')

    def save_orbits_limit_lat(self):
        fig_save_path = f'G:/0_postgraduate/DMSP/fig/orbits_limit_lat/{self.year}/'
        # 确保保存图像的路径存在，如果不存在则创建
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        fig = self.draw_orbits_limit_lat()
        # 构建保存图像的完整路径
        save_fig_path = os.path.join(fig_save_path, f'magnetic_disturbance_orbits_limit_lat_{self.date}.png')
        # 保存图像
        fig.savefig(save_fig_path)
        # 关闭图像以释放内存
        plt.close(fig)
        print(f'magnetic_disturbance_orbits_limit_lat_{self.date}.png 已保存')

    # def batch_save(self):
    #     # 2011年
    #     nc_files_path = 'G:/0_postgraduate/DMSP/data/2011/15s1/'
    #     for nc_file in glob.glob(nc_files_path + '*.nc')[:50]:
    #         file_path = os.path.join(nc_files_path,nc_file)
    #         disturbance = MagneticFieldDisturbance(file_path)
    #         disturbance.save_1d()
    #         disturbance.save_orbits()
    #         disturbance.save_orbits_full()


class BandPassFilter:
    def __init__(self,variable_data,time_x:datetime,start_time_for_title: datetime | str):
        self.variable_data = variable_data
        self.time_x = time_x
        self.t = self.variable_data.index.to_numpy()
        self.signal = self.variable_data.values
        self.signal_fft = fft(self.signal)
        self.sample_freq = fftfreq(self.signal.size, d=self.t[1] - self.t[0])
        self.start_time_for_title = start_time_for_title
        # 将频率和振幅数组按频率升序排列
        indices = np.argsort(self.sample_freq)  # 获取排序后的索引数组
        sorted_freqs = self.sample_freq[indices]
        sorted_fft = self.signal_fft[indices]
        # 取绝对值计算振幅（模长）
        sorted_magnitudes = np.abs(sorted_fft)
        # 过滤出正频率部分用于绘制（通常我们只关心正频率部分）
        positive_freqs = sorted_freqs > 0
        self.final_freqs = sorted_freqs[positive_freqs]
        self.final_magnitudes = sorted_magnitudes[positive_freqs]


    def draw_original_signal(self):
        # 绘制原始信号
        fig,ax = plt.subplots(figsize=(20, 5))
        ax.plot(self.time_x,self.signal)
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0,15,30,45]))  # 只在某些时刻显示刻度
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 格式化显示格式
        ax.set_xlabel("time (s)")
        ax.set_ylabel(f"{self.variable_data.name}")
        ax.set_title(f'original {self.variable_data.name} start at {self.start_time_for_title}')
        # plt.xlabel("time (s)")
        # plt.ylabel(f"{self.variable_data.name}")
        # plt.title(f'original {self.variable_data.name}')
        return fig
    def draw_frequency_amplitude(self,is_log=False):
        # 绘制傅里叶变换结果的幅度谱（双边谱）
        fig = plt.figure(figsize=(20, 5))
        if is_log:
            final_magnitudes_log = np.log(self.final_magnitudes)
            plt.plot(self.final_freqs, final_magnitudes_log)
            plt.ylabel('Amplitude (lg)')
        else:
            plt.plot(self.final_freqs, self.final_magnitudes)
            plt.ylabel('Amplitude')
        plt.xlabel('Frequency (Hz)')
        plt.title(f'Spectrogram after Fourier Transform start at {self.start_time_for_title}')
        return fig

    def signal_after_bandpass_filter(self, epoch1, epoch2, is_draw=False):
        # 选取不同频率段，创建一个带通滤波器
        bandpass_filter = (self.sample_freq > 1/epoch2) & (self.sample_freq < 1/epoch1)
        filtered_signal_fft = self.signal_fft.copy()
        filtered_signal_fft[~bandpass_filter] = 0
        # 进行傅里叶反变换，得到过滤后的信号
        filtered_signal = np.real(np.fft.ifft(filtered_signal_fft))
        if is_draw:
            # 绘制过滤后的信号
            fig,ax = plt.subplots(figsize=(20, 5))
            ax.plot(self.time_x, filtered_signal)
            ax.set_xlabel('Time (seconds)')
            ax.set_ylabel(f'{self.variable_data.name}')
            ax.set_title(f'{self.variable_data.name} after bandpass filter ({epoch1}s to {epoch2}s) '
                         f'start at {self.start_time_for_title}')
            return fig, filtered_signal
        return filtered_signal


def return_suffix_file_paths(directory, suffix='.cdf'):
    """ 返回指定目录下所有 assigned suffix 文件的完整路径列表。 directory为绝对路径"""
    file_paths = []
    # 遍历指定目录
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(suffix):
                # 拼接完整的文件路径
                full_path = os.path.join(root, file)
                file_paths.append(full_path)
    return file_paths


def walen_test_s1_sseis3(data_s1:pd.DataFrame, data_ssies3:pd.DataFrame,
                         start_idx:int, end_idx:int, vx_set_nan, freq='S'):
    """
    for ssies2 data from Madrigal database named "ut...".
    :param data_s1:
    :param data_ssies3:
    :param start_idx:
    :param end_idx:
    :param vx_set_nan:
    :param freq:
    :return:
    """
    data_ssies3_clipped = pd.DataFrame()
    timestamps_ssies3 = data_ssies3['timestamps'].copy()
    timestamps_ssies3_clipped = timestamps_ssies3.iloc[start_idx:end_idx].reset_index(drop=True)
    start_timestamp = timestamps_ssies3_clipped.iloc[0]
    end_timestamp = timestamps_ssies3_clipped.iloc[-1]
    vx_ssies3_clipped = vx_set_nan.iloc[start_idx:end_idx].reset_index(drop=True)
    data_ssies3_clipped['timestamps'] = timestamps_ssies3_clipped
    data_ssies3_clipped['vx'] = vx_ssies3_clipped  # 还没处理nan
    differences = np.diff(timestamps_ssies3_clipped.to_numpy())
    if np.any(differences != 1):
        data_ssies3_clipped = return_full_time_df(data_ssies3_clipped)
    # # 将Unix时间戳转换为datetime
    # data_ssies3['timestamps'] = pd.to_datetime(data_ssies3['timestamps'], unit='s')
    # # 创建完整的时间序列
    # min_time = data_ssies3['timestamps'].min()
    # max_time = data_ssies3['timestamps'].max()
    # all_times = pd.date_range(start=min_time, end=max_time, freq='S')
    # full_df = pd.DataFrame(all_times, columns=['timestamps'])
    # full_df = full_df.merge(data_ssies3, on='timestamps', how='left')
    # full_df['Epoch'] = full_df['timestamps'].astype('int64') // 1e9
    # full_df.columns = ['Epoch','vx','timestamps']

    timestamps_s1 = data_s1['timestamps']
    start_idx = np.where(timestamps_s1 == start_timestamp)[0][0]
    end_idx = np.where(timestamps_s1 == end_timestamp)[0][0]
    data_s1_clipped = data_s1.iloc[start_idx:end_idx+1].reset_index(drop=True)
    data_s1_select_v = pd.DataFrame()
    data_s1_select_v['Epoch'] = data_s1_clipped['Epoch']
    data_s1_select_v['timestamps'] = data_s1_clipped['timestamps']
    data_s1_select_v['diff_bx'] = data_s1_clipped['diff_b_for'] * 1e9
    # data_s1_select_v['diff_by'] = data_s1_clipped['diff_b_perp'] * 1e9
    timestamps_s1_clipped = data_s1_select_v['timestamps']
    differences = np.diff(timestamps_s1_clipped.to_numpy())
    if np.any(differences!=1):
        data_s1_select_v = return_full_time_df(data_s1_select_v)
    # 合并
    if np.all(data_ssies3_clipped['timestamps']==data_s1_select_v['timestamps']):
        data_for_walen = pd.concat([data_ssies3_clipped[['Epoch','timestamps','vx']],data_s1_select_v['diff_bx']],axis=1)

    return data_for_walen


def return_full_time_df(df):
    """时间戳列名为timestamps，单位为s（分辨率为1s）"""
    # 将Unix时间戳转换为datetime
    df['timestamps'] = pd.to_datetime(df['timestamps'], unit='s')
    # 创建完整的时间序列
    min_time = df['timestamps'].min()
    max_time = df['timestamps'].max()
    all_times = pd.date_range(start=min_time, end=max_time, freq='S')
    full_df = pd.DataFrame(all_times, columns=['timestamps'])
    full_df = full_df.merge(df, on='timestamps', how='left')
    # 将datetime转换回Unix时间戳（秒）
    full_df['timestamps'] = full_df['timestamps'].astype(np.int64) // 10 ** 9
    full_df['Epoch'] = pd.to_datetime(full_df['timestamps'],unit='s')
    return full_df


def read_csv(file_path,comment='#'):
    # 步骤1: 读取 CSV 文件
    data = pd.read_csv(file_path, comment=comment)
    # 如果描述性文本可能被误认为是实际数据，或者文件中有用特定字符（如#）开头的注释行，你可以使用 comment 参数指定这个字符。Pandas 在读取数据时会忽略任何以这个字符开始的行。
    # 步骤2: 检查数据的前几行
    print(data.head())
    # 步骤3: 检查数据的基本信息（如数据类型、非空值数量等）
    print(data.info())
    return data


def read_hdf5(file_path):
    with h5py.File(file_path, 'r') as file:
        # 列出文件中的主要组和数据集
        print("Keys: %s" % file.keys())
        for key in file.keys():
            print(key)
        # # 假设我们知道要访问的数据集的名称是 'dataset'
        # if 'dataset' in file:
        #     data = file['dataset'][:]
        #     print(data)
        #
        # # 如果数据集包含在一个组内，我们可以这样访问
        # if 'group' in file and 'dataset' in file['group']:
        #     data = file['group']['dataset'][:]
        #     print(data)
        #
        # # 读取属性
        # if 'dataset' in file:
        #     print("Attributes of dataset:")
        #     for attr_name in file['dataset'].attrs.keys():
        #         attr_value = file['dataset'].attrs[attr_name]
        #         print(attr_name, attr_value)
        return


def read_hdf4(file_path):
    return


def data_for_draw(ssies3_fp,ssm_fp,is_f17):
    """
    prepare the data for draw.
    note: for ssies3, I need the velocity. If I need other variables, I need to refactor.
    :param ssies3_fp:
    :param ssm_fp:
    :return:
    """
    ssies3 = CDFFile(ssies3_fp,is_f17=is_f17)
    ssm = CDFFile(ssm_fp,is_f17=is_f17)
    # data
    ssies3_data = ssies3.ssies3_data()
    ssm_data = ssm.ssm_data()
    # v set nan
    vx_set_nan = ssies3.vx_set_nan()
    vy_set_nan = ssies3.v_yz_set_nan(v_str='vy')
    vz_set_nan = ssies3.v_yz_set_nan(v_str='vz')
    # change the coordinate system of v
    data_dic = {'Epoch': vx_set_nan['Epoch'], 'V_SC_x': -vz_set_nan['vz'], 'V_SC_y': vx_set_nan['vx'],
                'V_SC_z': -vy_set_nan['vy']}
    ssies3_data_SC = pd.DataFrame(data_dic)
    # ssm clip
    st = ssies3_data['timestamps'].iloc[0]
    et = ssies3_data['timestamps'].iloc[-1]
    s_idx = np.where(ssm_data['timestamps'] == st)
    e_idx = np.where(ssm_data['timestamps'] == et)
    ssm_data_clip = ssm_data.iloc[s_idx[0][0]:e_idx[0][0] + 1].reset_index(drop=True)

    return ssies3_data_SC, ssm_data_clip


def draw_ssies3_ssm(ssies3_data:pd.DataFrame, ssm_data:pd.DataFrame, ssies3_v_str:str, ssm_v_str:str, ssies3_unit:str,
                    ssm_unit:str, dhm:str, title_part:str,satellite:str, is_save=False,
                    fig_save_path:Optional[str]=None):
    """
    draw one orbit, different physical parameters, for example, plasma velocity and delta magnetic field or observed
    magnetic field and so on.
    the same coordinate system (for example: SC, GEO...).
    :param ssm_unit: the ssm physical parameter's unit
    :param ssies3_unit: the ssies3 physical parameter's unit
    :param ssm_v_str: the ssm physical parameter
    :param ssies3_v_str: the ssies3 physical parameter
    :param ssies3_data: the ssies3 data in the afssigned coordinate system, including 'Epoch',xyz components.
    :param ssm_data: the clipped ssm data, including 'Epoch',xyz components.
    :param dhm: the data start day hour minute.
    :param title_part:
    :return:
    """
    fig, axs = plt.subplots(3, 1, figsize=(20, 10 * 3))
    ax1 = axs[0]
    ax2 = axs[1]
    ax3 = axs[2]

    # fig title
    fig.suptitle(f'f{satellite} {dhm}: {title_part}')

    # x
    ax1_t = ax1.twinx()  # 创建共享x轴的第二个y轴
    # plot
    ax1.plot(ssies3_data['Epoch'], ssies3_data[f'{ssies3_v_str}_x'], 'tab:blue')
    ax1_t.plot(ssm_data['Epoch'], ssm_data[f'{ssm_v_str}_x'], 'tab:orange')
    # datetime locator
    # because ax1_t share the ax1, so the locator just used in ax1.
    ax1.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 15, 30, 45]))  # 只在某些时刻显示刻度
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 格式化显示格式
    # 在y=0处添加一条红色虚线
    ax1.axhline(y=0, color='black', linestyle='--', label='y = 0')
    # scale
    max_y1 = max(abs(ssies3_data[f'{ssies3_v_str}_x'].min()), ssies3_data[f'{ssies3_v_str}_x'].max())
    max_y2 = max(abs(ssm_data[f'{ssm_v_str}_x'].min()), ssm_data[f'{ssm_v_str}_x'].max())
    # 设定缩放比例，根据两侧数据的最大值决定
    scale_factor = max_y1 / max_y2
    max_y2_scaled = max_y2 * scale_factor
    common_limit = max(max_y1, max_y2_scaled) * 1.1
    ax1.set_ylim(-common_limit, common_limit)
    ax1_t.set_ylim(-common_limit / scale_factor, common_limit / scale_factor)  # 使用缩放比例调整
    # y label
    ax1.set_ylabel(f'{ssies3_v_str}_x ({ssies3_unit})', color='tab:blue')
    ax1_t.set_ylabel(f'{ssm_v_str}_x ({ssm_unit})', color='tab:orange')
    # title
    ax1.set_title('x direction')

    # y
    ax2_t = ax2.twinx()
    # plot
    ax2.plot(ssies3_data['Epoch'], ssies3_data[f'{ssies3_v_str}_y'], 'tab:blue')
    ax2_t.plot(ssm_data['Epoch'], ssm_data[f'{ssm_v_str}_y'], 'tab:orange')
    # datetime locator
    ax2.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 25, 30, 45]))  # 只在某些时刻显示刻度
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 格式化显示格式
    # 在y=0处添加一条红色虚线
    ax2.axhline(y=0, color='black', linestyle='--', label='y = 0')
    # scale
    max_y1 = max(abs(ssies3_data[f'{ssies3_v_str}_y'].min()), ssies3_data[f'{ssies3_v_str}_y'].max())
    max_y2 = max(abs(ssm_data[f'{ssm_v_str}_y'].min()), ssm_data[f'{ssm_v_str}_y'].max())
    # 设定缩放比例，根据两侧数据的最大值决定
    scale_factor = max_y1 / max_y2
    max_y2_scaled = max_y2 * scale_factor
    common_limit = max(max_y1, max_y2_scaled) * 1.1
    ax2.set_ylim(-common_limit, common_limit)
    ax2_t.set_ylim(-common_limit / scale_factor, common_limit / scale_factor)  # 使用缩放比例调整
    # y label
    ax2.set_ylabel(f'{ssies3_v_str}_y ({ssies3_unit})', color='tab:blue')
    ax2_t.set_ylabel(f'{ssm_v_str}_y ({ssm_unit})', color='tab:orange')
    # title
    ax2.set_title('y direction')

    # z
    ax3_t = ax3.twinx()
    # plot
    ax3.plot(ssies3_data['Epoch'], ssies3_data[f'{ssies3_v_str}_z'], 'tab:blue')
    ax3_t.plot(ssm_data['Epoch'], ssm_data[f'{ssm_v_str}_z'], 'tab:orange')
    # datetime locator
    ax3.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 25, 30, 45]))  # 只在某些时刻显示刻度
    ax3.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))  # 格式化显示格式
    # 在y=0处添加一条红色虚线
    ax3.axhline(y=0, color='black', linestyle='--', label='y = 0')
    # scale
    max_y1 = max(abs(ssies3_data[f'{ssies3_v_str}_z'].min()), ssies3_data[f'{ssies3_v_str}_z'].max())
    max_y2 = max(abs(ssm_data[f'{ssm_v_str}_z'].min()), ssm_data[f'{ssm_v_str}_z'].max())
    # 设定缩放比例，根据两侧数据的最大值决定
    scale_factor = max_y1 / max_y2
    max_y2_scaled = max_y2 * scale_factor
    common_limit = max(max_y1, max_y2_scaled) * 1.1
    ax3.set_ylim(-common_limit, common_limit)
    ax3_t.set_ylim(-common_limit / scale_factor, common_limit / scale_factor)  # 使用缩放比例调整
    # y label
    ax3.set_ylabel(f'{ssies3_v_str}_z ({ssies3_unit})', color='tab:blue')
    ax3_t.set_ylabel(f'{ssm_v_str}_z ({ssm_unit})', color='tab:orange')
    # title
    ax3.set_title('z direction')

    if is_save:
        # save
        save_fig_path = os.path.join(fig_save_path, f"f{satellite}_{dhm}_{title_part}.png")
        plt.savefig(save_fig_path)
        print(f"save f{satellite}_{dhm}_{title_part}.png")
    return fig


### code back
# pd.datetime 转换为 unix timestamps (s)
# data_ssies3['Epoch'].astype('int64') // 1e9
# unix timestamps (s) 转换为 pd.datetime
# pd.to_datetime(timestamps)
###