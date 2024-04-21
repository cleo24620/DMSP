# -*- coding: utf-8 -*-
# @Author  : cleo
# @Software: PyCharm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from netCDF4 import Dataset
import time
import re
from scipy.fft import fft, fftfreq


class File:
    def __init__(self,file_path):
        self.file_path = file_path
        self.data_details = self.data_details()
        self.variables = self.data_details.keys()
        self.original_data = self.original_data()
        self.directory = os.path.dirname(self.file_path)
        self.filename = os.path.basename(self.file_path)
        # 正则表达式匹配四位数字年份
        match = re.search(r'\b\d{4}\b', self.directory)
        if match:
            self.year = match.group(0)
        else:
            print("No year found in the path.")
        # 提取文件名中的日期
        self.date = os.path.basename(self.filename).split('_')[1]

    def data_details(self):
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

    def original_data(self):
        """返回原始数据。同时通过时间戳获取时间字符串，并将其作为DataFrame的第1列"""
        nc_obj = Dataset(self.file_path)
        data = pd.DataFrame()
        # col name is the v_name
        for v in self.variables:
            data[v] = nc_obj.variables[v][:]
        # add time_str col, i.e. turning unix time to str time. And the time is UTC.
        timestamps = data['timestamps']
        time_str = [time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime(t)) for t in timestamps]
        data.insert(0, 'time', time_str)
        return data

    def preprocess_data1(self):
        """处理经度180度和-180度的突变；去掉经度的变化率非常大的部分。最后得到分割好的纬度、经度、磁扰。"""
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
        """将preprocess_data1()预处理的数据分成北半球和南半球2个数据且按照时间顺序，逐元素对应，先北后南。得到的数据可用于绘制沿轨迹磁扰图像（每轨）。"""
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


class Disturbance(File):
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


class Transform(File):
    def original_signal(self, variable, is_b=False, fillna=True,is_draw=False):
        variable_data = self.original_data[variable]
        if is_b:
            variable_data = variable_data * 1e9
        if fillna:
            variable_data = variable_data.fillna(method='ffill').fillna(method='bfill')
        t = variable_data.index.to_numpy()
        signal = variable_data.values
        if is_draw:
            # 绘制原始信号
            plt.figure(figsize=(20, 5))
            plt.plot(t, signal)
            plt.title('Original signal')
        return t,signal

    def frequency_amplitude(self, variable, is_b=False, fillna=True, is_draw=False):
        variable_data = self.original_data[variable]
        if is_b:
            variable_data = variable_data * 1e9
        if fillna:
            variable_data = variable_data.fillna(method='ffill').fillna(method='bfill')
        t = variable_data.index.to_numpy()
        signal = variable_data.values
        # 进行傅里叶变换
        signal_fft = fft(signal)
        # 计算对应的频率
        sample_freq = fftfreq(signal.size, d=t[1] - t[0])
        if is_draw:
            # 绘制傅里叶变换结果的幅度谱（双边谱）
            plt.figure(figsize=(20, 5))
            plt.plot(sample_freq, np.abs(signal_fft))
            plt.title('Spectrogram after Fourier Transform')
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')
        return sample_freq,signal_fft

    def signal_after_bandpass_filter(self, variable, epoch1, epoch2, is_b=False, fillna=True, is_draw=False):
        variable_data = self.original_data[variable]
        if is_b:
            variable_data = variable_data * 1e9
        if fillna:
            variable_data = variable_data.fillna(method='ffill').fillna(method='bfill')
        t = variable_data.index.to_numpy()
        signal = variable_data.values
        # 进行傅里叶变换
        signal_fft = fft(signal)
        # 计算对应的频率
        sample_freq = fftfreq(signal.size, d=t[1] - t[0])
        # 选取不同频率段，创建一个带通滤波器
        bandpass_filter = (sample_freq > 1/epoch2) & (sample_freq < 1/epoch1)
        filtered_signal_fft = signal_fft.copy()
        filtered_signal_fft[~bandpass_filter] = 0
        # 进行傅里叶反变换，得到过滤后的信号
        filtered_signal = np.real(np.fft.ifft(filtered_signal_fft))
        if is_draw:
            # 绘制过滤后的信号
            plt.figure(figsize=(20, 5))
            plt.plot(t, filtered_signal)
            plt.title(f'Signal after bandpass filter ({epoch1}s to {epoch2}s)')
            plt.xlabel('Time (seconds)')
            plt.ylabel(f'{variable}')
            plt.show()
        return t,filtered_signal

# Transform类的用法
# file_path = r"G:\0_postgraduate\DMSP\data\2011\15s1\dms_20110101_15s1.001.nc"
# transform = Transform(file_path)
# t,filtered_signal = transform.signal_after_bandpass_filter('vert_ion_v',is_b=False,epoch1=1000,epoch2=1100,is_draw=True)