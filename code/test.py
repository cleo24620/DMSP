import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import glob
from netCDF4 import Dataset
import time
import re

class MagneticFieldDisturbance:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data_details = self.read_nc_details()
        self.variables = self.data_details.keys()
        self.data = self.get_data()

    def read_nc_details(self):
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

    def get_data(self):
        """return data as a DataFrame"""
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

    def preprocess_data(self):
        """处理经度180度和-180度的突变；去掉经度的变化率非常大的部分。最后得到分割好的纬度、经度、磁扰。"""
        latitude = self.data['gdlat']
        longitude = self.data['glon']
        # 磁场数据（T转nT）
        diff_bt = np.sqrt(
            self.data['diff_b_for'] ** 2 + self.data['diff_b_perp'] ** 2 + self.data['diff_bd'] ** 2) * 1e9
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

    def nor_south_segments(self):
        """将预处理的数据分成北半球和南半球2个数据且按照时间顺序，逐元素对应，先北后南。得到的数据可用于绘制沿轨迹磁扰图像（每轨）。"""
        preprocessed_segments = self.preprocess_data()
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

    def draw_1d(self):
        """所有轨迹的磁扰,分成南北半球"""
        preprocessed_segments = self.preprocess_data()
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

        return fig

    def draw_1d_limit_lat(self, threshold=60):
        """所有轨迹的磁扰，分成南北半球展示，限制纬度范围"""
        preprocessed_segments = self.preprocess_data()
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
            ax.set_ylim(0, 100 - threshold)  # 设置极坐标图的纬度范围
            ax.set_yticks(range(10, 90 - threshold, 10))
            degree_sign = u'\N{DEGREE SIGN}'
            ax.set_yticklabels([f"{90 - y}{degree_sign}" for y in range(10, 90 - threshold, 10)])
            ax.set_xticks(np.radians(range(0, 360, 30)))
            ax.set_xticklabels([str(int(x / 30)) for x in range(0, 360, 30)])

        ax1.set_title('Northern Hemisphere')
        ax2.set_title('Southern Hemisphere')
        # 添加颜色条
        plt.colorbar(sc_north, ax=[ax1, ax2], shrink=0.5, aspect=5)

        return fig

    def draw_orbits(self):
        """按轨绘制磁扰"""
        north_segments, south_segments = self.nor_south_segments()
        north_segments = north_segments[:5]
        south_segments = south_segments[:5]
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
        return fig

    def draw_orbits_limit_lat(self, threshold=60):
        """按轨绘制磁扰，限制纬度范围"""
        north_segments, south_segments = self.nor_south_segments()
        north_segments = north_segments[:5]
        south_segments = south_segments[:5]
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
        plt.suptitle(f'Magnetic Field Disturbance {date}')
        return fig

    def save_1d(self):
        # 定义文件路径
        # 使用 os.path.dirname 获取目录路径
        directory = os.path.dirname(self.file_path)
        # 使用 os.path.basename 获取文件名
        filename = os.path.basename(self.file_path)
        # 正则表达式匹配四位数字年份
        match = re.search(r'\b\d{4}\b', directory)
        if match:
            year = match.group(0)
        else:
            print("No year found in the path.")
        fig_save_path = f'G:/0_postgraduate/DMSP/fig/1d/{year}/'
        # 确保保存图像的路径存在，如果不存在则创建
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        # 提取文件名中的日期
        date = os.path.basename(filename).split('_')[1]
        fig = self.draw_1d_limit_lat()
        # 设置标题
        plt.suptitle(f'Magnetic Field Disturbance {date}')
        # 构建保存图像的完整路径
        save_fig_path = os.path.join(fig_save_path, f'magnetic_disturbance_1d_{date}.png')
        # 保存图像
        fig.savefig(save_fig_path, dpi=300)
        # 关闭图像以释放内存
        plt.close(fig)
        print(f"magnetic_disturbance_1d_{date}.png 已保存。")

    def save_orbits(self):
        # 定义文件路径
        # 使用 os.path.dirname 获取目录路径
        directory = os.path.dirname(self.file_path)
        # 使用 os.path.basename 获取文件名
        filename = os.path.basename(self.file_path)
        # 正则表达式匹配四位数字年份
        match = re.search(r'\b\d{4}\b', directory)
        if match:
            year = match.group(0)
        else:
            print("No year found in the path.")
        fig_save_path = f'G:/0_postgraduate/DMSP/fig/orbits/{year}/'
        # 确保保存图像的路径存在，如果不存在则创建
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        # 提取文件名中的日期
        date = os.path.basename(filename).split('_')[1]
        fig = self.draw_orbits_limit_lat()
        # 设置标题
        plt.suptitle(f'Magnetic Field Disturbance per orbit {date}')
        # 构建保存图像的完整路径
        save_fig_path = os.path.join(fig_save_path, f'magnetic_disturbance_orbits_{date}.png')
        # 保存图像
        fig.savefig(save_fig_path)
        # 关闭图像以释放内存
        plt.close(fig)
        print(f'magnetic_disturbance_orbits_{date}.png 已保存')

    def save_orbits_full(self):
        # 定义文件路径
        # 使用 os.path.dirname 获取目录路径
        directory = os.path.dirname(self.file_path)
        # 使用 os.path.basename 获取文件名
        filename = os.path.basename(self.file_path)
        # 正则表达式匹配四位数字年份
        match = re.search(r'\b\d{4}\b', directory)
        if match:
            year = match.group(0)
        else:
            print("No year found in the path.")
        fig_save_path = f'G:/0_postgraduate/DMSP/fig/orbits_full/{year}/'
        # 确保保存图像的路径存在，如果不存在则创建
        if not os.path.exists(fig_save_path):
            os.makedirs(fig_save_path)
        # 提取文件名中的日期
        date = os.path.basename(filename).split('_')[1]
        fig = self.draw_orbits()
        # 设置标题
        plt.suptitle(f'Magnetic Field Disturbance per orbit {date}')
        # 构建保存图像的完整路径
        save_fig_path = os.path.join(fig_save_path, f'magnetic_disturbance_orbits_full_{date}.png')
        # 保存图像
        fig.savefig(save_fig_path)
        # 关闭图像以释放内存
        plt.close(fig)
        print(f'magnetic_disturbance_orbits_full_{date}.png 已保存')

    # def batch_save(self):
    #     # 2011年
    #     nc_files_path = 'G:/0_postgraduate/DMSP/data/2011/15s1/'
    #     for nc_file in glob.glob(nc_files_path + '*.nc')[:50]:
    #         file_path = os.path.join(nc_files_path,nc_file)
    #         disturbance = MagneticFieldDisturbance(file_path)
    #         disturbance.save_1d()
    #         disturbance.save_orbits()
    #         disturbance.save_orbits_full()
