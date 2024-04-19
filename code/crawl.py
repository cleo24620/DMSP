# -*- coding: utf-8 -*-
# @Author  : cleo
# @Software: PyCharm
# 从madrigal数据库下载dmsp数据
import os

import requests
from bs4 import BeautifulSoup


url_prefix = "http://madrigal.iggcas.ac.cn"  # 这个加上link构成可点击的web链接（链接可以是网页也可以是文件）
year_url = "http://madrigal.iggcas.ac.cn/ftp/fullname/leo/email/1392787871@qq.com/affiliation/None/kinst/8100/"  # 从年份页面开始选择


def get_links(url,tag1='p',tag2='a'):
    """对于madrigal数据库的dmsp数据而言，所有的链接存在于网页中的<p>标签内的<a>标签。"""
    response = requests.get(url)
    if response.status_code != 200:
        print("请求失败，状态码：" + str(response.status_code))
    soup = BeautifulSoup(response.text, 'html.parser')
    tag1s = soup.find_all(tag1)
    links = []
    for tag in tag1s:
        tag2s = tag.find_all(tag2, href=True)  # 确保链接存在
        for tag in tag2s:
            # 检查链接是否符合特定格式
            if '/ftp/fullname/' in tag['href']:
                # 提取链接和链接文本
                link = tag['href']
                text = tag.get_text()
                links.append({'link': link, 'text': text})
    return links


def download_file(url, directory, filename):
    # 确保目录存在
    if not os.path.exists(directory):
        os.makedirs(directory)
    # 完整的文件路径
    file_path = os.path.join(directory, filename)
    # 如果文件已存在，则跳过下载
    if os.path.isfile(file_path):
        print(f"文件 {file_path} 已存在，跳过下载。")
        return
    # 尝试下载文件
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(file_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"文件已下载并保存到 {file_path}")
        else:
            print(f"下载失败，HTTP 状态码：{response.status_code}")
    except requests.RequestException as e:
        print(f"下载失败：{e}")


year_links = get_links(year_url)
url = url_prefix + year_links[29]['link']  # 2011
kind_of_data_links = get_links(url)
url = url_prefix + kind_of_data_links[0]['link']  # 15s1
format_links = get_links(url)
url = url_prefix + format_links[1]['link']  # nc
file_links = get_links(url)

dir = r"G:\0_postgraduate\DMSP\data\2011\15s1"
for dic in file_links:
    text = dic['text'].strip()  # 因为获取的text字符串前面有1个多余的空格
    link = dic['link']
    url = url_prefix + link
    download_file(url,directory=dir,filename=text)