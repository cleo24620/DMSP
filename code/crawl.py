import os
import time

import requests
from bs4 import BeautifulSoup

def get_links(url,tag1='td',tag2='a'):
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
            if 'v1.0.4.cdf' in tag['href']:
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
            time.sleep(1)
        else:
            print(f"下载失败，HTTP 状态码：{response.status_code}")
    except requests.RequestException as e:
        print(f"下载失败：{e}")


url_prefix = f"https://spdf.gsfc.nasa.gov/pub/data/dmsp/dmspf18/ssm/magnetometer/2014/"
links_texts = get_links(url_prefix)
dir = fr"G:\0_postgraduate\DMSP\data\2014\f18\ssm"
for link_text in links_texts:
    start_time = time.time()
    try:
        download_file(url_prefix+link_text['link'], directory=dir, filename=link_text['text'])
    except Exception as e:
        print(e)
    end_time = time.time()
    print(f"the time for the loop is {end_time - start_time}")