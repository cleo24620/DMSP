import re

# 文件路径
fp = 'G:\\0_postgraduate\\DMSP\\data\\2014\\f17\\ssies3\\dmsp-f17_ssies-3_thermal-plasma_201401010056_v01.cdf'

# 正则表达式匹配日期和时间
pattern = r"dmsp-f17_ssies-3_thermal-plasma_(\d{8})(\d{4})_v\d{2}\.cdf"

# 搜索匹配
match = re.search(pattern, fp)

if match:
    date = match.group(1)  # 获取日期部分
    time = match.group(2)  # 获取时间部分
    print("Matched Date:", date)
    print("Matched Time:", time)
else:
    print("No match found.")
