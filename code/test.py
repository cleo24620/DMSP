import pandas as pd
import numpy as np

# 示例数据
data = [1, 2, np.nan, np.nan, 5, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan,
        20, 21, np.nan, 22]
series = pd.Series(data)


def split_series(series, threshold=10):
    # 获取连续NaNs的长度
    is_nan = series.isna()
    cumsum = is_nan.cumsum()
    shifted = cumsum.shift().fillna(0)
    reset = cumsum.sub(shifted).where(is_nan)
    reset_groups = reset.groupby((reset != reset.shift()).cumsum())
    lengths = reset_groups.cumcount() + 1
    starts = reset_groups.apply(lambda x: x.index[0])
    ends = reset_groups.apply(lambda x: x.index[-1])

    # 寻找长度超过阈值的连续NaNs
    mask = lengths > threshold
    start_ends = list(zip(starts[mask], ends[mask]))

    # 分割数据
    if not start_ends:
        return [series]

    # 初始化分割点为第一个连续NaN序列之前
    split_points = [0] + [end + 1 for _, end in start_ends]
    if split_points[-1] != len(series):
        split_points.append(len(series))  # 确保包含最后一个段

    # 生成分割的数据段
    parts = [series.iloc[split_points[i]:split_points[i + 1]] for i in range(len(split_points) - 1)]

    # 过滤掉全为NaN的段
    parts = [part for part in parts if not part.isna().all()]

    return parts


# 应用函数并输出结果
parts = split_series(series)
for i, part in enumerate(parts):
    print(f"Part {i + 1}:")
    print(part.tolist())
