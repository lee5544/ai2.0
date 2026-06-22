import pandas as pd
import numpy as np
from scipy import stats
import pywt
from collections import Counter


def calculate_entropy(list_values):
    counter_values = Counter(list_values).most_common()
    probabilities = [elem[1] / len(list_values) for elem in counter_values]
    entropy = stats.entropy(probabilities)
    return entropy


def calculate_statistics(list_values):
    n5 = np.nanpercentile(list_values, 5)
    n25 = np.nanpercentile(list_values, 25)
    n75 = np.nanpercentile(list_values, 75)
    n95 = np.nanpercentile(list_values, 95)
    median = np.nanpercentile(list_values, 50)
    mean = np.nanmean(list_values)
    std = np.nanstd(list_values)
    var = np.nanvar(list_values)
    rms = np.nanmean(np.sqrt(list_values ** 2))
    return {
        "p05": n5, "p25": n25, "p75": n75, "p95": n95,
        "median": median, "mean": mean, "std": std,
        "var": var, "rms": rms
    }


def calculate_crossings(list_values):
    zero_crossing_indices = np.nonzero(np.diff(np.array(list_values) > 0))[0]
    no_zero_crossings = len(zero_crossing_indices)
    mean_crossing_indices = np.nonzero(np.diff(np.array(list_values) > np.nanmean(list_values)))[0]
    no_mean_crossings = len(mean_crossing_indices)
    return {"zero_crossings": no_zero_crossings, "mean_crossings": no_mean_crossings}


def get_feature_trend_fluctuation(signal, window_size):
    signal = signal.T.squeeze()
    idx_start = 0
    idx_end = len(signal)
    values = []
    for idx in range(idx_start, idx_end, window_size):
        idx_window = idx + window_size
        if idx_window > idx_end:
            idx_window = idx_end
        short_sig = signal[idx:idx_window]
        value = (np.mean(short_sig), np.mean(np.abs(short_sig)))
        values.append(value)

    df_value = pd.DataFrame(values)
    df_value.columns = ["value_mean", "value_mean_abs"]

    var_mean = np.nanvar(df_value.value_mean)
    var_mean_abs = np.nanvar(df_value.value_mean_abs)

    return {"trend_var_mean": var_mean, "trend_var_mean_abs": var_mean_abs}


def get_features(list_values, prefix=""):
    feats = {}
    feats[f"{prefix}entropy"] = calculate_entropy(list_values)

    crossings = calculate_crossings(list_values)
    for k, v in crossings.items():
        feats[f"{prefix}{k}"] = v

    statistics = calculate_statistics(list_values)
    for k, v in statistics.items():
        feats[f"{prefix}{k}"] = v

    fluct = get_feature_trend_fluctuation(list_values, 1000)
    for k, v in fluct.items():
        feats[f"{prefix}{k}"] = v

    return feats


def extract_ai_v1(raw_data, up_or_down, level=4):    
    # if up_or_down == "down":
    #     dat_cut = raw_data[5000:107000]
    # elif up_or_down == "up":
    #     dat_cut = raw_data[5000:90000]

    dat_cut = raw_data[5000:-5000]
    wavelet_name = "db22"
    level = int(level)

    features = {}

    list_coeff = pywt.wavedec(dat_cut, wavelet_name, level=level)
    for i, coeff in enumerate(list_coeff):
        feats = get_features(coeff, prefix=f"coeff{i}_")
        features.update(feats)

    return features


# def extract_ai_v1(raw_data, up_or_down, level=4, window_size=10000, overlap=0.5):
#     # if up_or_down == "down":
#     #     dat_cut = raw_data[5000:107000]
#     # elif up_or_down == "up":
#     #     dat_cut = raw_data[5000:90000]
#     dat_cut = raw_data[5000:90000]

#     wavelet_name = "db22"
#     level = int(level)

#     step_size = int(window_size * (1 - overlap))  
#     num_windows = (len(dat_cut) - window_size) // step_size + 1  

#     features = {}

#     for win_idx in range(num_windows):
#         window_data = dat_cut[win_idx * step_size:win_idx * step_size + window_size]
#         list_coeff = pywt.wavedec(window_data, wavelet_name, level=level)
#         for i, coeff in enumerate(list_coeff):
#             feats = get_features(coeff, prefix=f"win{win_idx}_coeff{i}_")
#             features.update(feats)
    
#     # print(features)

#     return features


if __name__ == "__main__":
    raw_data = np.random.randn(110000)  # 模拟数据
    feats1 = extract_ai_v1(raw_data, "up")

    print("extract_ai_v1 输出 keys:", list(feats1.keys())[:10])
