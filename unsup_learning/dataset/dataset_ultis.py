import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
from tqdm import tqdm
from typing import List, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed
from ultis.tdms_ultis import find_tdms
from unsup_learning.dataset.extract_features_v1 import extract_ai_v1
from ultis.tdms_ultis import extract_acc_raw_data, pre_handle_labels

def extract_features_ai2(acc_raw, sampling_rate, up_or_down=None):
    # 返回特征value列表

    features = extract_ai_v1(acc_raw, 4)

    values = list(features.values())
    values = np.expand_dims(values, axis=0)

    return values

def extract_feature_keys_bootstrap(labels_files: List[Dict[str, str]],
                                   tdms_folders: List[str]) -> List[str]:
    """
    遍历标签与TDMS目录，找到第一条可用样本，调用 extract_features_v2
    返回稳定的特征名列表（dict的keys顺序保持插入顺序）
    """
    for entry in labels_files:
        df = pre_handle_labels(entry["path"])
        line_name = entry["line"]
        for _, row in df.iterrows():
            sn = row.get("sn")
            tdms_file = find_tdms(tdms_folders, sn)
            if tdms_file and os.path.exists(tdms_file):
                # 读取TDMS
                data = extract_acc_raw_data(tdms_file, line_name)
                # 先尝试 up 通道
                up_raw = getattr(getattr(data, "up", None), "vib", None)
                if up_raw is not None:
                    # feats = extract_features_v2(up_raw, data.sampling_rate)
                    feats = extract_ai_v1(up_raw, 4)

                    return list(feats.keys())
                # 再尝试 down 通道
                down_raw = getattr(getattr(data, "down", None), "vib", None)
                if down_raw is not None:
                    # feats = extract_features_v2(down_raw, data.sampling_rate)
                    feats = extract_ai_v1(down_raw, 4)

                    return list(feats.keys())
    raise RuntimeError("无法通过样本自举特征名：未找到任何可用的 TDMS 样本。")


def process_record(line_name: str, row: pd.Series, tdms_folders: List[str]) -> List[List]:
    """
    处理单条记录：读取 TDMS、提取特征并返回记录列表
    返回值：
        [
          [filename, sn, channel, label, seq_length, num_features, f1, f2, ...],
          ...
        ]
    """
    recs = []
    sn = row.get("sn")
    up_label = row.get("up_label")
    down_label = row.get("down_label")

    tdms_file = find_tdms(tdms_folders, sn)

    if tdms_file and os.path.exists(tdms_file):

        # 从 TDMS 中提取原始信号
        data = extract_acc_raw_data(tdms_file, line_name)

        # 提取特征（AI v3）
        acc_raw = data.up.vib
        if acc_raw is not None:
            up_feat = extract_features_ai2(acc_raw, data.sampling_rate)

            recs.append([
                os.path.basename(tdms_file), sn, "up", up_label,
                up_feat.shape[0], up_feat.shape[1], *up_feat.flatten()
            ])

        down_raw = data.down.vib
        if down_raw is not None:
            down_feat = extract_features_ai2(down_raw, data.sampling_rate)
            recs.append([
                os.path.basename(tdms_file), sn, "down", down_label,
                down_feat.shape[0], down_feat.shape[1], *down_feat.flatten()
            ])

    else:
        print(f"⚠️ 未找到 TDMS 文件: {sn}, {tdms_file}")

    return recs

def extract_features_processing(
    labels_files: List[Dict[str, str]],
    tdms_folders: List[str],
    output_feature_folder: str,
    output_file_prefix: str = "features",
    batch_size: int = 1000,
    max_workers: int = os.cpu_count()
):
    """
    多进程特征提取函数，按批次保存 CSV 文件。

    参数:
        labels_files: [{'path': ..., 'line': ...}, ...]
        tdms_folders: 存放 TDMS 文件的文件夹列表
        output_feature_folder: 特征文件输出目录
        output_file_prefix: 特征文件名前缀
        batch_size: 每个文件包含的最大样本数量
        max_workers: 并发进程数（默认 CPU 核数）
    """
    os.makedirs(output_feature_folder, exist_ok=True)
    batch = []
    batch_index = 1
    feature_columns = []

    # 构造所有任务
    tasks = []
    for entry in labels_files:
        df = pre_handle_labels(entry["path"])
        for _, row in df.iterrows():
            tasks.append((entry, row))

    # 并行执行
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_record, entry["line"], row, tdms_folders): (entry, row)
            for entry, row in tasks
        }

        if not feature_columns:
            try:
                feature_keys = extract_feature_keys_bootstrap(labels_files, tdms_folders)
            except Exception as e:
                # 保险兜底：若自举失败，退回到无名特征列（不推荐，但可继续跑通）
                print(f"⚠️ 自举特征名失败：{e}. 将回退为 f1..fn 命名。")
                n_feats = len(batch[0]) - 6
                feature_keys = [f"f{i}" for i in range(1, n_feats + 1)]

            feature_columns = [
                'filename', 'sn', 'channel', 'label', 'seq_length', 'num_features'
            ] + feature_keys

        for future in tqdm(as_completed(futures), total=len(futures), desc="🔄 提取特征"):
            recs = future.result()
            if recs:
                batch.extend(recs)

            # 达到批次大小则写入文件
            if len(batch) >= batch_size:

                df_batch = pd.DataFrame(batch, columns=feature_columns)
                out_path = os.path.join(
                    output_feature_folder,
                    f"{output_file_prefix}_batch_{batch_index}.csv"
                )
                df_batch.to_csv(out_path, index=False, encoding="utf-8-sig")
                print(f"✅ 批次 {batch_index}: 保存 {len(batch)} 条记录 -> {out_path}")

                batch.clear()
                batch_index += 1

    # 保存最后一批不足 batch_size 的记录
    if batch:
        df_batch = pd.DataFrame(batch, columns=feature_columns)
        out_path = os.path.join(
            output_feature_folder,
            f"{output_file_prefix}_batch_{batch_index}.csv"
        )
        df_batch.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"✅ 最后一批: 保存 {len(batch)} 条记录 -> {out_path}")

if __name__ == "__main__":
    # 示例调用
    labels_files = [
        {'path': '/Volumes/.../epump3_labels.csv', 'line': 'EPUMP3'},
        {'path': '/Volumes/.../epump4_labels.csv', 'line': 'EPUMP4'},
        # ...
    ]
    tdms_folders = [
        '/Volumes/.../fault/laser/epump3',
        '/Volumes/.../fault/laser/epump4',
        # ...
    ]
    extract_features_processing(
        labels_files,
        tdms_folders,
        output_feature_folder='./features_output',
        output_file_prefix='epump_features',
        batch_size=1000,
        max_workers=4  # 或 os.cpu_count()
    )
