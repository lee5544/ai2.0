import os
import sys
import argparse
import yaml
import glob

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from unsup_learning.dataset.dataset_ultis import extract_features_processing

def parse_args():
    parser = argparse.ArgumentParser(
        description="根据 YAML 配置提取 TDMS 特征")
    parser.add_argument(
        'config',
        help="YAML 配置文件路径，包含 dataset.tdms_folder、dataset.label_files、results_path 等字段")
    return parser.parse_args()

def main(config_path):
    # 加载标签和TDMS配置文件
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    TDMS_FOLDER = cfg["dataset"]["tdms_folder"]
    LABEL_FILES = cfg["dataset"]["label_files"]
    OUTPUT_FEATURE_FOLDER = (
        cfg["results_path"] + 
        cfg["line_name"] + "_" +
        cfg["model"]["model_name"] + "_" +
        cfg["train"]["model_type"] + "/dataset_csv"
    )
    os.makedirs(OUTPUT_FEATURE_FOLDER, exist_ok=True)

    # —— 新增：删除已有的 CSV 文件 —— #
    csv_pattern = os.path.join(OUTPUT_FEATURE_FOLDER, '*.csv')
    existing_csvs = glob.glob(csv_pattern)
    if existing_csvs:
        print(f"检测到 {len(existing_csvs)} 个旧的 CSV 特征文件，正在删除…")
        for csv_file in existing_csvs:
            try:
                os.remove(csv_file)
                print(f"  删除：{os.path.basename(csv_file)}")
            except Exception as e:
                print(f"  无法删除 {csv_file}：{e}")
    else:
        print("输出目录中未发现已有 CSV 文件。")

    print(f"特征文件输出目录已就绪：{OUTPUT_FEATURE_FOLDER}")
    print(f"处理标签文件：{LABEL_FILES}")

    # 提取特征并保存
    extract_features_processing(
        LABEL_FILES,
        TDMS_FOLDER,
        OUTPUT_FEATURE_FOLDER,
        output_file_prefix="features",
        batch_size=1000
    )

    print("所有数据处理完成！")

if __name__ == "__main__":
    args = parse_args()
    main(args.config)

