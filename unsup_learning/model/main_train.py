#!/usr/bin/env python3
import argparse
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from unsup_learning.model.IFModel import IFModel
from unsup_learning.model.OCSVMModel import OCSVMModel
import yaml

def parse_args():
    parser = argparse.ArgumentParser(
        description="使用指定的 config 文件训练并评估模型")
    parser.add_argument(
        'config',
        help="YAML 配置文件路径，传给 模型训练")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # todo: 支持多种模型
    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    model_type = cfg['train']['model_type']
    train_mode = cfg['train']['train_mode']


    if model_type == "if":
        model = IFModel(args.config)
    elif model_type == "ocsvm":
        model = OCSVMModel(args.config)

    # 根据 --grid 标志选择训练方式
    if train_mode == "normal":
        model.train_and_evaluate()
    elif train_mode == "cross":
        model.train_and_cross_validate(5)
    elif train_mode == "grid":
        model.train_and_evaluate_grid()
    else:
        print("Error train_mode")
