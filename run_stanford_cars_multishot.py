# -*- coding: utf-8 -*-
import sys
import os
import torch
from core.config import Config
from core import Trainer

def run_experiment(shot_num):
    """运行指定shot数的实验"""
    print(f"\n Running {shot_num}-shot experiment on Stanford Cars...")
    
    # 变量字典，用于覆盖配置文件中的参数
    var_dict = {
        "shot_num": shot_num,
        "test_shot": shot_num,
        "tag": f"stanford_cars_{shot_num}shot",
        "device_ids": "0",  # 根据你的GPU情况调整
        "n_gpu": 1,
    }
    
    # 加载配置
    config = Config(
        "./config/stanford_cars_baseline.yaml", 
        var_dict
    ).get_config_dict()
    
    # 创建训练器并训练
    trainer = Trainer(0, config)
    trainer.train_loop(0)
    
    print(f" {shot_num}-shot experiment completed!")
    return config["result_root"]

def main():
    """运行1-16 shots的完整实验"""
    shot_numbers = list(range(1, 17))  # 1到16 shots
    
    results_summary = {}
    
    for shot_num in shot_numbers:
        try:
            result_path = run_experiment(shot_num)
            results_summary[shot_num] = {
                "status": "completed",
                "result_path": result_path
            }
        except Exception as e:
            print(f" Error in {shot_num}-shot experiment: {e}")
            results_summary[shot_num] = {
                "status": "failed",
                "error": str(e)
            }
    
    # 打印总结
    print("\n Experiment Summary:")
    print("=" * 50)
    for shot, result in results_summary.items():
        status = result["status"]
        print(f"{shot:2d}-shot: {status}")
    
    print("\n All experiments completed!")

if __name__ == "__main__":
    main()