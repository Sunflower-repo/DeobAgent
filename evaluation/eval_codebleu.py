#!/usr/bin/env python3
"""
批量计算js-deobfuscator文件夹中每个jsonl文件的平均CodeBLEU值
参考: /users/wcy/work/deob_agent_final/scripts/evaluation/cal_codebleu.py
"""

import json
import os
import glob
from typing import Any, Dict, Iterable, Optional
from codebleu import calc_codebleu

# 目标文件夹路径
TARGET_FOLDER = "/root/work/deob_agent_final/final_process_dataset/sample100_dataset/RQ1/mix/synchrony/"
# TARGET_FOLDER = "/root/work/deob_agent_final/final_process_dataset/sample100_dataset/RQ1/fewshotcot_qwen_processed/"

def calculate_codebleu(reference_code: str, prediction_code: str) -> float:
    """计算两段代码的CodeBLEU分数"""
    reference_list = [reference_code]
    prediction_list = [prediction_code]
    score = calc_codebleu(reference_list, prediction_list, "javascript")
    return float(score["codebleu"])  # 确保返回纯浮点数

def extract_code(sample: Dict[str, Any], key: str) -> Optional[str]:
    """从样本中提取代码"""
    value = sample.get(key)
    if value is None:
        return None
    # 如果是字典类型且包含'code'，优先使用那个
    if isinstance(value, dict):
        code_value = value.get("code")
        if isinstance(code_value, str):
            return code_value
        return None
    # 如果是字符串，直接作为代码处理
    if isinstance(value, str):
        return value
    return None

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """迭代读取jsonl文件"""
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)

def process_file(input_path: str) -> float:
    """处理单个jsonl文件，返回平均CodeBLEU分数"""
    total_score = 0.0
    num_scored = 0
    
    print(f"正在处理文件: {os.path.basename(input_path)}")
    
    for sample in iter_jsonl(input_path):
        # 提取原始函数和去混淆后的函数
        orig_code = extract_code(sample, "original_function")
        deob_code = extract_code(sample, "deobfuscated")
        # deob_code = extract_code(sample, "raw_deobfuscated_function")
        #print(orig_code)
        #print(deob_code)
        
        if orig_code is not None and deob_code is not None:
            try:
                score = calculate_codebleu(orig_code, deob_code)
                #print(score)
                total_score += score
                num_scored += 1
            except Exception as e:
                print(f"  警告: 计算CodeBLEU时出错: {e}")
                continue
        else:
            print(f"  警告: 缺少必要的代码字段")
            continue
    
    if num_scored == 0:
        print(f"  错误: 没有有效的样本用于计算CodeBLEU")
        return 0.0
    
    avg_score = total_score / num_scored
    print(f"  有效样本数: {num_scored}, 平均CodeBLEU: {avg_score:.6f}")
    return avg_score

def main():
    """主函数"""
    if not os.path.isdir(TARGET_FOLDER):
        print(f"错误: 目标文件夹不存在: {TARGET_FOLDER}")
        return
    
    # 查找所有jsonl文件
    jsonl_pattern = os.path.join(TARGET_FOLDER, "*.jsonl")
    jsonl_files = glob.glob(jsonl_pattern)
    
    if not jsonl_files:
        print(f"错误: 在 {TARGET_FOLDER} 中没有找到jsonl文件")
        return
    
    print(f"找到 {len(jsonl_files)} 个jsonl文件")
    print("=" * 60)
    
    results = []
    
    # 处理每个文件
    for jsonl_file in sorted(jsonl_files):
        try:
            avg_codebleu = process_file(jsonl_file)
            filename = os.path.basename(jsonl_file)
            results.append((filename, avg_codebleu))
        except Exception as e:
            print(f"处理文件 {jsonl_file} 时出错: {e}")
            continue
        print("-" * 40)
    
    # 输出最终结果
    print("\n最终结果:")
    print("=" * 60)
    print(f"{'文件名':<50} {'平均CodeBLEU':<15}")
    print("-" * 65)
    
    # 将每个文件的平均值保留2位小数
    rounded_results = []
    for filename, avg_codebleu in results:
        rounded_score = round(avg_codebleu, 2)
        rounded_results.append((filename, rounded_score))
        print(f"{filename:<50} {rounded_score:<15.2f}")
    
    # 计算总体平均值（基于保留2位小数后的值）
    if rounded_results:
        overall_avg = sum(score for _, score in rounded_results) / len(rounded_results)
        print("-" * 65)
        print(f"{'总体平均值':<50} {overall_avg:<15.2f}")

if __name__ == "__main__":
    main()
