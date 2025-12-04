#!/usr/bin/env python3
"""
批量计算js-deobfuscator或synchrony文件夹中每个jsonl文件的平均ROUGE-L F1值
"""

import json
import os
import glob
from typing import Any, Dict, Iterable, Optional, Tuple
from rouge_score import rouge_scorer

# 目标文件夹路径 - 可以修改为不同的baseline文件夹
# TARGET_FOLDER = "/root/work/deob_agent_final/final_process_dataset/sample100_dataset/RQ3_new/claude_processed/"
TARGET_FOLDER = "/root/work/deob_agent_final/final_process_dataset/sample100_dataset/RQ1/mix/synchrony/"

def calculate_rouge(reference_code: str, prediction_code: str) -> float:
    """计算两段代码的ROUGE-L F1分数"""
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    scores = scorer.score(reference_code, prediction_code)
    
    return scores['rougeL'].fmeasure

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
    """处理单个jsonl文件，返回平均ROUGE-L F1分数"""
    rouge_total = 0.0
    num_scored = 0
    
    print(f"正在处理文件: {os.path.basename(input_path)}")
    
    for sample in iter_jsonl(input_path):
        # 提取原始函数和去混淆后的函数
        orig_code = extract_code(sample, "original_function")
        # orig_code = extract_code(sample, "original")
        deob_code = extract_code(sample, "deobfuscated")
        
        if orig_code is not None and deob_code is not None:
            try:
                rouge_score = calculate_rouge(orig_code, deob_code)
                rouge_total += rouge_score
                num_scored += 1
            except Exception as e:
                print(f"  警告: 计算ROUGE时出错: {e}")
                continue
        else:
            print(f"  警告: 缺少必要的代码字段")
            continue
    
    if num_scored == 0:
        print(f"  错误: 没有有效的样本用于计算ROUGE")
        return 0.0
    
    # 计算平均值
    rouge_average = rouge_total / num_scored
    
    print(f"  有效样本数: {num_scored}")
    print(f"  ROUGE-L F1: {rouge_average:.2f}")
    
    return rouge_average

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
    print("=" * 80)
    
    results = []
    
    # 处理每个文件
    for jsonl_file in sorted(jsonl_files):
        try:
            rouge_score = process_file(jsonl_file)
            filename = os.path.basename(jsonl_file)
            results.append((filename, rouge_score))
        except Exception as e:
            print(f"处理文件 {jsonl_file} 时出错: {e}")
            continue
        print("-" * 60)
    
    # 输出最终结果
    print("\n最终结果:")
    print("=" * 80)
    print(f"{'文件名':<50} {'ROUGE-L F1':<12}")
    print("-" * 80)
    
    rouge_total = 0.0
    
    for filename, rouge_score in results:
        print(f"{filename:<50} {rouge_score:<12.2f}")
        rouge_total += rouge_score
    
    # 计算总体平均值
    if results:
        num_files = len(results)
        overall_average = rouge_total / num_files
        print("-" * 80)
        print(f"{'总体平均值':<50} {overall_average:<12.2f}")

if __name__ == "__main__":
    main()
