#!/usr/bin/env python3
"""
批量使用LLM作为评判者评估反混淆代码质量
参考: /root/work/deob_agent_final/scripts/evaluation/batch_cal_codebleu.py
基于论文中的LLM-as-Judge评估方法
"""

import json
import os
import glob
import time
import threading
import random
from typing import Any, Dict, Iterable, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

# 目标文件夹路径
TARGET_FOLDER = "/root/work/deob_agent_final/final_process_dataset/sample100_dataset/RQ1/fewshotcot_deepseek_processed/"

# LLM API配置
# LLM_API_URL = "https://api.bianxie.ai/v1"
# LLM_API_KEY = "sk-lZFPwlQ7xjUciiHsM9rjMJ1fc2KHzN2fc6rczc3eeJiCs75K"
# LLM_MODEL = "gpt-4"

LLM_API_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
LLM_API_KEY = "sk-76468e3fe8124c42a7282da310b0a182"
LLM_MODEL = "qwen3-coder-plus"


# 多线程配置
MAX_WORKERS = 20  # 最大线程数

# 评估配置
NUM_EVALUATIONS_PER_DIMENSION = 1  # 每个维度评估5次以提高可靠性

# 线程局部存储，每个线程都有自己的client实例
thread_local = threading.local()

def get_client():
    """获取线程局部的OpenAI客户端"""
    if not hasattr(thread_local, 'client'):
        thread_local.client = OpenAI(
            api_key=LLM_API_KEY,
            base_url=LLM_API_URL
        )
    return thread_local.client

def create_evaluation_prompt(original_code: str, deobfuscated_code: str, randomize: bool = True) -> str:
    """
    创建评估提示词
    根据论文要求，比较反混淆代码与原始代码的相似度
    评估四个维度，每个维度1-10分
    """
    # 匿名化和随机化代码顺序（论文要求）
    if randomize and random.random() < 0.5:
        code_a, code_b = deobfuscated_code, original_code
        code_a_label, code_b_label = "Code A", "Code B"
        is_swapped = True
    else:
        code_a, code_b = original_code, deobfuscated_code
        code_a_label, code_b_label = "Code A", "Code B"
        is_swapped = False
    
    prompt = f"""You are a professional code evaluator. You are given two JavaScript code snippets. Your task is to evaluate how similar Code B is to Code A across four dimensions.

Evaluation Dimensions (each scored 1 to 10, where 10 is highest similarity):

1. Identifier Name Similarity (标识符命名相似度):
Measures whether variable names, function names, and other identifiers match the original naming conventions and are semantically consistent.
9-10 — Nearly identical identifier naming; all major identifiers match or have equivalent semantic meaning.
7-8 — Most identifiers are similar or semantically equivalent; minor differences in naming style.
5-6 — Moderate similarity; some identifiers match, others differ but are still meaningful.
3-4 — Low similarity; most identifiers are different, though some patterns may remain.
1-2 — Very low similarity; identifier names are completely different or generic.

2. Code Style Similarity (代码风格相似度):
Measures whether the code follows the same coding style, including formatting conventions, indentation patterns, spacing, and bracket placement.
9-10 — Nearly identical code style; formatting, spacing, and conventions match perfectly.
7-8 — Highly similar style; minor differences in formatting or conventions.
5-6 — Moderate style similarity; some formatting differences but overall consistent approach.
3-4 — Low style similarity; significant differences in formatting and conventions.
1-2 — Very low similarity; completely different coding styles.

3. AST Structure Similarity (抽象语法树结构相似度):
Measures whether the code maintains the same abstract syntax tree structure, including node types, tree topology, and hierarchical organization.
9-10 — Nearly identical AST structure; same node types and tree organization.
7-8 — Highly similar AST; minor structural differences but same overall organization.
5-6 — Moderate AST similarity; some structural differences but core patterns remain.
3-4 — Low AST similarity; significant structural differences in tree organization.
1-2 — Very low similarity; completely different AST structures.

4. Cyclomatic Complexity Similarity (圈复杂度相似度):
Measures whether the code has similar cyclomatic complexity, indicating comparable logical complexity in terms of decision points and control flow paths.
9-10 — Nearly identical complexity; same number of decision points and control flow paths.
7-8 — Highly similar complexity; minor differences in control flow.
5-6 — Moderate complexity similarity; some differences in decision points.
3-4 — Low complexity similarity; significant differences in logical complexity.
1-2 — Very low similarity; completely different complexity levels.

INPUT:
{code_a_label}:
```javascript
{code_a}
```

{code_b_label}:
```javascript
{code_b}
```

OUTPUT:
Please evaluate how similar Code B is to Code A. Output ONLY a JSON object with 4 integer fields in the range [1,10]:
{{
"identifier_name_similarity": <1-10>,
"code_style_similarity": <1-10>,
"ast_structure_similarity": <1-10>,
"cyclomatic_complexity_similarity": <1-10>
}}

REQUIREMENT:
No explanation, no extra text, no formatting other than valid JSON. Ensure the JSON is properly formatted and parseable."""
    
    return prompt, is_swapped

def call_llm_api(prompt: str, max_retries: int = 3) -> Optional[Dict[str, int]]:
    """调用LLM API进行评估"""
    client = get_client()  # 使用线程局部客户端
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,  # 稍微提高温度以支持多次评估
                max_tokens=200,
                stream=False  # 明确禁用流式响应
            )
            
            # 提取内容
            content = response.choices[0].message.content.strip()
            
            # 尝试解析JSON
            try:
                # 处理可能的代码块格式
                if content.startswith("```json"):
                    # 提取代码块中的JSON
                    lines = content.split('\n')
                    json_lines = []
                    in_json = False
                    for line in lines:
                        if line.strip() == "```json":
                            in_json = True
                            continue
                        elif line.strip() == "```":
                            break
                        elif in_json:
                            json_lines.append(line)
                    content = '\n'.join(json_lines)
                elif content.startswith("```"):
                    # 处理其他代码块格式
                    lines = content.split('\n')
                    json_lines = []
                    in_json = False
                    for line in lines:
                        if line.strip().startswith("```"):
                            in_json = True
                            continue
                        elif line.strip() == "```":
                            break
                        elif in_json:
                            json_lines.append(line)
                    content = '\n'.join(json_lines)
                
                scores = json.loads(content)
                # 验证分数范围（论文要求：1-10）
                for key, value in scores.items():
                    if not isinstance(value, int) or value < 1 or value > 10:
                        raise ValueError(f"Invalid score {value} for {key} (must be 1-10)")
                return scores
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  警告: JSON解析失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                print(f"  LLM响应: {content}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                return None
                
        except Exception as e:
            print(f"  警告: API调用失败 (尝试 {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None
    
    return None

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

def evaluate_with_multiple_runs(original_code: str, deobfuscated_code: str, num_runs: int = NUM_EVALUATIONS_PER_DIMENSION) -> Optional[Dict[str, float]]:
    """
    对每个维度进行多次评估并计算平均分
    论文要求：每个维度评估5次以提高可靠性
    """
    all_scores = []
    
    for run in range(num_runs):
        prompt, is_swapped = create_evaluation_prompt(original_code, deobfuscated_code, randomize=True)
        scores = call_llm_api(prompt)
        
        if scores is None:
            continue
            
        all_scores.append(scores)
        time.sleep(0.5)  # 避免API限流
    
    if not all_scores:
        return None
    
    # 计算每个维度的平均分
    avg_scores = {}
    dimensions = ["identifier_name_similarity", "code_style_similarity", 
                  "ast_structure_similarity", "cyclomatic_complexity_similarity"]
    
    for dim in dimensions:
        dim_scores = [s[dim] for s in all_scores if dim in s]
        if dim_scores:
            avg_scores[dim] = sum(dim_scores) / len(dim_scores)
        else:
            avg_scores[dim] = 0.0
    
    # 添加统计信息
    avg_scores['num_evaluations'] = len(all_scores)
    
    return avg_scores

def process_single_sample(sample_data: Tuple[int, Dict[str, Any]]) -> Tuple[int, Optional[Dict[str, float]]]:
    """
    处理单个样本
    论文要求：比较反混淆代码与原始代码（非混淆版本）
    """
    index, sample = sample_data
    
    # 提取原始函数（未混淆）和反混淆后的函数
    orig_code = extract_code(sample, "original_function")
    deob_code = extract_code(sample, "deobfuscated")
    
    if orig_code is not None and deob_code is not None:
        try:
            scores = evaluate_with_multiple_runs(orig_code, deob_code)
            return index, scores
        except Exception as e:
            print(f"  警告: 处理样本 {index+1} 时出错: {e}")
            return index, None
    else:
        print(f"  警告: 样本 {index+1} 缺少必要的代码字段")
        return index, None

def process_file(input_path: str, max_workers: int = MAX_WORKERS) -> Tuple[float, Dict[str, float], int, float]:
    """处理单个jsonl文件，返回平均分数和详细统计"""
    print(f"正在处理文件: {os.path.basename(input_path)}")
    
    # 读取所有数据
    data_items = []
    with open(input_path, 'r', encoding='utf-8') as infile:
        for index, line in enumerate(infile):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                data_items.append((index, obj))
            except Exception as e:
                print(f"JSON 解析失败，已跳过第{index+1}行: {e}")
                continue

    if not data_items:
        print(f"  错误: 文件 {input_path} 没有有效数据可处理")
        return 0.0, {}, 0, 0.0

    print(f"  开始处理 {len(data_items)} 个样本，使用 {max_workers} 个线程...")
    print(f"  每个样本将进行 {NUM_EVALUATIONS_PER_DIMENSION} 次评估")
    
    # 存储结果的字典，key是原始索引
    results = {}
    
    # 多线程处理
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_index = {
            executor.submit(process_single_sample, item): item[0] 
            for item in data_items
        }
        
        # 收集结果
        for future in as_completed(future_to_index):
            try:
                index, scores = future.result()
                if scores is not None:
                    results[index] = scores
                    # 计算总分（排除num_evaluations字段）
                    dimension_scores = {k: v for k, v in scores.items() if k != 'num_evaluations'}
                    total_score = sum(dimension_scores.values())
                    num_evals = scores.get('num_evaluations', 0)
                    print(f"  样本 {index+1} ({num_evals}次评估): 总分={total_score:.2f}")
                    for dim, score in dimension_scores.items():
                        print(f"    {dim}: {score:.2f}")
                else:
                    print(f"  样本 {index+1}: 评估失败")
            except Exception as e:
                original_index = future_to_index[future]
                print(f"  处理样本 {original_index+1} 时出错: {e}")
    
    if not results:
        print(f"  错误: 没有有效的样本用于评估")
        return 0.0, {}, 0, 0.0
    
    # 计算平均分数（论文中的四个维度）
    dimensions = ["identifier_name_similarity", "code_style_similarity", 
                  "ast_structure_similarity", "cyclomatic_complexity_similarity"]
    
    total_scores = {dim: 0.0 for dim in dimensions}
    
    # 计算每个样本的总分
    sample_totals = []
    for scores in results.values():
        dimension_scores = {k: v for k, v in scores.items() if k in dimensions}
        sample_total = sum(dimension_scores.values())
        sample_totals.append(sample_total)
        for dim in dimensions:
            if dim in scores:
                total_scores[dim] += scores[dim]
    
    num_scored = len(results)
    avg_scores = {dim: total_scores[dim] / num_scored for dim in dimensions}
    overall_avg = sum(avg_scores.values()) / len(avg_scores)
    
    # 计算总分的平均值
    avg_total_score = sum(sample_totals) / len(sample_totals) if sample_totals else 0.0
    
    print(f"\n  有效样本数: {num_scored}")
    print(f"  平均分数 (基于论文评估维度):")
    print(f"    Textual Similarity:")
    print(f"      Identifier Name Similarity: {avg_scores['identifier_name_similarity']:.2f}")
    print(f"      Code Style Similarity: {avg_scores['code_style_similarity']:.2f}")
    print(f"    Structure Similarity:")
    print(f"      AST Structure Similarity: {avg_scores['ast_structure_similarity']:.2f}")
    print(f"      Cyclomatic Complexity Similarity: {avg_scores['cyclomatic_complexity_similarity']:.2f}")
    print(f"  总体平均: {overall_avg:.2f}")
    print(f"  总分平均: {avg_total_score:.2f}/40")
    
    return overall_avg, avg_scores, num_scored, avg_total_score

def main():
    """主函数"""
    target_folder = TARGET_FOLDER
    
    if not os.path.isdir(target_folder):
        print(f"错误: 目标文件夹不存在: {target_folder}")
        return
    
    # 查找所有jsonl文件
    jsonl_pattern = os.path.join(target_folder, "*.jsonl")
    jsonl_files = glob.glob(jsonl_pattern)
    
    if not jsonl_files:
        print(f"错误: 在 {target_folder} 中没有找到jsonl文件")
        return
    
    print(f"========== LLM-as-Judge 评估 (基于论文方法) ==========")
    print(f"找到 {len(jsonl_files)} 个jsonl文件")
    print(f"使用模型: {LLM_MODEL}")
    print(f"每个样本评估次数: {NUM_EVALUATIONS_PER_DIMENSION}")
    print(f"评估维度: Textual Similarity (Identifier Name, Code Style) + Structure Similarity (AST, Cyclomatic Complexity)")
    print(f"分数范围: 1-10 (10为最高)")
    print("=" * 100)
    
    results = []
    detailed_results = []
    
    # 处理每个文件
    for jsonl_file in sorted(jsonl_files):
        try:
            overall_avg, avg_scores, num_samples, avg_total_score = process_file(jsonl_file)
            filename = os.path.basename(jsonl_file)
            results.append((filename, overall_avg, num_samples, avg_total_score))
            detailed_results.append((filename, avg_scores, num_samples))
        except Exception as e:
            print(f"处理文件 {jsonl_file} 时出错: {e}")
            continue
        print("-" * 100)
    
    # 输出最终结果
    print("\n" + "=" * 100)
    print("最终结果汇总:")
    print("=" * 100)
    print(f"{'文件名':<50} {'样本数':<10} {'总体平均':<12} {'总分平均':<12}")
    print("-" * 100)
    
    total_samples = 0
    weighted_sum = 0.0
    weighted_total_sum = 0.0
    
    for filename, overall_avg, num_samples, avg_total_score in results:
        print(f"{filename:<50} {num_samples:<10} {overall_avg:<12.2f} {avg_total_score:<12.2f}")
        total_samples += num_samples
        weighted_sum += overall_avg * num_samples
        weighted_total_sum += avg_total_score * num_samples
    
    # 计算加权平均
    if total_samples > 0:
        overall_weighted_avg = weighted_sum / total_samples
        overall_weighted_total = weighted_total_sum / total_samples
        print("-" * 100)
        print(f"{'加权总体平均':<50} {total_samples:<10} {overall_weighted_avg:<12.2f} {overall_weighted_total:<12.2f}")
    
    # 输出详细统计（按论文中的分类）
    print("\n" + "=" * 100)
    print("详细维度统计 (基于论文评估方法):")
    print("=" * 100)
    
    dimensions = ["identifier_name_similarity", "code_style_similarity", 
                  "ast_structure_similarity", "cyclomatic_complexity_similarity"]
    
    print("\n【Textual Similarity (文本相似度)】")
    for dim in ["identifier_name_similarity", "code_style_similarity"]:
        dim_sum = 0.0
        dim_samples = 0
        
        for filename, avg_scores, num_samples in detailed_results:
            if dim in avg_scores:
                dim_sum += avg_scores[dim] * num_samples
                dim_samples += num_samples
        
        if dim_samples > 0:
            dim_avg = dim_sum / dim_samples
            dim_name = "Identifier Name Similarity" if dim == "identifier_name_similarity" else "Code Style Similarity"
            print(f"  {dim_name:<40}: {dim_avg:.2f}/10")
    
    print("\n【Structure Similarity (结构相似度)】")
    for dim in ["ast_structure_similarity", "cyclomatic_complexity_similarity"]:
        dim_sum = 0.0
        dim_samples = 0
        
        for filename, avg_scores, num_samples in detailed_results:
            if dim in avg_scores:
                dim_sum += avg_scores[dim] * num_samples
                dim_samples += num_samples
        
        if dim_samples > 0:
            dim_avg = dim_sum / dim_samples
            dim_name = "AST Structure Similarity" if dim == "ast_structure_similarity" else "Cyclomatic Complexity Similarity"
            print(f"  {dim_name:<40}: {dim_avg:.2f}/10")
    
    print("\n" + "=" * 100)
    print("评估完成！")
    print("=" * 100)

if __name__ == "__main__":
    main()
