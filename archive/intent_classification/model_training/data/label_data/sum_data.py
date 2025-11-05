import json
from collections import Counter
import csv

# 你可以在这里自定义主意图字段名称
PRIMARY_FROM_NAMES = {"primary", "主意图"}  # 可修改为你项目中的实际字段名
FILES = ["../intent/ds_fixed_async_train.json"]
# FILES = ["./ds_fix_ls_tasks.json"]

def extract_primary_labels(item, primary_from_names):
    """从单个样本中提取主意图标签（去重后返回集合）"""
    labels = set()
    for ann in item.get("annotations", []):
        for res in ann.get("result", []):
            if res.get("type") == "choices" and res.get("from_name") in primary_from_names:
                choices = res.get("value", {}).get("choices", [])
                labels.update(choices)
    return labels

def count_primary(file_path, primary_from_names):
    """统计一个文件中的主意图标签数量"""
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    counter = Counter()
    for item in data:
        labels = extract_primary_labels(item, primary_from_names)
        counter.update(labels)

    print(f"\n📄 文件: {file_path}（样本数: {len(data)}）")
    if counter:
        for label, c in counter.most_common():
            print(f"  {label}: {c}")
        print(f"  合计（主意图标签项）: {sum(counter.values())}")
    else:
        print("  未找到主意图标签。")
    return counter

def main():
    total_counter = Counter()

    for file in FILES:
        try:
            c = count_primary(file, PRIMARY_FROM_NAMES)
            total_counter.update(c)
        except FileNotFoundError:
            print(f"⚠️ 文件未找到：{file}")

    print("\n🔹 全部文件汇总（仅主意图）：")
    if total_counter:
        for label, c in total_counter.most_common():
            print(f"  {label}: {c}")
        print(f"  总计（主意图标签项）: {sum(total_counter.values())}")
    else:
        print("  未找到主意图标签。")


if __name__ == "__main__":
    main()
