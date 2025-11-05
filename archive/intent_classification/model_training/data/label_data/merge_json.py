import os
import json


def merge_json_files(folder_path, output_file):
    """
    将指定文件夹中的所有 .json 文件合并成一个列表，并保存为新的 JSON 文件。
    :param folder_path: 包含 JSON 文件的文件夹路径
    :param output_file: 输出合并后文件的路径
    """
    merged_data = []

    # 遍历文件夹
    for filename in os.listdir(folder_path):
        if filename.endswith(".json"):
            file_path = os.path.join(folder_path, filename)
            print(f"🔹 正在读取: {file_path}")
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # 如果文件是列表，直接扩展
                    if isinstance(data, list):
                        merged_data.extend(data)
                    else:
                        merged_data.append(data)
            except Exception as e:
                print(f"⚠️ 读取 {filename} 出错: {e}")

    # 保存合并结果
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 合并完成！共合并 {len(merged_data)} 条数据，已保存到: {output_file}")


# ==== 使用示例 ====
if __name__ == "__main__":
    folder = r"./time"  # 👈 改成你的文件夹路径
    output = r"./ds_fix_ls_tasks.json"
    merge_json_files(folder, output)
