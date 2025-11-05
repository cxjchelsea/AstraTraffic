import json
import re

def md_to_labelstudio(md_file_path, output_file_path, primary_label, secondary_label=None):
    # 读取Markdown文件内容
    with open(md_file_path, 'r', encoding='utf-8') as f:
        md_content = f.read()

    # 提取Markdown中的问题列表
    questions = re.findall(r"^\d+\.\s*(.*?)\s*$", md_content, re.MULTILINE)

    # 处理每个问题，并生成Label Studio格式的数据
    labelstudio_data = []
    for i, question in enumerate(questions, start=1):
        task_data = {
            "id": i,  # 任务ID
            "annotations": [
                {
                    "id": i,
                    "completed_by": 1,  # 用户ID
                    "result": [
                        {
                            "from_name": "primary",  # 主意图标签
                            "to_name": "text",
                            "type": "choices",
                            "value": {
                                "choices": [primary_label]  # 设置主意图
                            }
                        }
                    ],
                    "was_cancelled": False,
                    "ground_truth": False,
                    "created_at": "2025-10-23T05:52:32.798024Z",
                    "updated_at": "2025-10-23T05:52:32.798024Z",
                    "draft_created_at": None,
                    "lead_time": None,
                    "prediction": {},
                    "result_count": 0,
                    "unique_id": f"cd013262-39d2-4e1f-b24b-c998369c6e7e-{i}",
                    "task": i,
                    "project": 1,
                    "updated_by": None
                }
            ],
            "file_upload": "56817ef4-ls_tasks_ascii.json",  # 文件上传的ID
            "data": {
                "text": question,  # 每个问题文本
                "weak_primary": primary_label,  # 主意图
                "weak_secondary": secondary_label if secondary_label else "",  # 次意图（如果没有次意图则为空）
                "weak_confidence": 0.95,  # 置信度
                "weak_reason": "报告解读相关问题"  # 解释标注原因
            },
            "meta": {},
            "created_at": "2025-10-23T05:52:32.228343Z",
            "updated_at": "2025-10-23T05:52:32.228343Z",
            "inner_id": i,
            "total_annotations": 1,
            "cancelled_annotations": 0,
            "total_predictions": 0,
            "comment_count": 0,
            "unresolved_comment_count": 0,
            "last_comment_updated_at": None,
            "project": 1,
            "updated_by": None,
            "comment_authors": []
        }

        # 添加当前问题的任务数据
        labelstudio_data.append(task_data)

    # 将数据写入JSON文件
    with open(output_file_path, 'w', encoding='utf-8') as f:
        json.dump(labelstudio_data, f, ensure_ascii=False, indent=4)

    print(f"文件已成功转换并保存到 {output_file_path}")

# 示例调用
md_file_path = './ds/闲聊其他意图数据集 (500条).md'  # 输入Markdown文件路径
output_file_path = 'ds/chat.json'  # 输出Label Studio格式的JSON文件路径
primary_label = "闲聊其他"  # 主意图标签
secondary_label = None  # 次意图标签

md_to_labelstudio(md_file_path, output_file_path, primary_label, secondary_label)
