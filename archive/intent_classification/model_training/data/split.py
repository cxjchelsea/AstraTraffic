import json
import random

def split_json_dataset(input_file, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1):
    # 读取数据
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # 打乱顺序
    random.shuffle(data)

    total = len(data)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))

    # 按比例拆分
    train_data = data[:train_end]
    val_data = data[train_end:val_end]
    test_data = data[val_end:]

    # 保存到三个文件
    with open('intent/ds_fixed_async_train.json', 'w', encoding='utf-8') as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open('intent/ds_fixed_async_dev.json', 'w', encoding='utf-8') as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)
    with open('intent/ds_fixed_async_test.json', 'w', encoding='utf-8') as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    print(f"数据集拆分完成：共 {total} 条")
    print(f"训练集: {len(train_data)} 条，验证集: {len(val_data)} 条，测试集: {len(test_data)} 条")

if __name__ == "__main__":
    split_json_dataset("label_data/ds_fix_ls_tasks.json")
