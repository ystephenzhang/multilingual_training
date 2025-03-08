import os, json, shutil, sys
import numpy as np
import torch
def deduplicate(neuron_target, neuron_delete):
    for set in neuron_target:
        for layer in neuron_target[set]:
            neuron_target[set][layer] = neuron_target[set][layer] - neuron_delete[set][layer]

    return neuron_target

def convert_numpy_types(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(v) for v in obj]
    elif isinstance(obj, np.int64):  # numpy int 类型转换为 Python int
        return int(obj)
    elif isinstance(obj, np.floating):  # numpy float 类型转换为 Python float
        return float(obj)
    else:
        return obj

def save_neuron(activate_neurons, path):
    for group in activate_neurons:
        entry = activate_neurons[group]
        activate_neurons[group] = {key: list(value) if isinstance(value, set) else value for key, value in entry.items()}
    with open(path, 'w') as f:
        json.dump(activate_neurons, f)

def read_neuron(path):
    with open(path, 'r') as f:
        data = json.load(f)
    for group in data:
        entry = data[group]
        data[group] = {key: set(value) if isinstance(value, list) else value for key, value in entry.items()}

    return data


def copy_file(src_path, dest_path):
    try:
        # 确保目标路径存在，如果 dest_path 是一个目录，则复制到该目录下
        if os.path.isdir(dest_path):
            dest_path = os.path.join(dest_path, os.path.basename(src_path))
        
        # 复制文件
        shutil.copy2(src_path, dest_path)  # copy2 保留原始文件的元数据
        print(f"文件已成功复制到: {dest_path}")
    except Exception as e:
        print(f"复制文件时出错: {e}")

def compare_pt_files(file1, file2, atol=1e-6, rtol=1e-5):
    """
    比较两个 .pt 文件中存储的张量是否相等
    
    :param file1: 第一个 .pt 文件路径
    :param file2: 第二个 .pt 文件路径
    :param atol: 绝对误差容限（默认 1e-6）
    :param rtol: 相对误差容限（默认 1e-5）
    :return: 若张量完全相同返回 True，否则返回 False，并打印不同的张量信息
    """
    # 加载 .pt 文件
    tensor1 = torch.load(file1)
    tensor2 = torch.load(file2)

    # 确保数据类型一致
    if isinstance(tensor1, dict) and isinstance(tensor2, dict):
        keys1, keys2 = set(tensor1.keys()), set(tensor2.keys())
        if keys1 != keys2:
            print("两个 .pt 文件的键不同：")
            print("file1 keys:", keys1)
            print("file2 keys:", keys2)
            return False

        # 遍历字典中的所有键，逐个比较张量
        for key in keys1:
            if not torch.allclose(tensor1[key], tensor2[key], atol=atol, rtol=rtol):
                print(f"张量 '{key}' 在两个 .pt 文件中不同！")
                return False
        return True

    elif isinstance(tensor1, torch.Tensor) and isinstance(tensor2, torch.Tensor):
        print("Loading all clear, comparing:")
        return torch.allclose(tensor1, tensor2, atol=atol, rtol=rtol)

    else:
        print("文件格式不同或不支持的类型")
        return False


if __name__ == "__main__":
    if sys.argv[1] == '0':
        copy_file('/home/progressgym/miniconda3/envs/SeaExam/lib/python3.9/site-packages/transformers/_backup/models/llama/modeling_llama.py', '/home/progressgym/miniconda3/envs/SeaExam/lib/python3.9/site-packages/transformers/models/llama/')
        copy_file('/home/progressgym/miniconda3/envs/SeaExam/lib/python3.9/site-packages/transformers/_backup/generation/utils.py', '/home/progressgym/miniconda3/envs/SeaExam/lib/python3.9/site-packages/transformers/generation/')
    elif sys.argv[1] == '1':
        copy_file('./transformers/modeling_llama.py', '/home/progressgym/miniconda3/envs/SeaExam/lib/python3.9/site-packages/transformers/models/llama/')
        copy_file('./transformers/utils.py', '/home/progressgym/miniconda3/envs/SeaExam/lib/python3.9/site-packages/transformers/generation/')
    elif sys.argv[1] == '2':
        copy_file('../multilingual_analysis/neuron_deactivate/modeling_llama.py', '/home/progressgym/miniconda3/envs/SeaExam/lib/python3.9/site-packages/transformers/models/llama/')
        copy_file('../multilingual_analysis/neuron_deactivate/utils.py', '/home/progressgym/miniconda3/envs/SeaExam/lib/python3.9/site-packages/transformers/generation/')
    else:
        copy_file('/home/progressgym/miniconda3/envs/SeaExam/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py', '../multilingual_analysis/neuron_deactivate/')
        copy_file('/home/progressgym/miniconda3/envs/SeaExam/lib/python3.9/site-packages/transformers/generation/utils.py', '../multilingual_analysis/neuron_deactivate/')