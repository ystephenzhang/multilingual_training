import os, json, shutil, sys, itertools, re
import numpy as np
import torch
import random
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import importlib
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

def read_neuron(path, top_k = -1):
    with open(path, 'r') as f:
        data = json.load(f)
    for group in data:
        entry = data[group]
        data[group] = {key: set(value) if isinstance(value, list) else value for key, value in entry.items()}
        if top_k > 0:
            #data[group] = random.sample(data[group], min(top_k, len(data[group])))
            data[group] = itertools.islice(data[group], min(top_k, len(data[group])))

    return data

def get_hf_checkpoints(folder):
    # 获取所有 checkpoint 目录
    checkpoints = [d for d in os.listdir(folder) if re.match(r'checkpoint-\d+', d)]
    
    if not checkpoints:
        return [folder]

    #latest_checkpoint = max(checkpoints, key=lambda x: int(re.search(r'\d+', x).group()))
    
    return [os.path.join(folder, c) for c in checkpoints]

def get_swift_checkpoints(output_dir):
    """
    找出输出目录中 vX-时间戳 的最新版本文件夹，
    并返回该文件夹中所有 checkpoint-* 子目录的完整路径列表。
    """
    version_pattern = re.compile(r"v(\d+)-\d{8}-\d{6}")
    version_folders = []

    # 遍历 output_dir 中所有子目录，提取符合版本命名规则的
    for name in os.listdir(output_dir):
        full_path = os.path.join(output_dir, name)
        if os.path.isdir(full_path):
            match = version_pattern.fullmatch(name)
            if match:
                version_num = int(match.group(1))
                version_folders.append((version_num, full_path))

    if not version_folders:
        raise ValueError("未找到任何 vX-时间戳 格式的子目录")

    # 取出最大的 vX 作为最新版本
    latest_version_path = max(version_folders, key=lambda x: x[0])[1]

    # 查找该目录下所有 checkpoint-* 子文件夹
    checkpoint_paths = []
    for subname in os.listdir(latest_version_path):
        subpath = os.path.join(latest_version_path, subname)
        if os.path.isdir(subpath) and subname.startswith("checkpoint-"):
            checkpoint_paths.append(subpath)

    checkpoint_paths.sort()  # 可选：排序一下

    return checkpoint_paths

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

def compare_models(model1, model2):
    for (name1, param1), (name2, param2) in zip(model1.named_parameters(), model2.named_parameters()):
        if not torch.equal(param1, param2):
            print(f"参数不同: {name1} vs {name2}")
            return False
    return True

def count_english_ratio(text):
    english_letters = [ch for ch in text if ch.isalpha() and ch.isascii()]
    return len(english_letters) / len(text)

def count_chinese_ratio(text):
    chinese_chars = [char for char in text if '\u4e00' <= char <= '\u9fff']
    return len(chinese_chars) / len(text)

def load_model_from_name(model_name):
    model_path = get_latest_checkpoint(model_name)
    print("Loading model: ", model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", local_files_only = True)

    #tokenizer.padding_side = "left"
    #tokenizer.truncation_side = "left"
    tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token == None else tokenizer.pad_token
    tokenizer.pad_token_id = tokenizer.eos_token_id if tokenizer.pad_token_id == None else tokenizer.pad_token_id
    model.config.pad_token_id = model.config.eos_token_id if model.config.pad_token_id == None else model.config.pad_token_id
    return model, tokenizer

import random

def sample_large_file(input_path, output_path, sample_size=150000, seed=42):
    random.seed(seed)
    reservoir = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i < sample_size:
                reservoir.append(line)
            else:
                # 随机替换已有样本
                j = random.randint(0, i)
                if j < sample_size:
                    reservoir[j] = line

    with open(output_path, 'w', encoding='utf-8') as f_out:
        f_out.writelines(reservoir)

    print(f"✅ 完成，从 {input_path} 随机采样 {sample_size} 行保存到 {output_path}")

def plot_line(array, title="折线图", xlabel="Layer", ylabel="Log-10 Value"):
    """
    画出一维数组的折线图

    参数:
        array (list or 1D np.ndarray): 一维数组或列表
        title (str): 图表标题
        xlabel (str): x轴标签
        ylabel (str): y轴标签
    """
    plt.figure(figsize=(8, 4))
    plt.plot(array, marker='o', linestyle='-', color='blue')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('./output/hidden_state_log/' + title + 'png')

def plot_two_lines(arr1, arr2, label1="Model_1", label2="Model_2", title="双折线图", xlabel="Layer", ylabel="Log-10 Value"):
    """
    在一个图中画两个一维数组的折线图

    参数:
        arr1, arr2: 两个一维数组（list 或 np.ndarray）
        label1, label2: 两条线的标签（用于图例）
        title: 图标题
        xlabel, ylabel: 坐标轴标签
    """
    plt.figure(figsize=(8, 4))
    plt.plot(arr1, marker='o', linestyle='-', label=label1)
    plt.plot(arr2, marker='s', linestyle='--', label=label2)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend()  # 添加图例
    plt.tight_layout()
    plt.savefig('./output/hidden_state_log/' + title + 'png')

def shorten_param_type(param_type: str) -> str:
    """
    将类似 "self_attn.q_proj.weight" 的字符串简化为 "q_proj"
    可根据实际需求灵活修改
    """
    # 先去掉 "self_attn."、"mlp." 等前缀
    param_type = param_type.replace("self_attn.", "")
    param_type = param_type.replace("mlp.", "")
    # 再去掉 ".weight"、".bias" 等后缀
    param_type = param_type.replace(".weight", "")
    param_type = param_type.replace(".bias", "")
    return param_type

def plot_param_heatmap(param_dict, title="Ratio of Trained Params Heatmap"):
    """
    根据输入字典画热力图，解决拥挤问题：
      - 对参数类型做简写
      - 增大图像尺寸
      - 调整注释文字大小等

    输入字典的键形如:
        "model.layers.3.self_attn.k_proj.weight"
    要求:
      - 每行表示一个简写后的参数类型 (如 "k_proj")
      - 每列表示一个 layer (如 "3")
      - 单元格颜色表示该键对应的数值大小
    """

    data = {}
    pattern = re.compile(r"layers\.(\d+)\.(.+)")  # 匹配: layers.x.y.z.w
    
    for key, value in param_dict.items():
        m = pattern.search(key)
        if m:
            layer = m.group(1)           # x：层号 (字符串)
            param_type_full = m.group(2) # y.z.w：完整参数类型
            # 简写
            short_name = shorten_param_type(param_type_full)

            if short_name not in data:
                data[short_name] = {}
            data[short_name][layer] = value

    # 收集所有层号，并按数字排序
    all_layers = set()
    for short_pt, layer_values in data.items():
        all_layers.update(layer_values.keys())
    all_layers = sorted(all_layers, key=lambda x: int(x))  # 按数字排序

    # 构造 DataFrame: 行=参数类型, 列=layer
    df = pd.DataFrame(data).T  # 行:参数类型, 列:layer(字符串)
    df = df.reindex(columns=all_layers)  # 重新按层排序
    # 你也可以填充 NaN
    # df = df.fillna(0)  # 若需要将缺失值填0，可取消注释

    # 设置 Seaborn 样式
    sns.set(style="whitegrid", font_scale=1.0)  # 可根据需要调大或调小

    # 画图
    plt.figure(figsize=(12, 6))  # 调整图像尺寸 (宽, 高)
    heatmap = sns.heatmap(
        df, 
        annot=True,            # 在格子里显示数值
        fmt=".2f",            # 数值格式
        cmap="viridis",       # 颜色映射
        cbar=True, 
        annot_kws={"fontsize": 8},   # 注释字体大小
        cbar_kws={"shrink": 0.8}     # 缩小颜色条
    )

    # 设置 x/y 轴标签
    plt.xlabel("Layer", fontsize=12)
    plt.ylabel("Parameter Type", fontsize=12)
    plt.title(title, fontsize=14, pad=12)

    # 调整 x 轴标签角度，避免重叠 (对于多层可用 90°)
    plt.xticks(rotation=0)
    # plt.xticks(rotation=90)  # 若层数很多，可试试 90 度

    # 调整布局，避免标签被裁剪
    plt.tight_layout()
    plt.savefig("./output/process_log/" + title + '.png')

def inspect_params(log, model_name):
    with open(log, 'r') as f:
        data = json.load(f)
    grad_data = {}
    params_of_concern = ["self_attn", "up_proj"]
    for key in data[0]:
        if any([x in key for x in params_of_concern]):
            grad_data[key] = data[0][key][1]
    plot_param_heatmap(grad_data, title=model_name)

def retrieve_trainer_from_conda(conda, destination='./transformers', version="3.9"):
    print('Fetching')
    copy_file(f'/home/zhangyang/miniconda3/envs/{conda}/lib/python{version}/site-packages/transformers/models/llama/modeling_llama.py', destination + version)
    copy_file(f'/home/zhangyang/miniconda3/envs/{conda}/lib/python{version}/site-packages/transformers/generation/utils.py', destination + version)
    copy_file(f'/home/zhangyang/miniconda3/envs/{conda}/lib/python{version}/site-packages/transformers/trainer.py', destination + version)

def send_trainer_to_conda(conda, source="./transformers", version="3.9"):
    print(f'Sending from {source}')
    print(f'/home/zhangyang/miniconda3/envs/{conda}/lib/python{version}/site-packages/transformers/models/llama/')
    copy_file(f'{source}{version}/modeling_llama.py', f'/home/zhangyang/miniconda3/envs/{conda}/lib/python{version}/site-packages/transformers/models/llama/')
    copy_file(f'{source}{version}/utils.py', f'/home/zhangyang/miniconda3/envs/{conda}/lib/python{version}/site-packages/transformers/generation/')
    copy_file(f'{source}{version}/trainer.py', f'/home/zhangyang/miniconda3/envs/{conda}/lib/python{version}/site-packages/transformers/') 

def retrieve_swift_from_conda(conda="swift", destination="./swift"):
    print("Fetching")
    copy_file(f"/home/zhangyang/miniconda3/envs/{conda}/lib/python3.10/site-packages/swift/llm/argument/train_args.py", destination)
    copy_file(f"/home/zhangyang/miniconda3/envs/{conda}/lib/python3.10/site-packages/swift/llm/train/sft.py", destination)

def send_swift_to_conda(conda="swift", source="./swift"):
    print(f'Sending from {source}')
    copy_file(f'{source}/train_args.py', f'/home/zhangyang/miniconda3/envs/{conda}/lib/python3.9/site-packages/swift/llm/argument/')
    copy_file(f'{source}/sft.py', f'/home/zhangyang/miniconda3/envs/{conda}/lib/python3.9/site-packages/swift/llm/train/')


def replace_transformers_with_local(local_transformers_path="./transformers", target_lib="transformers"):
    # 找到 transformers 包的安装路径
    spec = importlib.util.find_spec(target_lib)
    if spec is None or spec.origin is None:
        raise ImportError("transformers package is not installed or cannot be found.")
    
    # 得到安装路径
    transformers_install_path = os.path.dirname(spec.origin)
    print(f"Detected transformers install path: {transformers_install_path}")
    
    # 确保本地路径存在
    if not os.path.isdir(local_transformers_path):
        raise FileNotFoundError(f"Local path '{local_transformers_path}' not found.")
    
    # 遍历本地目录，复制替换文件
    for root, dirs, files in os.walk(local_transformers_path):
        rel_path = os.path.relpath(root, local_transformers_path)
        target_dir = os.path.join(transformers_install_path, rel_path)

        # 确保目标目录存在
        os.makedirs(target_dir, exist_ok=True)

        for file in files:
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_dir, file)
            shutil.copy2(src_file, dst_file)
            print(f"Copied: {src_file} -> {dst_file}")

    print("transformers package has been replaced with local version.")

if __name__ == "__main__":
    if sys.argv[1] == '0':
        print('Restoring from backup')
        copy_file('/home/zhangyang/miniconda3/envs/seaexam/lib/python3.9/site-packages/transformers/_backup/models/llama/modeling_llama.py', '/home/zhangyang/miniconda3/envs/seaexam/lib/python3.9/site-packages/transformers/models/llama/')
        copy_file('/home/zhangyang/miniconda3/envs/seasxam/lib/python3.9/site-packages/transformers/_backup/generation/utils.py', '/home/zhangyang/miniconda3/envs/seaexam/lib/python3.9/site-packages/transformers/generation/')
    elif sys.argv[1] == '1':
        '''
        print('Sending from training')
        copy_file('./transformers/modeling_llama.py', '/home/zhangyang/miniconda3/envs/seaexam/lib/python3.9/site-packages/transformers/models/llama/')
        copy_file('./transformers/utils.py', '/home/zhangyang/miniconda3/envs/seaexam/lib/python3.9/site-packages/transformers/generation/')
        copy_file('./transformers/trainer.py', '/home/zhangyang/miniconda3/envs/seaexam/lib/python3.9/site-packages/transformers/')
        '''
        send_trainer_to_conda(sys.argv[2], version=sys.argv[3])
    elif sys.argv[1] == '2':
        retrieve_trainer_from_conda(sys.argv[2], version=sys.argv[3])
    elif sys.argv[1] == '3':
        send_swift_to_conda(sys.argv[2])
    elif sys.argv[1] == '4':
        retrieve_swift_from_conda(sys.argv[2])
    
    '''
        print('Sending from analysis')
        copy_file('../multilingual_analysis/neuron_deactivate/modeling_llama.py', '/home/zhangyang/miniconda3/envs/seaexam/lib/python3.9/site-packages/transformers/models/llama/')
        copy_file('../multilingual_analysis/neuron_deactivate/utils.py', '/home/zhangyang/miniconda3/envs/seaexam/lib/python3.9/site-packages/transformers/generation/')
    else:
        print('Fetching')
        copy_file('/home/zhangyang/miniconda3/envs/seaexam/lib/python3.9/site-packages/transformers/models/llama/modeling_llama.py', './transformers/')
        copy_file('/home/zhangyang/miniconda3/envs/seaexam/lib/python3.9/site-packages/transformers/generation/utils.py', './transformers/')
        copy_file('/home/zhangyang/miniconda3/envs/seaexam/lib/python3.9/site-packages/transformers/trainer.py', './transformers/')
    '''