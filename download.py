from datasets import load_dataset
from itertools import islice
import json
from tqdm import tqdm
from pathlib import Path
# culturax, madlad, cc-news, wiki

def download_oscar(lang, lines=150000):
	dataset = load_dataset(
		"oscar-corpus/OSCAR-2301",
		lang,
		split="train",
		streaming=True)
	with open("./corpus_all/" + lang + ".txt", "w") as f:
		for i, sample in enumerate(islice(dataset, lines)):
			line = sample["text"].strip().replace("\n", " ")  # 清洗换行符
			f.write(line + "\n")
			if (i + 1) % 1000 == 0:
				print(f"Saved {i + 1} samples...")

def download_culturax(lang, lines=150000, output="./pretrain_tokens/"):
	dataset = load_dataset("uonlp/CulturaX", lang, split="train", streaming=True)

	with open(output + lang + "_culturax.jsonl", "w", encoding="utf-8") as f:
		for i, example in tqdm(enumerate(dataset)):
			if i >= lines:
				break
			json_line = json.dumps(example, ensure_ascii=False)
			f.write(json_line + "\n")

def download_madlad(lang, lines=150000, output="./pretrain_tokens/"):
	dataset = load_dataset("allenai/madlad-400", lang, split="clean", streaming=True)

	with open(output + lang + "_madlad.jsonl", "w", encoding="utf-8") as f:
		for i, example in tqdm(enumerate(dataset)):
			if i >= lines:
				break
			json_line = json.dumps({"text":example["text"]}, ensure_ascii=False)
			f.write(json_line + "\n")

def download_ccnews(lang, lines=150000, output="./pretrain_tokens/"):
	dataset = load_dataset("stanford-oval/ccnews", name="2016")
	count = 0
	with open(output + lang + "_ccnews.jsonl", "w", encoding="utf-8") as f:
		for example in tqdm(dataset["train"]):
			print(example.get("language", None))
			if example.get("language", None) == lang:
				json_line = json.dumps(example["plain_text"], ensure_ascii=False)
				f.write(json_line + "\n")
				count += 1
				if count >= lines:
					break

def download_wiki(lang, lines=150000, output="./pretrain_tokens/"):
	dataset = load_dataset("wikimedia/wikipedia", "20231101." + lang, split="train", streaming=True)

	with open(output + lang + "_wiki.jsonl", "w", encoding="utf-8") as f:
		for i, example in tqdm(enumerate(dataset)):
			if i >= lines:
				break
			json_line = json.dumps({"text":example["text"]}, ensure_ascii=False)
			f.write(json_line + "\n")

def merge(dir_path, prefix, output_file):
	dir_path = Path(dir_path)
	input_files = sorted(dir_path.glob(f"{prefix}*.jsonl"))  # 自动匹配

	if not input_files:
		print(f"⚠️ 没有找到以 '{prefix}' 开头的 .jsonl 文件")
		return

	count = 0
	with open(output_file, 'w', encoding='utf-8') as fout:
		for file_path in input_files:
			print(f"📥 正在合并: {file_path.name}")
			with open(file_path, 'r', encoding='utf-8') as fin:
				for line in fin:
					line = line.strip()
					if not line:
						continue
					try:
						json.loads(line)  # 验证是合法 JSON
						fout.write(line + '\n')
						count += 1
					except json.JSONDecodeError as e:
						print(f"❗ 跳过非法 JSON 行: {e} in {file_path.name}") 

if __name__ == "__main__":
	lan = ["fr", "de", "th", "sw"]
	for l in lan:
		if l != "fr":
			download_culturax(l)
		download_madlad(l)
		#download_ccnews(l)
		download_wiki(l)
