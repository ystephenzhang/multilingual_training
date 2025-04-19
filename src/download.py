from datasets import load_dataset
from itertools import islice
import json
from tqdm import tqdm
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

if __name__ == "__main__":
	lan = ["zh", "fr", "de", "th", "sw"]
	for l in lan:
		download_culturax(l)
		download_madlad(l)
		download_ccnews(l)
		download_wiki(l)
