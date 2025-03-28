import itertools

def get_dataset(lang):
    from datasets import load_dataset
    for l in lang:
        output_file = "./assets/" + l + ".txt"

        '''with open(output_file, "w", encoding="utf-8") as f:
            print("Resetted", output_file)'''
            
        num_lines = 1000000
        dataset = load_dataset("wikimedia/wikipedia",
                                "20231101." + l, 
                                streaming=True, # optional
                    trust_remote_code=True,
                                split="train")     
        print("loaded")
        with open(output_file, "a", encoding="utf-8") as f:
            for sample in itertools.islice(dataset, num_lines):  # 只取前 10,000 行
                print("written", sample["id"])
                if "text" in sample:  # 确保 "text" 字段存在
                    f.write(sample["text"].replace('\n',' ') + "\n")

if __name__ == "__main__":
    get_dataset(["zh", "fr", "de", "sw", "th"]) 