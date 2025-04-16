from src.detection import detect_key_neurons
from src.utils import replace_transformers_with_local
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
lang_set = ["english", "chinese", "french"]

def detection_all(model_name, lang, atten_num=4000, ffn_num=12000, test_size=-1, 
                  detection_path="./corpus_all", output_path="./output", 
                  suffix=""):
    tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only = True)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", local_files_only = True)
    for l in lang:
        print("Detecting neurons for", l)
        neurons = detect_key_neurons(model, tokenizer, l, atten_num=atten_num, ffn_num=ffn_num, test_size=test_size, detection_path=detection_path, output_path=output_path, suffix=suffix)
        print(l, "complete", len(neurons["attn_q"].keys()), len(neurons["attn_q"][0]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--corpus_path", type=str, default='./corpus_all/')
    parser.add_argument("--corpus_size", type=int, default=-1)
    parser.add_argument("--base", type=str, default="./models/base/Llama-3-8B")
    parser.add_argument("--output_path", type=str, default="./output/")
    parser.add_argument("--lang", type=int, default=1)
    parser.add_argument("--atten_num", type=int, default=4000)
    parser.add_argument("--ffn_num", type=int, default=12000)
    parser.add_argument("--suffix", type=str, default="")
    args = parser.parse_args()

    replace_transformers_with_local("./transformers3.9")

    detection_all(args.base, lang_set[:args.lang],
                  args.atten_num,
                  args.ffn_num,
                  args.corpus_size,
                  args.corpus_path,
                  args.output_path,
                  args.suffix)