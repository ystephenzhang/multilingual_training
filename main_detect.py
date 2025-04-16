from src.scripts import detection_all
from src.utils import *
import argparse
lang_set = ["english", "chinese", "french"]
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