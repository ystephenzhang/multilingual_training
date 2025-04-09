import argparse
from src.scripts import *
def main_reverse_experiment(language, base_model, data_path, output_path, evaluation_mode, training_mode, training_args):
    for l in language:
        reverse_experiment(base_model, "english", l, 
                           training_args, training_mode=training_mode, eval_method=evaluation_mode, data_path=data_path,
                           output_path=output_path)
lang_set = ["zh", "sw", "fr", "de", "th"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--e_step", type=int, default=500)
    parser.add_argument("--s_step", type=int, default=500)
    parser.add_argument("--num_device", type=int, default=8)
    parser.add_argument("--log_grad", type=bool, default=True)
    parser.add_argument("--b_size", type=int, default=16)
    parser.add_argument("--g_acc", type=int, default=4)
    parser.add_argument("--max_len", type=int, default=8192)
    parser.add_argument("--lr", type=float, default=2e-6)
    parser.add_argument("--deepspeed", type=str, default="zero3")

    # Setting
    parser.add_argument("--base", type=str, default="./models/base/Llama-3-8B")
    parser.add_argument("--data_path", type=str, default="./models/base/Llama-3-8B")
    parser.add_argument("--output_path", type=str, default="./models/trained/")
    parser.add_argument("--lang", type=int, default=5)
    args = parser.parse_args()
    
    lang = lang_set[:args.lang] 
    training_args = {
        "n":args.num_device,
        "n_devices":",".join(map(str, range(args.num_device))),
        "log_grad":args.log_grad,
        "b_size":args.b_size,
        "lr":args.lr,
        "g_acc":args.g_acc,
        "e_step":args.e_step,
        "s_step":args.s_step,
        "deepspeed":args.deepspeed,
        "max_len":args.maxlen
    }
    
    main_reverse_experiment(
        lang,
        args.base,
        args.data_path,
        args.output_path,
        evaluation_mode="parallel",
        training_mode="swift",
        training_args=training_args
    )
    