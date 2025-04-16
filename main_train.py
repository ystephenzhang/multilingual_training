import argparse
from src.scripts import *
def main_reverse_experiment(language, base_model, train_data_path, test_data_path, output_path, 
                            evaluation_mode, log_eval, evaluation_set,
                            training_mode, training_args):
    for l in language:
        reverse_experiment(base_model, "english", l, 
                           training_args, training_mode=training_mode, 
                           eval_method=evaluation_mode, eval_dataset=evaluation_set,
                           full_record=log_eval, train_data_path=train_data_path, force_retrain=True,
                           test_data_path=test_data_path, output_path=output_path)
lang_set = ["zh", "sw", "fr", "de", "th"]
eval_set = ["mmlu", "gsm", "ppl"]
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Training
    parser.add_argument("--e_step", type=int, default=500)
    parser.add_argument("--s_step", type=int, default=500)
    parser.add_argument("--num_device", type=int, default=8)
    parser.add_argument("--log_grad", type=bool, default=False)
    parser.add_argument("--b_size", type=int, default=4)
    parser.add_argument("--g_acc", type=int, default=2)
    parser.add_argument("--max_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=4e-6)
    parser.add_argument("--deepspeed", type=str, default="zero3")

    # Setting
    parser.add_argument("--base", type=str, default="./models/base/Llama-3-8B")
    parser.add_argument("--train_data_path", type=str, default="./assets/")
    parser.add_argument("--test_data_path", type=str, default="./test_data/")
    parser.add_argument("--output_path", type=str, default="./models/trained/")
    parser.add_argument("--activate_path", type=str, default="./output/Llama-3-8B_english.json")
    parser.add_argument("--activate_layers", type=str, default="all")
    parser.add_argument("--activate_types", type=str, default="all")
    parser.add_argument("--lang", type=int, default=5)
    parser.add_argument("--eval_sets", type=int, default=3)
    parser.add_argument("--log_eval", type=bool, default=False)
    args = parser.parse_args()

    # Logging
    
    lang = lang_set[:args.lang] 
    eval = eval_set[:args.eval_sets]
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
        "max_len":args.max_len,
        "activate_layers":args.activate_layers,
        "activate_path": args.activate_path,
        "activate_types": args.activate_types
    }
    
    main_reverse_experiment(
        lang,
        args.base,
        args.train_data_path,
        args.test_data_path,
        args.output_path,
        evaluation_mode="parallel",
        log_eval=args.log_eval,
        evaluation_set=eval,
        training_mode="swift",
        training_args=training_args
    )
    