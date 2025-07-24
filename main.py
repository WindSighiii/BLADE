import argparse
import importlib
from utils import set_color, load_config_files, show_args_info, setup_global_seed, \
    check_output_path, data_partition, data_preparation
import torch

def get_class_from_module(module_path, class_name):
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
def main():

    parser = argparse.ArgumentParser()

    # main args
    parser.add_argument('--dataset', default='QKV', type=str)
    parser.add_argument('--config_files', type=str, default='./config/', help='config yaml files')
    parser.add_argument("--do_eval", action="store_true")
    parser.add_argument('--model', type=str, default='model.model1_cross',
                    help='Path to the model module to import')
    parser.add_argument('--trainer', type=str, default='trainer_preBehavior',
                    help='Trainer module path to import the trainer from')
    # model args

    parser.add_argument("--no", type=int, default=1, help="model/process idenfier, e.g., 1, 2, 3...")
    parser.add_argument("--dropout", type=float, help="hidden dropout p for embedding layer and FFN")
    parser.add_argument("--num_heads", type=int, help="num of attention heads")
    parser.add_argument("--num_experts", type=int, help="num of experts")
    parser.add_argument("--ffn_acti", type=str, help="activation function of FFN")

    parser.add_argument("--num_blocks", type=int, help="num of blocks")
    parser.add_argument("--hidden_dims", type=int, help="hidden dims")
    
    parser.add_argument("--early_fusion_type", type=str, help="early fusion")
    parser.add_argument("--transformer_num_heads", type=int, help="num of attention heads")
    parser.add_argument("--transformer_num_blocks", type=int, help="num of blocks")
    parser.add_argument("--transformer_dropout", type=float, help="dropout")
    parser.add_argument("--trade_off", type=float, help="trade off")


    # train args
    parser.add_argument("--lr", type=float, default=0.001, help="learning rate of optimizer")
    parser.add_argument("--batch_size", type=int, default=1280, help="batch size of training phase")
    parser.add_argument("--seed", type=int, default=2023, help="global random seed for CUDA and pytorch")
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--max_epochs", type=int, help="max epochs")

    #loss args
    parser.add_argument("--contrast_type", type=str, default="Hybrid", help="contrast type")
    parser.add_argument("--rec_weight", type=float, default=1.0, help="rec weight")

    parser.add_argument("--cf_weight", type=float, default=0.1, help="cf weight")
    parser.add_argument("--temperature", type=float, default=1.0, help="temperature")

    #representation args
    parser.add_argument("--seq_representation_type", type=str, default="mean", help="seq representation type")
    parser.add_argument("--seq_representation_instancecl_type", type=str, default="concat", help="seq representation type")


    #augment_args
    parser.add_argument("--augment_type", type=str, default="Random", help="augmentation type")
    parser.add_argument("--click_prob", type=float, default=0.2, help="click probability")
    parser.add_argument("--mask_ratio", type=float, default=0.2, help="mask ratio")
    parser.add_argument("--time_prob", type=float, default=0.4, help="clip percentile")



    args = parser.parse_args()

    TrainerClass = get_class_from_module(args.trainer, 'BLADETrainer')

    model_module = get_class_from_module(args.model, 'BLADE')

    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda
    print(set_color('Using CUDA: ' + str(args.cuda_condition) + '\n', 'green'))

    config_dict = load_config_files(args.config_files, args)
    show_args_info(argparse.Namespace(**config_dict))

    setup_global_seed(config_dict['seed'])
    check_output_path(config_dict['output_dir'])

    data = data_partition(config_dict['dataset'])
    train_dataloader, valid_dataloader, test_dataloader, config_dict = data_preparation(config_dict, data)

    model = model_module(config=config_dict)
    trainer = TrainerClass(model, train_dataloader, valid_dataloader, test_dataloader, config_dict)

    if config_dict['do_eval']:
        trainer.load()
        _, test_info = trainer.test()
        print(set_color(f'\nFinal Test Metrics: ' +
                        test_info + '\n', 'pink'))
    else:
        trainer.train()
if __name__ == '__main__':
    main()