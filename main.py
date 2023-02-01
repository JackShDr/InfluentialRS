### Server code 
import sys 
sys.path.append('/home/hzhual/.conda/envs/pytorch/lib/python3.6/site-packages')
### Import package
import torch
from prs import train_prsnn, evaluate_prsnn, evaluate_prob
from data_provider import DataProvider
from sweep_params import prs_sweep_config
from utils import str2bool, set_seed
from params import NetParams
import argparse

### Configuration
parser = argparse.ArgumentParser()
# Problem related
parser.add_argument("--use_wandb", type=str2bool, default = False)
parser.add_argument("--seed", type=int, default = 1234)
parser.add_argument("--project_name", type=str, default = "influential")
parser.add_argument('--dataset',type=str, default="lastfm_small")
parser.add_argument("--task", type=str, default="prs")
parser.add_argument("--method", type=str, default="prsnn")
parser.add_argument("--datapath", type=str, default='./data/')
parser.add_argument("--model_store_path", type=str, default='./save_model/')
parser.add_argument("--is_train", type=str2bool, default = True)
parser.add_argument("--is_eval", type=str2bool, default = False)

# Dataset related
parser.add_argument("--first", type=str2bool, default = False)
parser.add_argument("--load", type=str2bool, default = False)
parser.add_argument("--save", type=str2bool, default = True)
parser.add_argument("--max_len", type=int, default = 50)
parser.add_argument("--min_len", type=int, default = 20)
parser.add_argument("--step", type=int, default = -1)
parser.add_argument("--infrequent_thredshold", type=int, default=5)
parser.add_argument('--train_ratio',type=float, default=0.9)
#parser.add_argument("--random_target", type=str2bool, default=False)
# Model related
parser.add_argument('--batch_size',type=int, default=128)
parser.add_argument('--epoch',type=int, default=2000)
parser.add_argument('--emb_dim',type=int, default=40)
parser.add_argument('--u_emb_dim',type=int, default=10)
parser.add_argument('--n_layers',type=int, default=5)
parser.add_argument('--n_heads',type=int, default=4)
parser.add_argument('--ffn_dim',type=int, default=120)
parser.add_argument('--dropout',type=float, default=0.2)
parser.add_argument('--early_stop_patience',type=int, default=15)
parser.add_argument('--lr1',type=float, default=8e-3)
# Evaluation related
parser.add_argument('--top_k',type=int, default=20)
parser.add_argument('--max_path_len',type=int, default=20)
parser.add_argument('--gap_len',type=int, default=0)
parser.add_argument('--use_train',type=str2bool, default=True)
parser.add_argument('--use_h',type=str2bool, default=True)
parser.add_argument('--sample',type=str2bool, default=False)
parser.add_argument('--sample_k',type=int, default=3)
args = parser.parse_args()

# Evaluator Configuration
params = NetParams()
nn1_config = getattr(params, "params_nn1")

# Use wandb to track the model
if args.use_wandb == True:
    try:
        import wandb
        print("Use WandB to track the model...\n")
    except ImportError:
        print("WandB has not been installed...\n")
        args.use_wandb = False

# Transform the argparser config to a dictionary, which will be used by wandb later
if args.use_wandb == True:
    default_config = vars(args)

# Device control
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()
# Ensure deterministic behavior
set_seed(args.seed, cuda=torch.cuda.is_available())


def pipeline(config):
    dp = DataProvider(config, verbose = True)
    config.n_item = dp.n_item
    config.n_user = dp.n_user
        
    # Train nn1
    if config.is_train == True:
        train_prsnn(config, dp, device)
    
    # Test nn1
    evaluate_prsnn(config, dp, device)
    
    # Evaluate the path
    if config.is_eval == True:
        evaluate_prob(config, nn1_config, dp, device)
    
def prsnn_agent():
    """
    Pipeline for prs nn1. Include training, validation, and testing. 
    """
    if args.use_wandb == True:
        with wandb.init(project=args.project_name, config=default_config):
            config = wandb.config 
            pipeline(config)
    else:
        pipeline(args)

if __name__ == '__main__':
    if args.task == "sweep" and args.use_wandb == True:
        sweep_config = prs_sweep_config
        sweep_id = wandb.sweep(sweep_config, project=args.project_name)
        #sweep_id = "8hygl7oy"
        wandb.agent(sweep_id, prsnn_agent)
    elif args.task == "prs":
        prsnn_agent()


