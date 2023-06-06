import torch
from pipeline import pipeline
from model_params import irs_sweep_config, NetParams
from utils import str2bool, set_seed
import argparse

### Configuration
parser = argparse.ArgumentParser()
# Problem related
parser.add_argument("--use_wandb", type=str2bool, default = False)
parser.add_argument("--seed", type=int, default = 1234)
parser.add_argument("--project_name", type=str, default = "influential")
parser.add_argument('--dataset',type=str, default="ml-1m")
parser.add_argument("--task", type=str, default="prs")
parser.add_argument("--method", type=str, default="prsnn")
parser.add_argument("--datapath", type=str, default='./data/')
parser.add_argument("--model_store_path", type=str, default='./save_model/')
parser.add_argument("--is_train", type=str2bool, default = True)
parser.add_argument("--is_test", type=str2bool, default = True)
parser.add_argument("--is_eval", type=str2bool, default = True)

# Dataset related
parser.add_argument("--first", type=str2bool, default = False)
parser.add_argument("--load", type=str2bool, default = False)
parser.add_argument("--save", type=str2bool, default = True)
parser.add_argument("--max_len", type=int, default = 60)
parser.add_argument("--min_len", type=int, default = 20)
parser.add_argument("--step", type=int, default = -1)
parser.add_argument("--infrequent_thredshold", type=int, default=5)
parser.add_argument('--train_ratio',type=float, default=0.9)
#parser.add_argument("--random_target", type=str2bool, default=False)

# Model related
parser.add_argument('--batch_size',type=int, default=128)
parser.add_argument('--epoch',type=int, default=200)
parser.add_argument('--emb_dim',type=int, default=30)
parser.add_argument('--u_emb_dim',type=int, default=10)
parser.add_argument('--n_layers',type=int, default=6)
parser.add_argument('--n_heads',type=int, default=6)
parser.add_argument('--ffn_dim',type=int, default=256)
parser.add_argument('--dropout',type=float, default=0.05)
parser.add_argument('--early_stop_patience',type=int, default=10)
parser.add_argument('--lr1',type=float, default=3e-3)
parser.add_argument('--w_h',type=float, default=0.05)
parser.add_argument('--w_obj',type=float, default=1)

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
# Here we use SampleNet as an example
params = NetParams()
tran_config = getattr(params, "params_tran")

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

# Agent for IRN
# * Control whether to use wandb to track the model 
def irn_agent():
    if args.use_wandb == True:
        with wandb.init(project=args.project_name, config=default_config):
            config = wandb.config 
            pipeline(config, tran_config, device=device)
    else:
        pipeline(args, tran_config, device=device)

if __name__ == '__main__':
    if args.task == "sweep" and args.use_wandb == True:
        sweep_config = irs_sweep_config
        sweep_id = wandb.sweep(sweep_config, project=args.project_name)
        wandb.agent(sweep_id, irn_agent)
    else: # args.task == normal
        irn_agent()



