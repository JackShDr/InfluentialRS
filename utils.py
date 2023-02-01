import torch
import numpy as np
import pandas as pd
import random
from sklearn.preprocessing import Normalizer


def delete_item_in_history(tensor, indices, h=50):
    """
    Remove items in the indices from the tensor
    """
    return tensor[~tensor.unsqueeze(1).eq(indices[-h:]).any(1)]

def set_seed(seed, cuda=False):
    """
    Set the random seed
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    if cuda:
        torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    
def assert_no_grad(variable):
    if variable.requires_grad:
        raise ValueError(
            "nn criterions don't compute the gradient w.r.t. targets - please "
            "mark these variables as volatile or not requiring gradients"
        )
        
def str2bool(v):
    return v.lower() in ('true')


def get_end_index(seq, pad = 0):
    """
    Obtain the end position of sequence, given post-padding.
    """
    pos = np.where(seq == pad)[0]
    if len(pos) == 0:
        return len(seq) - 1
    else:
        return pos[0] - 1
    
def get_start_index(seq, pad = 0):
    """
    Obtain the start position of sequence, given pre-padding. 
    The sequence cannot be all padding.
    """
    return np.where(seq != pad)[0][0] 

def get_item_index(seq, item):
    """
    Obtain the position of an item in a sequence.
    """
    pos = np.where(seq == item)[0]
    if len(pos) == 0:
        return -1
    else:
        return pos[0]

class EarlyStopping:
    """
    Early stops the training if validation loss doesn't improve after a given patience.
    """
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, save_path):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
        elif score < self.best_score:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}'
                )
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, save_path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, save_path):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...'
            )
        # torch.save(
        #     model, save_path + "/" +
        #     "checkpoint_{}_{:.6f}.pth.tar".format(epoch, val_loss))
        if model != None:
            torch.save(model, save_path)
        self.val_loss_min = val_loss

def visual_nn1(input_seqs, derived_seqs, targets, id_dict, text_data):
    display_seqs = []
    n_success = 0
    tags = np.array(text_data.columns[5:])
    
    for i in range(len(derived_seqs)):
        # Obtain the sequence
        d_seq = derived_seqs[i]
        input_seq = input_seqs[i]
        target = targets[i]
            
        # Find end position of the history
        h_end = np.where(input_seq==0)[0]
        if len(h_end) > 0:
            h_end = h_end[0] - 1
        else:
            h_end = len(input_seq)-1
        # Get the history
        history = input_seq[:h_end+1]
                     
        # Find the end position of path in the derived sequence
        pos = np.where(d_seq == target)[0]
        if len(pos) > 0:
            path = d_seq[:pos[0]+1]
            n_success += 1
        else:
            path = np.concatenate([d_seq, np.array([target])])
        
        # Get the text of the history and path
        history_id = [id_dict[j] for j in history]
        history = [text_data.at[j - 1, "name"] for j in history_id]
        path_id = [id_dict[j] for j in path]
        path = [text_data.at[j - 1, "name"] for j in path_id]
        # The tag will contain the last three history and the path
        genre_path_id = history_id[-3:] + path_id 
        genre = [list(tags[np.where(text_data.iloc[j-1][tags].to_numpy().astype(int)==1)]) for j in genre_path_id if j != 0]
        length = len(path)
        
        # Put the result together
        display_seqs.append([history, path, genre, length])
        
    print(n_success)
    return display_seqs

def get_feature_vector(dataset, datapath, use_fv = True):
    binary = True
    if dataset == "ml-100k":
        print("Using feature vector...\n")
        id_dict = np.load(datapath + dataset + "/item_reverse_dict.npy", allow_pickle=True).item()
        names = ['itemId' , 'name', 'date', 'video release date', 
              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
              'Children' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western']
        text_data =  pd.read_csv(datapath + dataset + "/u.item", sep='|',encoding='unicode_escape',names = names)
        tags = np.array(text_data.columns[5:])
        feature_dict = {key:text_data.iloc[value-1][tags].to_numpy().astype(int) for key, value in id_dict.items()}
    elif dataset ==  "ml-1m" and use_fv == True:
        print("Using feature vector...\n")
        id_dict = np.load(datapath + dataset + "/item_reverse_dict.npy", allow_pickle=True).item()
        names = ['Action' , 'Adventure' , 'Animation' ,
              "Children's" , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
              'Thriller' , 'War' , 'Western']
        str2num = {names[i]:i for i in range(len(names))}
        text_data =  pd.read_csv(datapath + dataset + "/movies.dat", sep="::", header=None, engine="python").set_index(0)
        def to_onehot(str_list, str2num):
            vec = np.zeros(len(str2num))
            for name in str_list:
                vec[str2num[name]] = 1
            return vec
        feature_dict = {key:to_onehot(text_data.loc[value][2].split('|'),str2num).astype(int) for key, value in id_dict.items()}            
    elif dataset == "lastfm":
        fv = np.load(datapath + dataset + "/fv.npy")
        feature_dict = {i+1:fv[i+1] for i in range(len(fv)-1)}
        binary = False        
    else:
        print("Using Embedding...\n")
        fv = np.load(datapath + dataset + "/fv.npy")
#        transformer = Normalizer().fit(fv)
#        fv = transformer.transform(fv)
        
#        print(fv)
        feature_dict = {i+1:fv[i+1] for i in range(len(fv)-1)}
        binary = False
    return feature_dict, binary

def cal_fv_dist(i, j, fv_dict, binary=False):
    if binary == True:
        return np.count_nonzero(fv_dict[int(i)]!=fv_dict[int(j)]) 
    else:
        return np.linalg.norm(fv_dict[int(i)]-fv_dict[int(j)])

### Test the visualization
if __name__ == "__main__":
#    id_dict = np.load("./data/ml-1m/item_reverse_dict.npy", allow_pickle=True).item()
#    names = ['itemId' , 'name', 'date', 'video release date', 
#              'IMDb URL' , 'unknown' , 'Action' , 'Adventure' , 'Animation' ,
#              'Children' , 'Comedy' , 'Crime' , 'Documentary' , 'Drama' , 'Fantasy' ,
#              'Film-Noir' , 'Horror' , 'Musical' , 'Mystery' , 'Romance' , 'Sci-Fi' ,
#              'Thriller' , 'War' , 'Western']

#    text_data =  pd.read_csv("./data/ml-100k/u.item", sep='|',encoding='unicode_escape',names = names)
    
#    display_seqs = visual_nn1(input_seqs, derived_seqs, targets, id_dict, text_data)
#    np.save(result_save_dir + "r_display_seqs1.npy", display_seqs)


##    np.save(result_save_dir + "r_display_seqs.npy", display_seqs)
    fv_dict, binary = get_feature_vector("lastfm_small", "./data/")


