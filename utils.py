import os
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler, RobustScaler
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torch.nn.functional as F

from args import get_parser
parser = get_parser()
args = parser.parse_args()

def normalize_data(data, scaler=None):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)

    if scaler is None:
        scaler = MinMaxScaler()
        scaler.fit(data)
    data = scaler.transform(data)

    return data, scaler

def z_score_standardization(data):
    data = np.asarray(data, dtype=np.float32)
    if np.any(sum(np.isnan(data))):
        data = np.nan_to_num(data)
    mean = np.mean(data)  
    std = np.std(data)    
    
    standardized_data = (data - mean) / std
    return standardized_data

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window

def normalize_anomaly_scores(scores):

    normalized_scores = (scores - np.min(scores)) / (np.max(scores) - np.min(scores))
    return normalized_scores

def merge_pkl_files(file_list, directory):
    data_list = np.array([])
    for file in file_list:
        file_path = os.path.join(directory, file)
        with open(file_path, "rb") as f:
            data = pickle.load(f)
                   
        if data_list.size == 0:
            data_list = np.array(data)
        else:
            data_list = np.concatenate((data_list, np.array(data)), axis=0)
            
    return data_list


def get_data(path, dataset, batch_size, max_train_size=None, max_test_size=None,
             normalize=False, spec_res=False, train_start=0, test_start=0, group=None):

    root_cause_labels = None
    if dataset == "SMD":

        files = os.listdir(path)
        grouped_files = {"train": [], "test": [], "labels": []}
        for file in files:
            if group != None:
                if group not in file:
                    continue
            if "train" in file:
                grouped_files["train"].append(file)
            elif "test" in file:
                grouped_files["test"].append(file)
            elif "labels" in file:
                grouped_files["labels"].append(file)

        merged_data = {}
        for kind, files in grouped_files.items():
            merged_data[kind] = merge_pkl_files(files, path)
        train_data = merged_data['train']
        test_data = merged_data['test']
        test_label = merged_data['labels']
        print(f"{dataset}, {train_data.shape}")
        print(f"{dataset}, {test_data.shape}")

        root_cause_labels = np.zeros(test_data.shape)
        files = os.listdir(os.path.join(path, 'interpretation_label'))
        for filename in files:
            if group != None:
                if group not in filename:
                    continue
            with open(os.path.join(path, 'interpretation_label', filename), "r") as f:
                ls = f.readlines()
            for line in ls:
                pos, values = line.split(':')[0], line.split(':')[1].split(',')
                start, end, indx = int(pos.split('-')[0]), int(pos.split('-')[1]), [int(i)-1 for i in values]
                root_cause_labels[start-1:end-1, indx] = 1

    elif dataset in ["SWaT", "WADI", "PSM", 'creditcard', 'GECCO', 'HAI']:
        if dataset != 'PSM':
            train_data = pd.read_csv(path+f"/downsampled_{dataset}_train.csv", index_col=0)
            test_data = pd.read_csv(path+f"/downsampled_{dataset}_test.csv", index_col=0)
            print(f"{dataset}, {train_data.shape}")
            print(f"{dataset}, {test_data.shape}")
            train, test = train_data, test_data
        else:
            train_data = pd.read_csv(path+f"/train.csv", index_col=0)
            test_data = pd.read_csv(path+f"/test.csv", index_col=0)
            print(f"{dataset}, {train_data.shape}")
            print(f"{dataset}, {test_data.shape}")
            train, test = train_data, test_data
            train = train.fillna(train.mean())
            test = test.fillna(test.mean())
        
        test_label = test_data.iloc[:, -1]
        test_data = test_data.iloc[:, :-1]
        test_data.columns = [''] * len(test_data.columns)

    elif dataset in ['SMAP', 'MSL']:
        train_data = pd.read_csv(path+f"/{dataset}_train.csv", index_col=0)
        test_data = pd.read_csv(path+f"/{dataset}_test.csv", index_col=0)

        print(f"{dataset}, {train_data.shape}")
        print(f"{dataset}, {test_data.shape}")
        
        test_label = test_data.iloc[:, -1]
        test_data = test_data.iloc[:, :-1]
        test_data.columns = [''] * len(test_data.columns)

    elif dataset=='JumpStarter':
        train_data = pd.read_csv(path+f"/service_train.csv", index_col=0)
        test_data = pd.read_csv(path+f"/service_test.csv", index_col=0)

        print(f"{dataset}, {train_data.shape}")
        print(f"{dataset}, {test_data.shape}")
        
        test_label = test_data.iloc[:, -1]
        test_data = test_data.iloc[:, :-1]
        test_data.columns = [''] * len(test_data.columns)

    if max_train_size is None:
        train_end = None
    else:
        train_end = train_start + max_train_size
    if max_test_size is None:
        test_end = None
    else:
        test_end = test_start + max_test_size
    print("load data of:", dataset)

    if normalize:
        train_data, scaler = normalize_data(train_data, scaler=None)
        test_data, _ = normalize_data(test_data, scaler=scaler)

    return (train_data, None), (test_data, test_label), root_cause_labels

class SlidingWindowDataset(Dataset):
    def __init__(self, data, window, target_dim=None, horizon=1):
        self.data = data
        self.window = window
        self.target_dim = target_dim
        self.horizon = horizon

    def __getitem__(self, index):
        x = self.data[index : index + self.window]
        y = self.data[index + self.window : index + self.window + self.horizon]
        return x, y

    def __len__(self):
        return len(self.data) - self.window


def create_data_loaders(train_dataset, batch_size, val_split=0.1, shuffle=True, test_dataset=None):
    train_loader, val_loader, test_loader = None, None, None
    if val_split == 0.0:
        print(f"train_size: {len(train_dataset)}")
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

    else:
        dataset_size = len(train_dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(val_split * dataset_size))
        if shuffle:
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
        val_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, sampler=valid_sampler)

        print(f"train_size: {len(train_indices)}")
        print(f"validation_size: {len(val_indices)}")

    if test_dataset is not None:
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        print(f"test_size: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def plot_losses(losses, save_path="", plot=True):
    """
    :param losses: dict with losses
    :param save_path: path where plots get saved
    """

    plt.plot(losses["train_forecast"], label="Forecast loss")
    plt.plot(losses["train_recon"], label="Recon loss")
    plt.plot(losses["train_total"], label="Total loss")
    plt.title("Training losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/train_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()

    plt.plot(losses["val_forecast"], label="Forecast loss")
    plt.plot(losses["val_recon"], label="Recon loss")
    plt.plot(losses["val_total"], label="Total loss")
    plt.title("Validation losses during training")
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.legend()
    plt.savefig(f"{save_path}/validation_losses.png", bbox_inches="tight")
    if plot:
        plt.show()
    plt.close()


def load(model, PATH, device="cpu"):
    """
    Loads the model's parameters from the path mentioned
    :param PATH: Should contain pickle file
    """
    model.load_state_dict(torch.load(PATH, map_location=device))


def get_series_color(y):
    if np.average(y) >= 0.95:
        return "black"
    elif np.average(y) == 0.0:
        return "black"
    else:
        return "black"


def get_y_height(y):
    if np.average(y) >= 0.95:
        return 1.5
    elif np.average(y) == 0.0:
        return 0.1
    else:
        return max(y) + 0.1


def adjust_anomaly_scores(scores, dataset, is_train, lookback):
    """
    Method for MSL and SMAP where channels have been concatenated as part of the preprocessing
    :param scores: anomaly_scores
    :param dataset: name of dataset
    :param is_train: if scores is from train set
    :param lookback: lookback (window size) used in model
    """

    if dataset.upper() not in ['SMAP', 'MSL']:
        return scores

    adjusted_scores = scores.copy()
    if is_train:
        md = pd.read_csv(f'/datasets/{dataset}/{dataset.lower()}_train_md.csv')
    else:
        md = pd.read_csv(f'/datasets/{dataset}/labeled_anomalies.csv')
        md = md[md['spacecraft'] == dataset.upper()]

    md = md[md['chan_id'] != 'P-2']

    md = md.sort_values(by=['chan_id'])

    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    sep_cuma = sep_cuma[:-1]
    buffer = np.arange(1, 20)
    i_remov = np.sort(np.concatenate((sep_cuma, np.array([i+buffer for i in sep_cuma]).flatten(),
                                      np.array([i-buffer for i in sep_cuma]).flatten())))
    i_remov = i_remov[(i_remov < len(adjusted_scores)) & (i_remov >= 0)]
    i_remov = np.sort(np.unique(i_remov))
    if len(i_remov) != 0:
        adjusted_scores[i_remov] = 0

    sep_cuma = np.cumsum(md['num_values'].values) - lookback
    s = [0] + sep_cuma.tolist()
    for c_start, c_end in [(s[i], s[i+1]) for i in range(len(s)-1)]:
        e_s = adjusted_scores[c_start: c_end+1]

        e_s = (e_s - np.min(e_s))/(np.max(e_s) - np.min(e_s))
        adjusted_scores[c_start: c_end+1] = e_s

    return adjusted_scores
