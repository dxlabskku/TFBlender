###############################################################################
# (0) Imports
###############################################################################
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

###############################################################################
# (A) Load .pt files and inspect feature dimensions
###############################################################################
x_condition_tensor = torch.load('static_data_128.pt',  weights_only=True)    # (N, C_cond)
binary_labels      = torch.load('binary_labels_128.pt', weights_only=True)   # (N,)
all_pyramid_data   = torch.load('times_series_data_128.pt', weights_only=True)  # list[dict]

first_item    = all_pyramid_data[0]
group_keys    = [k for k in first_item.keys() if k != 'price']
price_dim     = first_item['price'].shape[1]
group_dim_sum = sum(first_item[k].shape[1] for k in group_keys)
condition_dim = x_condition_tensor.shape[1]
time_seq_len  = 128  # fixed sequence length

###############################################################################
# (B) Helper: filter training indices by maturity
###############################################################################
def filter_train_set_by_maturity(train_idx, val_test_min_issue, x_cond):
    """Keep only samples whose maturity precedes the earliest val/test issue date."""
    return [
        idx for idx in train_idx
        if float(x_cond[idx, 0].item()) < val_test_min_issue
    ]

###############################################################################
# (1) Custom torch Dataset
###############################################################################
class FinancialDataset(Dataset):
    def __init__(self, x_cond, pyramid_list, labels):
        self.x_cond       = x_cond
        self.pyramid_list = pyramid_list
        self.labels       = labels
        self.group_keys   = sorted([k for k in pyramid_list[0].keys() if k != 'price'])

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        price  = self.pyramid_list[idx]['price']                       # (T, price_dim)
        groups = [self.pyramid_list[idx][k] for k in self.group_keys]  # list[(T, d_i)]
        group  = torch.cat(groups, dim=1) if groups else None          # (T, group_dim_sum)
        cond   = self.x_cond[idx]                                      # (C_cond,)
        label  = self.labels[idx]                                      # scalar
        return price, group, cond, label

###############################################################################
# (2) Split indices into train / val / test based on issue date
###############################################################################
dataset     = FinancialDataset(x_condition_tensor, all_pyramid_data, binary_labels)
issue_dates = x_condition_tensor[:, 1].numpy()   # column 1 assumed to store issue date (epoch)

train_years, val_years, test_years = 9, 1, 3
sec_per_year = 365 * 24 * 3600

train_start = issue_dates[0]
train_end   = train_start + train_years * sec_per_year
val_start   = train_end
val_end     = val_start + val_years * sec_per_year
test_start  = val_end
test_end    = test_start + test_years * sec_per_year

train_idx = [i for i, d in enumerate(issue_dates) if train_start <= d < train_end]
val_idx   = [i for i, d in enumerate(issue_dates) if val_start   <= d < val_end]
test_idx  = [i for i, d in enumerate(issue_dates) if test_start  <= d < test_end]

# Apply maturity filter: exclude samples maturing after validation/test range
val_test_min_issue = min([x_condition_tensor[i, 0].item() for i in (val_idx + test_idx)]) \
                     if (val_idx + test_idx) else float('inf')
train_idx = filter_train_set_by_maturity(train_idx, val_test_min_issue, x_condition_tensor)

###############################################################################
# (3) Build Subsets
###############################################################################
train_set = torch.utils.data.Subset(dataset, train_idx)
val_set   = torch.utils.data.Subset(dataset, val_idx)
test_set  = torch.utils.data.Subset(dataset, test_idx)

###############################################################################
# (4) Class-imbalance check (positive vs. negative counts)
###############################################################################
pos_cnt = sum(binary_labels[i].item() == 1 for i in train_idx)
neg_cnt = len(train_idx) - pos_cnt
pos_weight_value = (neg_cnt / pos_cnt) if pos_cnt > 0 else 1.0  # use when defining the loss

###############################################################################
# (5) DataLoaders
###############################################################################
batch_size = 128
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  pin_memory=True)
val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, pin_memory=True)
test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, pin_memory=True)

###############################################################################
# Pre-processing finished — train_loader / val_loader / test_loader and
# pos_weight_value are ready for the model pipeline
###############################################################################
