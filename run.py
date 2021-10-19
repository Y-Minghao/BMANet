import sys
import warnings
import torch
from BMANet import BMANet
import numpy as np
import random
from torch_geometric.datasets import MoleculeNet
from trainer_classification import Trainer_classification
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(123)
warnings.filterwarnings("ignore")
dataset_name = str(sys.argv[1])
gpu = int(sys.argv[2])
train_batch = int(sys.argv[3])
train_epoch = int(sys.argv[4])

option = {
    'train_epoch': train_epoch,
    'train_batch': train_batch,
    'task': dataset_name,  # [0~11]
    'lr': 5e-4,
    'lr_scheduler_patience': 20,

    'parallel': False,
    'cuda_devices': [gpu],  # works when parallel=True
    'early_stop_patience': -1,  # -1 for no early stop
}
print('Load Dataset...')
data = MoleculeNet(root = '.',name=dataset_name).shuffle()
data.data.x = data.data.x.float()
data.data.edge_attr = data.data.edge_attr.float()
data.data.y = data.data.y.long()
data_size = len(data)

print('Split the dataset...')
train_dataset = data[:int(data_size * 0.8)]
valid_dataset = data[int(data_size * 0.8):int(data_size * 0.9)]
test_dataset = data[int(data_size * 0.9):]

print('Training init...')
model = BMANet()
trainer = Trainer_classification(option, model, train_dataset, valid_dataset,test_dataset)
trainer.train()

print('Testing')
trainer.load_best_ckpt()
loss,auc, accuracy,recall,precision,F1 = trainer.valid_iterations(mode='test')
print('loss: {}\nauc: {}\naccuracy: {}\nrecall: {}\nprecision: {}\nF1: {}'.format(loss,auc,accuracy,recall,precision,F1))






