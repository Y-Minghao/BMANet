import warnings
import torch_geometric.transforms as T
from utils import Complete
from dataprocess import TCMSP
from torch_geometric.utils import degree
from trainer_classification import Trainer_classification
from trainer_regression import Trainer_regression
from torch_geometric.data import Batch
import sys
import torch
import numpy as np
import random
from ENMPNN import Add_MPNN,ADDMPNN
import pandas as pd
from model import MPNN,GCN,ARMA, ENMPNN
from AttentiveFP import AttentiveFP
from DMPNN import DMPNN
from trimNet import TrimNet
from BGNN import BGNN
# from ablation import  ENMPNN_no_BMA,ENMPNN_no_scaling,ENMPNN_no_attention,ENMPNN,ENMPNN_no_bro,ENMPNN_no_BMA_SSA,ENMPNN_no_Bro_SSA,ENMPNN_no_Bma_Bro_SSA


SPLIT_TARGET = 20

warnings.filterwarnings("ignore")

train_batch = int(sys.argv[1])
train_epoch = int(sys.argv[2])
gpu = int(sys.argv[3])
option = {
    'train_epoch': train_epoch,
    'train_batch': train_batch,
    'lr': 5e-4,
    'lr_scheduler_patience': 10,
    'early_stop_patience': -1,
    'cuda_devices': [gpu],
}

class SpecifyTarget(object):
    def __call__(self, data):
        for i in range(len(data.y)):
            if SPLIT_TARGET>data.y[i]>=0:
                data.y[i] = 0
            elif 100>=data.y[i]>=SPLIT_TARGET:
                data.y[i] = 1
        return data

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(2021)
print('Load Dataset...')
transform = T.Compose([T.Distance(norm=True),Complete()])



path = r'D:\dataset\oral bioavaibility'
train_dataset = TCMSP(root=path,split='train').shuffle()
test_dataset = TCMSP(root = path,split='test')
print('Split the train val...')

print('Training init...')
# model = AttentiveFP(in_channels=46,hidden_channels = 64,out_channels=3,edge_dim=14,num_layers=3,num_timesteps=3)
all_auc = []
all_accuracy = []
all_recall = []
all_precision = []
all_F1 = []
model = ENMPNN()
trainer = Trainer_classification(option, model, \
                  train_dataset = train_dataset, test_dataset=test_dataset)

# trainer = Trainer(option, model, \
#                   train_dataset = train_dataset,test_dataset = test_dataset)

trainer.train()
print('Testing')
trainer.load_best_ckpt()#加载最优模型

loss,auc, accuracy,recall,precision,F1 = trainer.valid_iterations(mode='test',need_y_probs=True)#测试
print('loss:{:.5f}\nauc:{:.5f}\naccuracy:{:.5f}\nrecall:{:.5f}\nprecision:{:.5f}\nF1:{:.5f}'.format(loss, auc,
                                                                                                        accuracy,
                                                                                                        recall,
                                                                                                        precision, F1))
# for i in range(5):
#     model = MPNN()
#     trainer = Trainer_classification(option, model, \
#                       train_dataset = train_dataset, test_dataset=test_dataset)
#
#     # trainer = Trainer(option, model, \
#     #                   train_dataset = train_dataset,test_dataset = test_dataset)
#
#     trainer.train()
#     print('Testing')
#     trainer.load_best_ckpt()#加载最优模型
#
#     loss,auc, accuracy,recall,precision,F1 = trainer.valid_iterations(mode='test',need_y_probs=True)#测试
#     all_auc.append(auc)
#     all_accuracy.append(accuracy)
#     all_recall.append(recall)
#     all_precision.append(precision)
#     all_F1.append(F1)
#     print('loss:{:.5f}\nauc:{:.5f}\naccuracy:{:.5f}\nrecall:{:.5f}\nprecision:{:.5f}\nF1:{:.5f}'.format(loss, auc,
#                                                                                                         accuracy,
#                                                                                                         recall,
#                                                                                                         precision, F1))
# mean_auc = pd.Series(all_auc).mean()
# mean_accuracy = pd.Series(all_accuracy).mean()
# mean_recall = pd.Series(all_recall).mean()
# mean_precision = pd.Series(all_precision).mean()
# mean_F1 = pd.Series(all_F1).mean()
# print('mean_auc: {:.5f}\nmean_accuracy: {:.5f}\nmean_recall: {:.5f}\nmean_precision: {:.5f}\nmean_F1: {:.5f}'.format(mean_auc,mean_accuracy,mean_recall,mean_precision,mean_F1))