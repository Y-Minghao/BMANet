import math
from texttable import Texttable
import pandas as pd
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score,precision_score,f1_score
from sklearn.metrics import roc_curve, auc
import os
import matplotlib.pyplot as plt

def result_table_valid(epoch,lr,train_loss,val_loss,test_loss,train_auc,val_auc,test_auc,train_accuracy,
                       val_accuracy,test_accuracy,train_recall, val_recall,test_recall,
                       train_precision, val_precision,test_precision,train_F1,val_F1,test_F1):
    table = Texttable()
    table.set_cols_align(['c','c','c','c','c'])
    table.set_cols_valign(['m', 'm','m','m','m'])
    train_report = 'train_loss:{:.5f}\nauc:{:.5f}\naccuracy:{:.5f}\nrecall:{:.5f}\nprecision:{:.5f}\nF1:{:.5f}'.format(train_loss,train_auc,train_accuracy,train_recall,train_precision,train_F1)
    val_report = 'val_loss:{:.5f}\nauc:{:.5f}\naccuracy:{:.5f}\nrecall:{:.5f}\nprecision:{:.5f}\nF1:{:.5f}'.format(val_loss,val_auc,val_accuracy,val_recall,val_precision,val_F1)
    test_report = 'test_loss:{:.5f}\nauc:{:.5f}\naccuracy:{:.5f}\nrecall:{:.5f}\nprecision:{:.5f}\nF1:{:.5f}'.format(test_loss,test_auc,test_accuracy,test_recall,test_precision,test_F1)
    table.add_rows([['epoch','%e'%lr,'train','val','test'],
                   [epoch,lr,train_report,val_report,test_report]])
    print(table.draw() + "\n")

def plot_one_roc(y_test,y_probs,title,time):
    plt.figure()
    lw = 2
    plt.rcParams.update({"font.size": 20})#设置调整字体大小
    false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_probs)
    roc_auc = auc(false_positive_rate, true_positive_rate)
    plt.figure(figsize=(10,10))
    plt.plot(false_positive_rate, true_positive_rate, color='darkorange',
             lw=lw, label='AUC {} = {:.4f}'.format(title,roc_auc)) ###假正率为横坐标，真正率为纵坐标做曲线
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=20)
    plt.ylabel('True Positive Rate',fontsize=20)
    plt.title(title,fontsize=20)
    plt.legend(loc="lower right",fontsize=20)
    plt.savefig(fname=os.path.join('ROC_image',title + '_' + time + '.png'))
    plt.show()

class Evaluate():
    def __init__(self,y_predict,y_probs,y_test,name,time):
        self.y_predict = y_predict
        self.y_probs = y_probs
        self.y_test = y_test
        self.name = name
        self.time = time
    def evaluate(self):
        self.auc = roc_auc_score(self.y_test, self.y_probs)
        self.accuracy = accuracy_score(self.y_test, self.y_predict)
        self.recall = recall_score(self.y_test, self.y_predict, average="macro")
        self.precision = precision_score(self.y_test, self.y_predict, average="macro")
        self.F1 = f1_score(self.y_test, self.y_predict, average="macro")
        return self.auc, self.accuracy, self.recall, self.precision, self.F1
    def save_evaluate_result(self):
        result_dic = {
            'AUC': [], 'Accuracy': [], 'recall': [], 'precision': [], 'F1': []
        }
        result_dic['AUC'].append(self.auc)
        result_dic['Accuracy'].append(self.accuracy)
        result_dic['recall'].append(self.recall)
        result_dic['precision'].append(self.precision)
        result_dic['F1'].append(self.F1)
        result_dic['state'] = self.name
        pd.DataFrame.from_dict(result_dic).to_csv(os.path.join('result', self.name +'_' + self.time +  '.csv'))
    def save_y_probs(self,mol_id):
        dic = {
            'MOL_ID': mol_id,
            'y_probs': self.y_probs,
            'y_predict':self.y_predict,
            'y_test':self.y_test
        }
        pd.DataFrame.from_dict(dic).to_csv(os.path.join('y_probs', self.name +'_' + self.time +  '.csv'))
    def plot_one_roc(self,save = False):
        plot_one_roc(y_test = self.y_test, y_probs = self.y_probs, title = self.name, time = self.time)

def zeros(tensor):
    if tensor is not None:
        tensor.data.fill_(0)


def uniform(size, tensor):
    bound = 1.0 / math.sqrt(size)
    if tensor is not None:
        tensor.data.uniform_(-bound, bound)


def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

