import os
from tqdm import tqdm
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import DataLoader
from utils import Evaluate,result_table_valid
import torch.nn.functional as F
import time

torch.backends.cudnn.enabled = True

def count_target_weight(dataset):
    target_pd = pd.Series(dataset.data.y.view(-1).numpy())
    return [len(target_pd) / len(target_pd[target_pd == 0]), len(target_pd) / len(target_pd[target_pd == 1])]
class Trainer_classification():
    def __init__(self, option, model, train_dataset, valid_dataset, test_dataset):
        self.option = option
        self.device = torch.device("cuda:{}".format(option['cuda_devices'][0]) if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        # Setting the train valid and test data loader
        self.train_dataloader = DataLoader(train_dataset, batch_size=self.option['train_batch'], follow_batch=['x','edge_attr'])

        self.valid_dataloader = DataLoader(valid_dataset, batch_size=self.option['train_batch'], follow_batch=['x','edge_attr'])
        if test_dataset: self.test_dataloader = DataLoader(test_dataset, batch_size=self.option['train_batch'], follow_batch=['x','edge_attr'])

        # Setting the Adam optimizer with hyper-param
        weights = count_target_weight(train_dataset)
        self.criterion  = torch.nn.CrossEntropyLoss(torch.Tensor(weights).to(self.device),reduction='mean')
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.option['lr'],weight_decay=5e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(#CosineAnnealingLR
            self.optimizer, mode='min', factor=0.7,
            patience=self.option['lr_scheduler_patience'], min_lr=0.0000001,verbose=False
        )

        self.start = time.time()
        self.file_name = model.__class__.__name__
        self.task = self.option['task']
        self.file_time = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
        self.abs_file_dir = os.path.dirname(os.path.abspath(__file__))
        self.ckpt_save_dir = os.path.join(self.abs_file_dir, 'ckpt',
                                          '{}_{}_{}'.format(self.task, self.file_name, self.file_time))
        self.log_save_path = os.path.join(self.abs_file_dir, 'log',
                                          '{}_{}_{}.txt'.format(self.task, self.file_name, self.file_time))
        self.y_probs_path = os.path.join(self.abs_file_dir, 'y_probs',
                                          '{}_{}_{}.csv'.format(self.task, self.file_name, self.file_time))
        self.record_save_path = os.path.join(self.abs_file_dir, 'record',
                                             '{}_{}_{}.csv'.format(self.task, self.file_name, self.file_time))

        os.makedirs('log',exist_ok=True)
        os.makedirs('record',exist_ok= True)
        os.makedirs('y_probs',exist_ok=True)
        os.makedirs('ROC_image',exist_ok=True)
        os.makedirs(self.ckpt_save_dir,exist_ok=True)

        self.records = {'epoch':[],'lr':[],
                        'train_auc':[],'val_auc':[],'test_auc':[],
                        'train_loss': [], 'val_loss':[],'test_loss':[],
                        'train_accuracy':[],'val_accuracy': [], 'test_accuracy':[],
                        'train_recall':[],'val_recall':[], 'test_recall':[],
                        'train_precision':[],'val_precision':[],'test_precision':[],
                        'train_F1':[],'val_F1':[],'test_F1':[],
                        'best_ckpt': None}
        self.log(msgs=['\t{}:{}\n'.format(k, v) for k, v in self.option.items()])
        self.log('time: {}'.format(self.file_time))
        self.log('Model:{}'.format(self.file_name))

        self.log('train set num:{} valid set num:{} test set num: {}'.format(
            len(train_dataset), len(valid_dataset), len(test_dataset)))
        self.log("Total Parameters:" + str(sum([p.nelement() for p in self.model.parameters()])))
        for i in self.model._modules.items():
            self.log('{}:{}'.format(i[0],i[1]))

    def train_iterations(self, epoch):
        self.model.train()
        losses = []
        outputs = []
        y_probs = []
        ys = []
        y_prediction = []
        self.model.train()
        for i, data in enumerate(self.train_dataloader):
            self.optimizer.zero_grad()
            data = data.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, data.y.view(-1))
            y_prob = F.softmax(output, dim=1)  # 预测概率
            _, y_predict = torch.max(output, 1)  # 预测标签
            y_predict = y_predict.to('cpu')
            y = data.y.to('cpu')
            y_prob = y_prob.to('cpu')
            y_probs.extend(y_prob.detach().numpy())
            y_prediction.extend(y_predict.detach().numpy())
            ys.extend(y.detach().numpy())
            outputs.extend(output)

            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
        trn_loss = np.array(losses).mean()
        ys = np.array(ys)
        y_probs = np.array(y_probs)[:, 1]
        y_prediction = np.array(y_prediction)
        result = Evaluate(y_predict=y_prediction, y_probs=y_probs, y_test=ys, name=self.file_name, time=self.file_time)
        auc, accuracy, recall, precision, F1 = result.evaluate()
        return trn_loss,auc, accuracy,recall,precision,F1

    def valid_iterations(self, mode='valid',need_y_probs=False):
        self.model.eval()
        if mode == 'test':
            dataloader = self.test_dataloader
        if mode == 'valid':
            dataloader = self.valid_dataloader
        outputs = []
        y_probs = []
        ys = []
        y_prediction = []
        with torch.no_grad():
            for data in dataloader:

                data = data.to(self.device)
                output = self.model(data)#模型输出
                y_prob = F.softmax(output,dim = 1)#预测概率
                _, y_predict = torch.max(output, 1)#预测标签
                y_predict = y_predict.to('cpu')
                y = data.y.to('cpu')
                y_prob = y_prob.to('cpu')
                y_probs.extend(y_prob.numpy())
                y_prediction.extend(y_predict)
                ys.extend(y.cpu())
                outputs.extend(output.to('cpu').numpy())
        outputs = np.array(outputs)
        loss = self.criterion(torch.from_numpy(outputs).to('cuda'), torch.Tensor(ys).to('cuda').long()).item()
        ys = np.array([data.numpy() for data in ys])
        y_probs = np.array(y_probs)[:,1]
        y_prediction = np.array([data.numpy() for data in y_prediction])
        result = Evaluate(y_predict = y_prediction,y_probs = y_probs,y_test = ys,name = self.file_name,time = self.file_time)
        auc,accuracy,recall,precision,F1 = result.evaluate()
        if mode == 'test' and need_y_probs == True:

            pd.DataFrame.from_dict(
                {
                    'y_probs':y_probs,
                    'y_predict':y_prediction.reshape(-1),
                    'y_test':ys.reshape(-1)
                }
            ).to_csv(self.y_probs_path)
        return loss,auc, accuracy,recall,precision,F1

    def train(self):
        self.log('Training start...')
        early_stop_cnt = 0
        for epoch in tqdm(range(self.option['train_epoch'])):
            train_loss,train_auc,train_accuracy,train_recall,train_precision,train_F1 = self.train_iterations(epoch)
            val_loss,val_auc,val_accuracy,val_recall,val_precision,val_F1 = self.valid_iterations(mode = 'valid')
            test_loss,test_auc, test_accuracy,test_recall,test_precision,test_F1 = self.valid_iterations(mode = 'test')

            self.scheduler.step(val_loss)
            lr_cur = self.scheduler.optimizer.param_groups[0]['lr']

            self.log('============================================================\n'
                     'Epoch:{} lr_cur:{:.7f}\n'
                     'train_loss:{:.5f} val_loss:{:.5f} tes_loss:{:.5f} \n'
                     'train_auc:{:.5f} val_auc:{:.5f} test_auc:{:.5f}\n'
                     'val_accuracy:{:.5f} tes_accuracy:{:.5f} \n'
                     'val_recall:{:.5f} tes_recall:{:.5f} \n'
                     'val_precision:{:.5f} tes_precision:{:.5f}\n'
                     'val_F1:{:.5f} tes_F1:{:.5f}'.format(
                epoch,lr_cur,
                train_loss, val_loss, test_loss,
                train_auc, val_auc, test_auc,
                train_accuracy, val_accuracy,test_accuracy,
                train_recall, val_recall,test_recall,
                train_precision, val_precision,test_precision,
                train_F1, val_F1,test_F1),
                     with_time=True,print_msg=False)
            result_table_valid(epoch = epoch,lr = lr_cur,train_loss=train_loss,val_loss = val_loss,test_loss = test_loss,
                               train_auc = train_auc, val_auc = val_auc,test_auc = test_auc,
                               train_accuracy = train_accuracy, val_accuracy = val_accuracy,test_accuracy = test_accuracy,
                               train_recall = train_recall, val_recall=val_recall,test_recall = test_recall,
                               train_precision = train_precision, val_precision = val_precision,test_precision =  test_precision,
                               train_F1 = train_F1, val_F1 = val_F1,test_F1 =test_F1)
            self.records['epoch'].append(epoch+1)
            self.records['lr'].append(lr_cur)
            self.records['train_auc'].append(train_auc)
            self.records['val_auc'].append(val_auc)
            self.records['test_auc'].append(test_auc)
            self.records['train_loss'].append(train_loss)
            self.records['val_loss'].append(val_loss)
            self.records['test_loss'].append(test_loss)
            self.records['train_accuracy'].append(train_accuracy)
            self.records['val_accuracy'].append(val_accuracy)
            self.records['test_accuracy'].append(test_accuracy)
            self.records['train_recall'].append(train_recall)
            self.records['val_recall'].append(val_recall)
            self.records['test_recall'].append(test_recall)
            self.records['train_precision'].append(train_precision)
            self.records['val_precision'].append(val_precision)
            self.records['test_precision'].append(test_precision)
            self.records['train_F1'].append(train_F1)
            self.records['val_F1'].append(val_F1)
            self.records['test_F1'].append(test_F1)
            if  val_auc == np.array(self.records['val_auc']).max():
                self.save_model_and_records(epoch = epoch,val_auc = val_auc,test_auc = test_auc)
                early_stop_cnt = 0
            else:
                early_stop_cnt += 1
            if self.option['early_stop_patience'] > 0 and early_stop_cnt > self.option['early_stop_patience']:
                self.log('Early stop hitted!')
                break
        self.save_loss_records()

    def save_model_and_records(self, epoch, val_auc = None,test_auc = None,final_save=False):
        if final_save:
            self.save_loss_records()
            file_name = 'Final_save_Epoch{}_val_auc{:.5f}_tes_auc{:.5f}'.format(epoch, val_auc, test_auc)
        else:
            file_name = 'Epoch{}_val_auc{:.5f}_tes_auc{:.5f}'.format(epoch, val_auc, test_auc)
            self.records['best_ckpt'] = file_name

        with open(os.path.join(self.ckpt_save_dir, file_name), 'wb') as f:
            torch.save({
                'option': self.option,
                'records': self.records,
                'model_state_dict': self.model.state_dict(),
            }, f)
        self.log('Model saved at epoch {}'.format(epoch))

    def save_loss_records(self):
        ret = pd.DataFrame.from_dict(self.records)
        ret.to_csv(self.record_save_path)
        return ret

    def load_best_ckpt(self):
        ckpt_path = self.ckpt_save_dir + '/' + self.records['best_ckpt']
        self.log('The best ckpt is {}'.format(ckpt_path))
        self.load_ckpt(ckpt_path)

    def load_ckpt(self, ckpt_path):
        self.log('Ckpt loading: {}'.format(ckpt_path))
        ckpt = torch.load(ckpt_path)
        self.option = ckpt['option']
        self.records = ckpt['records']
        self.model.load_state_dict(ckpt['model_state_dict'])

    def log(self, msg=None, msgs=None, with_time=False,print_msg = True):
        if with_time: msg = msg + ' Time elapsed {:.2f} hrs ({:.1f} mins)'.format(
            (time.time() - self.start) / 3600.,
            (time.time() - self.start) / 60.
        )
        with open(self.log_save_path, 'a+') as f:
            if msgs:
                f.writelines(msgs);
            if msg:  f.write(msg + '\n');
        if print_msg and msg:
            print(msg)
        if print_msg and msgs:
            for msg in msgs:
                print(msg,end = ' ')
            print('\n')


if __name__ == '__main__':
    pass
