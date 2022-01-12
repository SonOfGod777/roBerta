# encoding=utf-8
import torch
import sys
import numpy as np
import pandas as pd
from sklearn import metrics
from torch import nn
from torch.optim import Adam
from pre_data import BuildData, BatchData
from config import args
from bert_model.bert_model import BertModel
pd.set_option('display.max_columns', None)


class Train(object):
    def __init__(self):
        train_data, dev_data, test_data = BuildData().build_data()
        self.train_data = BatchData(train_data)
        self.dev_data = BatchData(dev_data)
        self.test_data = BatchData(test_data)
        self.BertModel = BertModel()
        # self.BertModel.load_pre_model(args.pre_path)
        self.optimizer = Adam(self.BertModel.parameters(), lr=args.learn_rate)
        self.loss = nn.CrossEntropyLoss().to(args.device)

    def train(self):
        total_batch = 0
        last_improve = 0  # 上次loss下降的batch数
        flag = False  # 如果loss很久没有下降，结束训练
        best_loss = float('inf')
        self.BertModel.train()
        for epoch in range(args.epochs):
            print('Epoch [{}/{}]'.format(epoch + 1, args.epochs))
            if flag:
                break
            for i, (trains, labels, positions) in enumerate(self.train_data):
                outputs, _ = self.BertModel(trains)
                batch_num = outputs.size()[0]
                outputs_ = outputs.view(args.sentence_len * batch_num, -1)
                loss = self.loss(outputs_, labels.view(-1))
                self.BertModel.zero_grad()
                loss.backward()
                self.optimizer.step()
                if total_batch % 2 == 0:
                    outputs = torch.nn.Softmax(dim=-1)(outputs[:, 1:])
                    predict_label = torch.max(outputs.data, -1)[1].cpu()   # 0是最大值，1是最大值索引
                    predict_labels, true_labels = self.gather_index(labels, predict_label, positions)
                    train_acc = metrics.accuracy_score(true_labels, predict_labels)
                    dev_acc, dev_loss = self.evaluate(self.BertModel, self.dev_data)
                    if dev_loss < best_loss:
                        best_loss = dev_loss
                        save_path = args.save_path + '/trans_point.ep{}'.format(total_batch)
                        torch.save(self.BertModel.state_dict(), save_path)
                        last_improve = total_batch
                    print('total_batch:{}, train_loss:{}, train_acc:{}, dev_loss:{}, dev_acc:{}, last_improve:{}'.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, total_batch))
                total_batch += 1
                if total_batch - last_improve > 1000:
                    flag = True
                    break

    def gather_index(self, labels, predict_label, positions):
        true_label = labels[:, 1:].data.cpu()
        true_labels, predict_labels = [], []
        for batch_id in range(predict_label.size()[0]):
            position = torch.tensor(positions[batch_id])
            labels = torch.gather(true_label[batch_id], -1, position).numpy()
            predicts = torch.gather(predict_label[batch_id], -1, position).numpy()
            assert len(labels) == len(predicts)
            true_labels.extend(labels)
            predict_labels.extend(predicts)
        return predict_labels, true_labels

    def test(self, model, data):
        model.load_state_dict(torch.load(args.save_path))
        model.eval()
        test_acc, test_loss = self.evaluate(model, self.test_data)

    def evaluate(self, model, data):
        model.eval()
        loss_total = 0
        with torch.no_grad():
            true_labels_lis, predict_labels_lis = [], []
            for i, (texts, labels, positions) in enumerate(data):
                output, _ = model(texts)
                batch_num = output.size()[0]
                output_ = output.view(args.sentence_len * batch_num, -1)
                loss = nn.CrossEntropyLoss()(output_, labels.view(-1)).to(args.device)
                loss_total += loss
                output = torch.nn.Softmax(dim=-1)(output[:, 1:])
                predict_label = torch.max(output.data, -1)[1].cpu()  # 0是最大值，1是最大值索引
                predict_labels, true_labels = self.gather_index(labels, predict_label, positions)
                true_labels_lis.extend(true_labels)
                predict_labels_lis.extend(predict_labels)
            train_acc = metrics.accuracy_score(true_labels, predict_labels)
        return train_acc, loss_total/i


if __name__ == '__main__':
    Train().train()



