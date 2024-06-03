import numpy as np
import torch
import torch.nn as nn
import utils
from model import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

import logging


class Worker(object):
    def __init__(self, actions_p, actions_log_p, actions_index, args, device):
        self.actions_p = actions_p
        self.actions_log_p = actions_log_p
        self.actions_index = actions_index
        self.genotype = utils.parse_actions_index(actions_index,steps=args.model_steps)#根据action_index生成cell的结构样式

        self.args = args
        self.device = device

        self.params_size = None
        self.acc = None

def get_acc(worker):
    torch.manual_seed(worker.args.seed)
    np.random.seed(worker.args.seed)
    if torch.cuda.is_available():
        device = torch.device(worker.device)
        cudnn.benchmark = True
        cudnn.enable = True
        torch.cuda.manual_seed(worker.args.seed)
    else:
        device = torch.device('cpu')

    # train_transform, valid_transform = utils._data_transforms_cifar10(worker.args)
    # train_data = torchvision.datasets.MNIST(root=worker.args.data, train=True,
    #                                         transform=train_transform,
    #                                         download=True)
    # num_train = len(train_data)
    # indices = list(range(num_train))
    # split = int(np.floor(worker.args.train_portion * num_train))
    # train_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=worker.args.batch_size,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:split]),
    #     pin_memory=False, num_workers=2)
    #
    # valid_queue = torch.utils.data.DataLoader(
    #     train_data, batch_size=worker.args.batch_size,
    #     sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
    #     pin_memory=False, num_workers=2)

    #改变数据集
    from Prodataset.MappPreProcess_dataset import get_Trainqueue_Validqueue,new_get_Trainqueue_Validqueue

    train_queue, valid_queue=new_get_Trainqueue_Validqueue(hdf5_filepath=worker.args.Mapp_hdf5_filepath,
                                                                 train_portion=worker.args.train_portion,
                                                                 batch_size=worker.args.batch_size,
                                                                 PacketOrByte=worker.args.packet_lengthORBytes)


    # criterion = nn.CrossEntropyLoss()
    criterion=nn.CrossEntropyLoss()
    model = Network(worker.genotype,num_classes=worker.args.num_classes,layers=worker.args.model_layers,steps=worker.args.model_steps,num_embeddings=worker.args.num_embeddings,embedding_dim=worker.args.embedding_dim).to(device)#根据结构样式生成网络模型

    worker.params_size = utils.count_params(model)

    # optimizer = torch.optim.SGD(model.parameters(),
    #                             worker.args.model_lr,
    #                             momentum=worker.args.model_momentum,
    #                             weight_decay=worker.args.model_weight_decay)
    optimizer=torch.optim.Adam(model.parameters(), lr=worker.args.model_lr)

    for model_epoch in range(worker.args.model_epochs):
        logging.info('model_epoch: {}'.format(model_epoch))
        print('model_epoch: {}'.format(model_epoch))
        train_loss, train_acc = train(model, train_queue, criterion, optimizer, device)
        #print('train loss {:.4f} acc {:.4f}'.format(train_loss, train_acc))

    valid_loss, valid_acc, macrof1score, precision_score,recall_score = infer(model, valid_queue, criterion, device)#推测小cell在验证集上的精度
    logging.info('valid loss {:.4f} acc {:.4f} recall {:.4f} precision {:.4f} f1 {:.4f}'.format(valid_loss, valid_acc,recall_score,precision_score,macrof1score))
    print('valid loss {:.4f} acc {:.4f} recall {:.4f} precision {:.4f} f1 {:.4f}'.format(valid_loss, valid_acc,recall_score,precision_score,macrof1score))

    from utils import save_model,update_score_to_file
    model_path = worker.args.model_save_path
    # F1 Acc Recall Precision
    if worker.args.valid_method=='F1':
        worker.acc = macrof1score#根据validate score标准
        utils.save_model(worker.genotype,worker.args.num_classes,worker.args.model_layers,worker.args.model_steps,model,
                   f'{model_path}/{worker.args.packet_lengthORBytes}/{worker.args.valid_method}_Layer{worker.args.model_layers}_step{worker.args.model_steps}/{worker.acc}/modelparams'
                   )
        update_score_to_file(worker.acc,f'{model_path}/{worker.args.packet_lengthORBytes}/{worker.args.valid_method}_Layer{worker.args.model_layers}_step{worker.args.model_steps}/maxscore.txt')
    elif worker.args.valid_method=='Recall':
        worker.acc = recall_score#根据validate score标准
        save_model(worker.genotype, worker.args.num_classes, worker.args.model_layers, worker.args.model_steps, model,
                   f'{model_path}/{worker.args.packet_lengthORBytes}/{worker.args.valid_method}_Layer{worker.args.model_layers}_step{worker.args.model_steps}/{worker.acc}/modelparams'
                   )
        update_score_to_file(worker.acc,
                             f'{model_path}/{worker.args.packet_lengthORBytes}/{worker.args.valid_method}_Layer{worker.args.model_layers}_step{worker.args.model_steps}/maxscore.txt')
    elif worker.args.valid_method=='Precision':
        worker.acc = precision_score#根据validate score标准
        save_model(worker.genotype, worker.args.num_classes, worker.args.model_layers, worker.args.model_steps, model,
                   f'{model_path}/{worker.args.packet_lengthORBytes}/{worker.args.valid_method}_Layer{worker.args.model_layers}_step{worker.args.model_steps}/{worker.acc}/modelparams'
                   )
        update_score_to_file(worker.acc,
                             f'{model_path}/{worker.args.packet_lengthORBytes}/{worker.args.valid_method}_Layer{worker.args.model_layers}_step{worker.args.model_steps}/maxscore.txt')
    elif worker.args.valid_method=='Acc':
        worker.acc = valid_acc#根据validate score标准
        save_model(worker.genotype, worker.args.num_classes, worker.args.model_layers, worker.args.model_steps, model,
                    f'{model_path}/{worker.args.packet_lengthORBytes}/{worker.args.valid_method}_Layer{worker.args.model_layers}_step{worker.args.model_steps}/{worker.acc}/modelparams'
                   )
        update_score_to_file(worker.acc,
                             f'{model_path}/{worker.args.packet_lengthORBytes}/{worker.args.valid_method}_Layer{worker.args.model_layers}_step{worker.args.model_steps}/maxscore.txt')
    else:
        worker.acc = macrof1score  # 根据validate score标准
        save_model(worker.genotype, worker.args.num_classes, worker.args.model_layers, worker.args.model_steps, model,
                   f'{model_path}/{worker.args.packet_lengthORBytes}/{worker.args.valid_method}_Layer{worker.args.model_layers}_step{worker.args.model_steps}/{worker.acc}/modelparams'
                   )
        update_score_to_file(worker.acc,
                             f'{model_path}/{worker.args.packet_lengthORBytes}/{worker.args.valid_method}_Layer{worker.args.model_layers}_step{worker.args.model_steps}/maxscore.txt')
    # logging.info(valid_acc)
    # if valid_acc > 0.94:
    #     torch.save(model, 'acc{:.4f}.pt'.format(valid_acc))
    #     print('acc{:.4f}  saved'.format(valid_acc))


def train(model, train_queue, criterion, optimizer, device):
    avg_loss = 0
    avg_acc = 0
    batch_num = len(train_queue)

    model.train()
    # model=model.type(torch.FloatTensor).to(device)
    for batch, (input, target) in enumerate(train_queue):
        # input=input.type(torch.float32)
        #增加了view
        # input=input.view(256,64,8,8)
        # input = input.view(256, 64, 64, 1)
        input = Variable(input, requires_grad=False).to(device)
        target = Variable(target, requires_grad=False).to(device)
        # print('input.dtype:{} '.format(input.dtype))
        # print(target)

        optimizer.zero_grad()
        logits ,_= model(input)
        # logits=torch.log(logits)
        loss = criterion(logits, target)
        # print("logits:{}  target:{}".format(logits.shape,target.shape))   #logits:torch.Size([256, 25])  target:torch.Size([256])
        loss.backward()
        optimizer.step()

        acc = utils.accuracy(logits.data, target.data)[0]
        avg_loss += float(loss)
        avg_acc += float(acc)

        del input,target,loss,logits

    return avg_loss / batch_num, avg_acc / batch_num


def savemdoelandDict(model,modelame):
    torch.save(model,modelame)
    torch.save(model.state_dict(),modelame+".dict")

def infer1(model, valid_queue, criterion, device):#返回全局准确度
    avg_loss = 0
    avg_acc = 0
    batch_num = len(valid_queue)

    model.eval()
    for batch, (input, target) in enumerate(valid_queue):
        with torch.no_grad():

            # input = input.type(torch.float32)
            # input = input.view(256, 64, 8, 8)
            # input = input.view(256, 64, 64, 1)

            input = Variable(input).to(device)
            target = Variable(target).to(device)

            logits = model(input)
            loss = criterion(logits, target)
        acc = utils.accuracy(logits.data, target.data)[0]
        avg_loss += float(loss)
        avg_acc += float(acc)

    valid_acc=avg_acc / batch_num
    if valid_acc > 0.88:
        savemdoelandDict(model,'0.75dictnew20{:.4f}.pt'.format(valid_acc))
        # torch.save(model, 'new20maskipacc{:.4f}.pt'.format(valid_acc))
        logging.info('0.75dictnew20{:.4f}  saved'.format(valid_acc))
        print('0.75dictnew20{:.4f}  saved'.format(valid_acc))

    return avg_loss / batch_num, avg_acc / batch_num

def infer(model, valid_queue, criterion, device):#返回平均准确度
    avg_loss = 0
    batch_num = len(valid_queue)
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)

    model.eval()
    for batch, (input, target) in enumerate(valid_queue):
        with torch.no_grad():

            # input = input.type(torch.float32)
            # input = input.view(256, 64, 8, 8)
            # input = input.view(256, 64, 64, 1)

            input = Variable(input).to(device)
            target = Variable(target).to(device)

            logits = model(input)
            loss = criterion(logits, target)

            y_predict = torch.cat([y_predict, torch.max(logits, 1)[1]], 0)
            y_true = torch.cat([y_true, target], 0)


        # acc = utils.accuracy(logits.data, target.data)[0]
        avg_loss += float(loss)
        # avg_acc += float(acc)

    y_true_list = y_true.cpu().numpy().tolist()
    y_predict_list = y_predict.cpu().numpy().tolist()
    y_true_trans = np.array(y_true_list)
    y_predict_trans = np.array(y_predict_list)
    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score

    avg_acc=valid_acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
    macrof1score = f1_score(y_true_trans, y_predict_trans, average="macro")
    precisionscore = precision_score(y_true_trans, y_predict_trans, average='macro')
    recallscore = recall_score(y_true_trans,y_predict_trans,average='macro')



    # if valid_acc > 0.82:
    #     savemdoelandDict(model,'8layer4steps128bytesRecall{:.4f}.pt'.format(valid_acc))
    #     # torch.save(model, 'new20maskipacc{:.4f}.pt'.format(valid_acc))
    #     logging.info('8layer4steps128bytesRecall{:.4f}  saved'.format(valid_acc))
    #     print('8layer4steps128bytesRecall{:.4f}  saved'.format(valid_acc))

    return avg_loss / batch_num, avg_acc, macrof1score, precisionscore,recallscore

def inferF1(model, valid_queue, criterion, device):#返回F1
    avg_loss = 0
    batch_num = len(valid_queue)
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)

    model.eval()
    for batch, (input, target) in enumerate(valid_queue):
        with torch.no_grad():

            input = input.type(torch.float32)
            # input = input.view(256, 64, 8, 8)
            # input = input.view(256, 64, 64, 1)

            input = Variable(input).to(device)
            target = Variable(target).to(device)

            logits = model(input)
            loss = criterion(logits, target)

            y_predict = torch.cat([y_predict, torch.max(logits, 1)[1]], 0)
            y_true = torch.cat([y_true, target], 0)


        # acc = utils.accuracy(logits.data, target.data)[0]
        avg_loss += float(loss)
        # avg_acc += float(acc)

    y_true_list = y_true.cpu().numpy().tolist()
    y_predict_list = y_predict.cpu().numpy().tolist()
    y_true_trans = np.array(y_true_list)
    y_predict_trans = np.array(y_predict_list)

    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
    avg_acc=valid_acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
    macrof1score = f1_score(y_true_trans, y_predict_trans, average="macro")


    if valid_acc > 0.82:
        savemdoelandDict(model,'CIC8layer4steps64bytesF1{:.4f}.pt'.format(valid_acc))
        # torch.save(model, 'new20maskipacc{:.4f}.pt'.format(valid_acc))
        logging.info('CIC8layer4steps64bytesF1{:.4f}  saved'.format(valid_acc))
        print('CIC8layer4steps64bytesF1{:.4f}  saved'.format(valid_acc))

    return avg_loss / batch_num, macrof1score



def inferPre(model, valid_queue, criterion, device):#返回F1
    avg_loss = 0
    batch_num = len(valid_queue)
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)

    model.eval()
    for batch, (input, target) in enumerate(valid_queue):
        with torch.no_grad():

            input = input.type(torch.float32)
            # input = input.view(256, 64, 8, 8)
            # input = input.view(256, 64, 64, 1)

            input = Variable(input).to(device)
            target = Variable(target).to(device)

            logits = model(input)
            loss = criterion(logits, target)

            y_predict = torch.cat([y_predict, torch.max(logits, 1)[1]], 0)
            y_true = torch.cat([y_true, target], 0)


        # acc = utils.accuracy(logits.data, target.data)[0]
        avg_loss += float(loss)
        # avg_acc += float(acc)

    y_true_list = y_true.cpu().numpy().tolist()
    y_predict_list = y_predict.cpu().numpy().tolist()
    y_true_trans = np.array(y_true_list)
    y_predict_trans = np.array(y_predict_list)

    from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
    avg_acc_score=valid_acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
    precision_score = precision_score(y_true_trans, y_predict_trans,average='macro')
    # f1_score=f1_score(y_true_trans, y_predict_trans,average='macro')


    if valid_acc > 0.82:
        savemdoelandDict(model,'CIC4layer4steps64bytesPre{:.4f}.pt'.format(valid_acc))
        # torch.save(model, 'new20maskipacc{:.4f}.pt'.format(valid_acc))
        logging.info('CIC4layer4steps64bytesPre{:.4f}  saved'.format(valid_acc))
        print('CIC4layer4steps64bytesPre{:.4f}  saved'.format(valid_acc))

    return avg_loss / batch_num, precision_score