import os

import numpy as np
import pickle
import subprocess
import torch
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.nn import functional as F

from utils import load_model, read_score_from_file

# dataset_path= '../Data_Processsed'
# model_path='genmodel/'
# def load_file(path):
#     with open(path+'/'+'label_dic.txt', 'rb') as f:
#         global label_dic
#         label_dic=pickle.load(f)
#
#     with open(path+'/'+'flow_list.txt', 'rb') as f:
#         global flow_list
#         flow_list=pickle.load(f)
#
#     with open(path+'/'+'label_list.txt', 'rb') as f:
#         global label_list
#         label_list=pickle.load(f)

class Network(nn.Module):
    def __init__(self, model1,model2,num_classes):
        super(Network, self).__init__()
        self.model1 = model1
        self.model2 = model2
        # 假设两个模型的输出维度与num_classes相同
        # 输出被concatenate后，维度将是num_classes的两倍
        self.fusion = nn.Linear(num_classes * 2, num_classes)  # 加入可训练的融合层

    def forward(self, input1, input2):
        logits1 = self.model1(input1)
        logits2 = self.model2(input2)

        # 使用concatenate合并两个模型的输出
        combined_logits = torch.cat((logits1, logits2), dim=1)

        # 通过融合层
        combined_logits = self.fusion(combined_logits)

        # 应用Softmax函数得到最终的输出
        final_output = F.softmax(combined_logits, dim=-1)
        return final_output

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value

def test_model(model, device, test_loader,result_save_path):
    # model_dir = "./model/"+ "best_model_few.pth"
    # model.load_state_dict(torch.load(model_dir))
    model.eval()
    y_true = torch.LongTensor(0).to(device)
    y_predict = torch.LongTensor(0).to(device)
    total_loss = 0.
    total=0.
    with torch.no_grad():
        for idx, (packet,byte, target) in enumerate(test_loader):
            packet, byte,target = packet.to(device),byte.to(device), target.to(device)
            # data = data.view(256, 64, 8, 8)
            pre = model(packet,byte)
            pred = pre.argmax(dim=1)
            y_predict = torch.cat([y_predict, torch.max(pre, 1)[1]], 0)
            y_true = torch.cat([y_true, target], 0)
            total += pred.shape[0]

        avg_loss = total_loss / total
        y_true_list = y_true.cpu().numpy().tolist()
        y_predict_list = y_predict.cpu().numpy().tolist()

        y_true_trans = np.array(y_true_list)
        y_predict_trans = np.array(y_predict_list)

        acc = balanced_accuracy_score(y_true_trans, y_predict_trans)
        microf1score = f1_score(y_true_trans, y_predict_trans, average="micro")
        macrof1score = f1_score(y_true_trans, y_predict_trans, average="macro")
        weightedf1score = f1_score(y_true_trans, y_predict_trans, average="weighted")
        matrix = confusion_matrix(y_true_trans, y_predict_trans)
        test_acc = 100. * acc



        recall = []
        precision = []
        F1 = []
        for i in range(matrix.shape[0]):
            recalli = matrix[i, i] / (matrix[i, :].sum())
            if (matrix[:, i].sum())==0:
                precisioni=0
            else:
                precisioni = matrix[i, i] / (matrix[:, i].sum())
            if (recalli + precisioni) == 0:
                F1i = 0
            else:
                F1i = (2 * recalli * precisioni) / (recalli + precisioni)
            recall.append(recalli)
            precision.append(precisioni)
            F1.append(F1i)
        accuracy = np.diag(matrix).sum() / matrix.sum()

        np.set_printoptions(threshold=np.inf)
        np.set_printoptions(linewidth=np.inf)

        print("microF1:{}  macroF1:{}  weightedF1:{}".format(microf1score, macrof1score, weightedf1score))
        print("---------------------------------------------------")
        print("测试集 Loss:{:.4f} 测试集 acc:{:.4f}".format(avg_loss, test_acc))
        print(matrix)

        print(recall)
        print(precision)
        print(F1)
        print(accuracy)
        print(np.array(recall).mean())
        print(np.array(precision).mean())
        print(np.array(F1).mean())

        with open(result_save_path,'a') as save_file:
            print("microF1:{}  macroF1:{}  weightedF1:{}".format(microf1score, macrof1score, weightedf1score),file=save_file)
            print("---------------------------------------------------",file=save_file)
            print("测试集 Loss:{:.4f} 测试集 acc:{:.4f}".format(avg_loss, test_acc),file=save_file)
            print(matrix,file=save_file)
            print(recall,file=save_file)
            print(precision,file=save_file)
            print(F1,file=save_file)
            print(accuracy, file=save_file)
            print(np.array(recall).mean(),file=save_file)
            print(np.array(precision).mean(),file=save_file)
            print(np.array(F1).mean(),file=save_file)

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    # valid_method_list = ['F1', 'Recall', 'Acc', 'Precision']
    valid_method_list = [ 'Recall']

    # valid_method='F1'
    labelcount = 28  # 这个数值是具体的分类数，大家可以自行修改
    fusion_model_epoch=20 #20

    byte_gpu='0'
    length_gpu = '0'
    test_gpu='0'
    device = torch.device('cuda:{}'.format(str(test_gpu)))

    # Mapp_dataset_path='/home/yintianan/python_workspace/new_autoMTC_MApp_embeded/MAppdata/preprocessed_data_Packet32_Byte64_label.hdf5'
    # Mapp_dataset_path='/home/yintianan/python_workspace/new_autoMTC_MApp_embeded/MAppdata/randomMIN500/preprocessed_data500_Packet_Byte64_label.hdf5'
    Mapp_dataset_path='/home/yintianan/python_workspace/new_autoMTC_MApp_embeded/MAppdata/data_Packet32_Byte64_label19'
    train_dataset_path = Mapp_dataset_path + '/Mapp_train_data1024_Packet_Byte64_label.hdf5'
    test_dataset_path = Mapp_dataset_path + '/Mapp_test_data200_Packet_Byte64_label.hdf5'
    model_path = '/home/yintianan/pythonwork/new_autoMTC_MApp_embeded/nasnet/genmodel_label19_randomMin2000_1024train_200test'


    #[layer,step]
    model_params=[
        [8, 4],
        # [8, 2],
                # [6,4],
                #   [6,2],
                #   [4,4],
                # [4, 2],
                #   [2,4],[2,2]
          ]
    # model_params = [
    #     [12, 12],
    #     ]

    labels = [x for x in range(labelcount)]

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    for valid_method in valid_method_list:
        for list in model_params:
            layer=list[0]
            step=list[1]

            #trainsearch first model
            print(f'____________________Train Bytes model,valid_method:{valid_method} ,layer={layer}, step={step}____________________')

            command = [
                '/home/yintianan/anaconda3/envs/torch-2.0/bin/python', 'nasnet/train_search.py',
                '--valid_method', valid_method,
                '--packet_lengthORBytes', 'Bytes',
                '--num_classes', str(labelcount),
                '--model_steps',str(step),
                '--model_layers',str(layer),
                '--model_save_path', model_path,
                '--Mapp_hdf5_filepath',train_dataset_path,
                '--model_epochs',str(0),#20
                '--arch_epochs',str(0),#15
                '--batch_size', str(256),
                '--num_embeddings',str(257),
                '--embedding_dim',str(257),
                '--gpu',byte_gpu

            ]

            import subprocess

            # 使用subprocess.Popen开启一个进程
            with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
                for line in p.stdout:
                    print(line, end='')  # 实时打印输出

            # 等待进程结束
            p.wait()



            # trainsearch second model
            print(f'____________________Train Length model,valid_method:{valid_method} , layer={layer}, step={step}____________________')

            command = [
                '/home/yintianan/anaconda3/envs/torch-2.0/bin/python', 'nasnet/train_search.py',
                '--valid_method', valid_method,
                '--packet_lengthORBytes', 'Length',
                '--num_classes', str(labelcount),
                '--model_steps', str(step),
                '--model_layers', str(layer),
                '--model_save_path',model_path,
                '--Mapp_hdf5_filepath',train_dataset_path,
                '--model_epochs', str(0),#15
                '--arch_epochs', str(0),#15
                '--batch_size',str(256),
                '--num_embeddings', str(3002),
                '--embedding_dim', str(257),
                '--gpu', length_gpu
            ]
            # 使用subprocess.Popen开启一个进程
            with subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=1, universal_newlines=True) as p:
                for line in p.stdout:
                    print(line, end='')  # 实时打印输出

            # 等待进程结束
            p.wait()

            print(f'____________________Train fusion model,valid_method:{valid_method} , layer={layer}, step={step}____________________')

            first_model_path = model_path + '/Bytes/'+f'{valid_method}'+f'_Layer{layer}_step{step}'
            second_model_path = model_path + '/Length/'+ f'{valid_method}' + f'_Layer{layer}_step{step}'
            first_model_score=read_score_from_file(first_model_path+'/maxscore.txt')
            second_model_score = read_score_from_file(second_model_path + '/maxscore.txt')
            first_model_path+=f'/{str(first_model_score)}/modelparams'
            second_model_path += f'/{str(second_model_score)}/modelparams'
            Byte_model=load_model(first_model_path,257,257)
            Packet_model = load_model(second_model_path,3002,257)

            fusion_model=Network( Packet_model,Byte_model, num_classes=labelcount)

            # 训练线性部分
            # set_requires_grad(fusion_model.model1,False)
            # set_requires_grad(fusion_model.model2,False)
            fusion_model.to(device)

            from Prodataset.MappPreProcess_dataset import new_get_Trainqueue_Validqueue,HDF5_Mapp_Dataset
            #
            # train_queue, valid_queue, test_queue = get_Trainqueue_Validqueue(hdf5_filepath=Mapp_dataset_path,
            #                                                                             train_portion=0.75,
            #                                                                             batch_size=512,
            #                                                                             PacketOrByte='Both')#还有包的数据没写

            train_queue, valid_queue=new_get_Trainqueue_Validqueue(hdf5_filepath=train_dataset_path, train_portion=0.75,
                                                                                        batch_size=256,
                                                                                        PacketOrByte='Both')
            test_dataset=HDF5_Mapp_Dataset(hdf5_filepath=test_dataset_path, packetorbyte='Both')
            test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=200, pin_memory=False,drop_last=True)#batch_size整数倍所有data都有 44*100
            # 接下来是训练过程
            # 例如使用 Adam 优化器，这里只会优化需要梯度的参数
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, fusion_model.parameters()), lr=1e-3)
            criterion=nn.CrossEntropyLoss()

            # 训练循环
            for model_epoch in range(fusion_model_epoch):
                print(f'fusion_model_epoch:{model_epoch}')
                loss_t=0
                for input1, input2, targets in train_queue:  # 要重写dataloader
                    optimizer.zero_grad()
                    outputs = fusion_model(input1.to(device), input2.to(device))
                    loss = criterion(outputs, targets.to(device))
                    loss_t+=loss.item()
                    loss.backward()
                    optimizer.step()
                print('loss:',loss_t/len(train_queue))



            fusion_model_save_path=model_path+f'/fusion_model/{valid_method}_Layer{layer}_step{step}'

            # 保存模型权重
            if not os.path.exists(fusion_model_save_path):
                os.makedirs(fusion_model_save_path)
            model_weights_path = fusion_model_save_path + '/weights.pth'
            torch.save(fusion_model.state_dict(), model_weights_path)

            print(f'____________________test fusion model, layer={layer}, step={step}____________________')

            fusion_model.eval()
            result_save_path = fusion_model_save_path+'/result_fusion.txt'

            test_model(fusion_model, device, test_queue, result_save_path)

            torch.cuda.empty_cache()
            del fusion_model






