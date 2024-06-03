import numpy as np
import torch
from torch.autograd import Variable
import utils
import torch.nn as nn
from AutoMTCModeltest1NoTrain import confusion_matrix
import pickle


def returnPercentage(conf_matrix_element, per_kinds_element):
    if per_kinds_element == 0:
        return 0
    else:
        return (conf_matrix_element / per_kinds_element) * 100

path='../Data_Processsed'
def load_file(path):
    with open(path+'/'+'label_dic.txt', 'rb') as f:
        global label_dic
        label_dic=pickle.load(f)

    with open(path+'/'+'flow_list.txt', 'rb') as f:
        global flow_list
        flow_list=pickle.load(f)

    with open(path+'/'+'label_list.txt', 'rb') as f:
        global label_list
        label_list=pickle.load(f)


# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':
    labelcount = 20  # 这个数值是具体的分类数，大家可以自行修改
    load_file(path)
    # labels=label_list
    # labels = ['google-home-mini', 'blink-camera', 'magichome-strip', 'insteon-hub', 'xiaomi-hub', 'sousvide', 'lightify-hub', 'wansview-cam-wired', 'blink-security-hub', 'firetv', 'ring-doorbell', 'tplink-bulb', 't-philips-hub', 'appletv', 'nest-tstat', 'smartthings-hub', 'echoplus', 'sengled-hub', 'yi-camera', 'echodot', 'samsungtv-wired', 't-wemo-plug', 'echospot', 'tplink-plug', 'roku-tv'] # 每种类别的标签
    # labels=['blink-camera', 'magichome-strip', 'insteon-hub', 'lightify-hub', 'tplink-bulb', 't-philips-hub', 'appletv', 'nest-tstat', 'smartthings-hub', 'echoplus', 'samsungtv-wired', 't-wemo-plug', 'echospot', 'tplink-plug', 'roku-tv']
    # labels = ['blink-camera', 'insteon-hub', 'lightify-hub', 'tplink-bulb', 't-philips-hub', 'appletv', 'nest-tstat', 'samsungtv-wired', 't-wemo-plug', 'echospot', 'tplink-plug', 'roku-tv']
    # labels=['google-home-mini', 'blink-camera', 'magichome-strip', 'insteon-hub', 'lightify-hub', 'firetv', 'ring-doorbell', 'tplink-bulb', 't-philips-hub', 'appletv', 'nest-tstat', 'smartthings-hub', 'echoplus', 'yi-camera', 'echodot', 'samsungtv-wired', 't-wemo-plug', 'echospot', 'tplink-plug', 'roku-tv']
    labels=[x for x in  range(20)]
    device=torch.device("cuda:0")
    model=torch.load('0.75dictnew200.8965.pt')
    model.load_state_dict(torch.load("0.75dictnew200.8965.pt.dict"))


    model.to(device)
    # print(model)

    from nasnet.DataTXT2Tensor import get_Trainqueue_Validqueue

    train_queue, valid_queue, test_queue,label_dic = get_Trainqueue_Validqueue(train_portion=0.75,batch_size=8)

    # batch_num = len(train_queue)
    # criterion = nn.CrossEntropyLoss()
    # optimizer=torch.optim.SGD(model.parameters(),
    #                             0.001,
    #                             momentum=0.9,
    #                             weight_decay=3e-4)
    #
    # model.train()
    # # model=model.type(torch.FloatTensor).to(device)
    # for model_epoch in range(50):
    #
    #     i=0
    #     avg_acc = 0
    #     avg_loss = 0
    #     for batch, (input, target) in enumerate(train_queue):
    #
    #         input = input.type(torch.float32)
    #         # 增加了view
    #         # input = input.view(256, 64, 8, 8)
    #         # input = input.view(256, 64, 64, 1)
    #         input = Variable(input, requires_grad=False).to(device)
    #         target = Variable(target, requires_grad=False).to(device)
    #         # print('input.dtype:{} '.format(input.dtype))
    #         # print(target)
    #
    #         optimizer.zero_grad()
    #         logits = model(input)
    #         # logits=torch.log(logits)
    #         loss = criterion(logits, target)
    #         loss.backward()
    #         optimizer.step()
    #
    #         acc = utils.accuracy(logits.data, target.data)[0]
    #         avg_loss += float(loss)
    #         avg_acc += float(acc)
    #         print('train model_epoch: {} batch :{} valid loss {:.4f} acc {:.4f}'.format(model_epoch, i,avg_loss/batch_num,avg_acc/batch_num))
    #         i += 1
    #
    #
    #
    #
    #         del input, target, loss, logits
    # avg_loss=avg_loss/batch_num
    # avg_acc=avg_acc/batch_num
    # print('train  valid loss {:.4f} acc {:.4f}'.format(avg_loss, avg_acc))
    # # torch.save(model, 'TESTacc{:.4f}.pt'.format(avg_acc))



    conf_matrix = torch.zeros(labelcount, labelcount)
    avg_acc=0
    batch_num = len(test_queue)
    model.eval()
    for batch, (input, target) in enumerate(test_queue):
        with torch.no_grad():

            input = input.type(torch.float32)
            # input = input.view(256, 64, 8, 8)
            # input = input.view(256, 64, 64, 1)

            input = Variable(input).to(device)
            target = Variable(target).to(device)
            # print(target)

            logits = model(input)
            conf_matrix = confusion_matrix(logits, target.squeeze(), conf_matrix)

        acc = utils.accuracy(logits.data, target.data)[0]
        avg_acc += float(acc)

    ##训练集测试
    # for batch, (input, target) in enumerate(train_queue):
    #     with torch.no_grad():
    #
    #         input = input.type(torch.float32)
    #         input = input.view(256, 64, 8, 8)
    #         # input = input.view(256, 64, 64, 1)
    #
    #         input = Variable(input).to(device)
    #         target = Variable(target).to(device)
    #         # print(target)
    #
    #         logits = model(input)
    #         conf_matrix = confusion_matrix(logits, target.squeeze(), conf_matrix)
    #
    #     acc = utils.accuracy(logits.data, target.data)[0]
    #     avg_acc += float(acc)
    # batch_num=len(valid_queue)+len(train_queue)

    conf_matrix = conf_matrix.cpu()
    conf_matrix = np.array(conf_matrix.cpu())  # 将混淆矩阵从gpu转到cpu再转到np
    corrects = conf_matrix.diagonal(offset=0)  # 抽取对角线的每种分类的识别正确个数
    per_kinds = conf_matrix.sum(axis=1)  # 抽取每个分类数据总的测试条数


    print("混淆矩阵总元素个数：{0},测试集总个数:{1}".format(int(np.sum(conf_matrix)), len(valid_queue) * 256))
    print(conf_matrix)

    # 获取每种Emotion的识别准确率
    print("每种总个数：", per_kinds)
    print("每种预测正确的个数：", corrects)
    print("每种的识别准确率为：{0}".format([rate * 100 for rate in corrects / per_kinds]))

    valid_acc=avg_acc / batch_num
    print('valid acc{:.4f}'.format(valid_acc))


    from matplotlib import pyplot as plt

    # 绘制混淆矩阵

     # 显示数据
    plt.imshow(conf_matrix, cmap=plt.cm.Blues)

    # 在图中标注数量/概率信息
    thresh = conf_matrix.max() / 2  # 数值颜色阈值，如果数值超过这个，就颜色加深。
    for x in range(labelcount):
        for y in range(labelcount):
            # 注意这里的matrix[y, x]不是matrix[x, y]
            # info = int((conf_matrix[y, x]/per_kinds[x])*100)
            # info = int(conf_matrix[y, x] )
            info=int(returnPercentage(conf_matrix[y, x],per_kinds[x]))
            plt.text(x, y, info,
                     verticalalignment='center',
                     horizontalalignment='center',
                     color="white" if info > thresh else "black")

    plt.tight_layout()  # 保证图不重叠
    plt.yticks(range(labelcount), labels)
    plt.xticks(range(labelcount), labels, rotation=45)  # X轴字体倾斜45°
    plt.show()
    plt.close()

