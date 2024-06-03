import pickle

import numpy as np
from operations import *
from genotypes import Genotype
import os
import shutil
import torch
import torchvision.transforms as transforms

def count_params(model):
    return sum(np.prod(v.shape) for name,v in model.named_parameters())/1e6

def parse_actions_index(actions_index,steps=5):
    # steps = step
    normal = []
    reduce = []
    normal_concat = set(range(2,2+steps))
    reduce_concat = set(range(2,2+steps))

    for i in range(2*steps):
        node1 = int(actions_index[i*5])
        node2 = int(actions_index[i*5+1])

        op1 = OP_NAME[actions_index[i*5+2]]
        op2 = OP_NAME[actions_index[i*5+3]]

        comb = COMB_NAME[actions_index[i*5+4]]

        block = (node1, node2, op1, op2, comb)
        if i < steps:
            if node1 in normal_concat:
                normal_concat.remove(node1)
            if node2 in normal_concat:
                normal_concat.remove(node2)
            normal.append(block)
        else:
            if node1 in reduce_concat:
                reduce_concat.remove(node1)
            if node2 in reduce_concat:
                reduce_concat.remove(node2)
            reduce.append(block)

    genotype = Genotype(normal = normal, normal_concat = normal_concat,
                        reduce = reduce, reduce_concat = reduce_concat)

    return genotype

def accuracy(logits, target, topk=(1,)):
    assert logits.shape[0]==target.shape[0]
    batch_size = logits.shape[0]
    result = []
    maxk = max(topk)
    target = target.view(-1,1)
    _, pred = torch.topk(logits, maxk, 1, True, True)
    # print("pred:{}".format(pred.shape))
    # print(pred)

    for k in topk:
        predk = pred[:,:k]
        targetk = target.expand_as(predk)
        correct = torch.eq(predk, targetk)
        correct_num = torch.sum(torch.sum(correct, 1),0)
        result.append(float(correct_num)/batch_size)
        # print(k)
    # print("predk:{} targetk:{} correct:{} correct_num:{}".format(predk.shape,targetk.shape,correct.shape,correct_num.shape) )
    # print(targetk)
    return result

def one_hot(index, num_classes):
    v = torch.zeros((num_classes), dtype=torch.float)
    v[int(index)] = 1
    return v

def create_exp_dir(path, scripts_to_save=None):
  if not os.path.exists(path):
    os.mkdir(path)
  print('Experiment dir : {}'.format(path))

  if scripts_to_save is not None:
    os.mkdir(os.path.join(path, 'scripts'))
    for script in scripts_to_save:
      dst_file = os.path.join(path, 'scripts', os.path.basename(script))
      shutil.copyfile(script, dst_file)

class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def _data_transforms_cifar10(args):
  CIFAR_MEAN = [0.5]
  CIFAR_STD = [0.25]

  train_transform = transforms.Compose([
    transforms.RandomCrop(28, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
  ])
  if args.cutout:
    train_transform.transforms.append(Cutout(args.cutout_length))

  valid_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
  return train_transform, valid_transform


import json

import torch
import json



def save_model(genotype, num_classes, layers, steps, model, save_path):
    """将四个变量和模型权重保存到指定的文件和路径。"""

    def namedtuple_to_dict(namedtuple_instance):
        """将namedtuple实例转换为字典"""
        return namedtuple_instance._asdict()


    # 准备数据
    data_to_save = {
        'genotype': genotype,
        'num_classes': num_classes,
        'layers': layers,
        'steps': steps
    }



    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 保存数据到pickle文件
    pickle_path = save_path+ '.pkl' # 使用join确保路径处理正确
    with open(pickle_path, 'wb') as f:
        pickle.dump(data_to_save, f)

    # 保存模型权重
    model_weights_path = save_path + '_weights.pth'
    torch.save(model.state_dict(), model_weights_path)


from model import Network
def load_model(load_path,num_embeddings, embedding_dim,map_location='cuda:0'):
    """从指定的文件读取变量和模型权重，并返回它们。"""
    # # 读取JSON文件
    # json_path = load_path + '.json'
    # with open(json_path, 'r') as f:
    #     data_loaded = json.load(f)

    # 读取pickle文件
    pickle_path = load_path+'.pkl' # 使用join确保路径处理正确
    with open(pickle_path, 'rb') as f:
        data_loaded = pickle.load(f)

    # 重新创建Genotype实例
    genotype_loaded = data_loaded['genotype']
    num_classes_loaded = data_loaded['num_classes']
    layers_loaded = data_loaded['layers']
    steps_loaded = data_loaded['steps']

    # 加载模型权重
    model_weights_path = load_path + '_weights.pth'
    # 注意：加载模型权重前，你需要先实例化你的模型
    model = Network(genotype_loaded, num_classes_loaded,layers=layers_loaded,steps=steps_loaded,num_embeddings=num_embeddings,embedding_dim=embedding_dim)
    model.load_state_dict(torch.load(model_weights_path,map_location=map_location))

    return model

def update_score_to_file(score, file_path):
    try:
        # 尝试读取文件中的分数
        with open(file_path, 'r') as file:
            existing_score = file.read().strip()
    except FileNotFoundError:
        # 如果文件不存在，直接写入新分数
        with open(file_path, 'w') as file:
            file.write(str(score))
            print(f"文件不存在，已创建文件并写入分数：{score}")
    else:
        # 如果文件存在但内容为空，或新分数大于文件中的分数，更新文件中的分数
        if not existing_score or score > float(existing_score):
            with open(file_path, 'w') as file:
                file.write(str(score))
                print(f"已更新分数为：{score}")
        else:
            print("当前分数未超过文件中的分数，不做更新。")

def read_score_from_file(file_path):
    try:
        # 尝试打开文件并读取内容
        with open(file_path, 'r') as file:
            score_str = file.read()
            # 确保文件内容不为空
            if score_str:
                # 将字符串分数转换为浮点数并返回
                return float(score_str)
            else:
                print("文件内容为空。")
                return None
    except FileNotFoundError:
        # 文件不存在
        print("文件不存在。")
        return None
    except ValueError:
        # 文件内容不是有效的浮点数
        print("文件内容不是有效的浮点数。")
        return None



