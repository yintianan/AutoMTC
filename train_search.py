import numpy as np
from policy_gradient import PolicyGradient
from PPO import PPO
from random_search import RandomSearch


import torch.backends.cudnn as cudnn
import torch

import argparse
import logging
import time
import os
import sys

parser = argparse.ArgumentParser('minst')
#data
parser.add_argument('--data', type=str, default='./mnist')
parser.add_argument('--train_portion', type=float, default=0.75)
parser.add_argument('--batch_size', type=int, default=64) #bytes时候128
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=10, help='cutout length')
#model
parser.add_argument('--model_epochs', type=int, default=0)#15
parser.add_argument('--model_lr', type=float, default=0.001)
parser.add_argument('--model_lr_min', type=float, default=0.001)
parser.add_argument('--model_weight_decay', type=float, default=3e-4)
parser.add_argument('--model_momentum', type=float, default=0.9)
parser.add_argument('--init_channel', type=int, default=4)
parser.add_argument('--model_steps', type=int, default=4)
parser.add_argument('--model_layers', type=int, default=4)
parser.add_argument('--num_classes', type=int, default=18)
parser.add_argument('--num_embeddings', type=int, default=257)
parser.add_argument('--embedding_dim', type=int, default=128)
#architecture
parser.add_argument('--arch_epochs', type=int, default=0)#40
parser.add_argument('--arch_lr', type=float, default=3.5e-4)
parser.add_argument('--episodes', type=int, default=1)
parser.add_argument('--entropy_weight', type=float, default=1e-5)
parser.add_argument('--baseline_weight', type=float, default=0.95)
parser.add_argument('--embedding_size', type=int, default=32)
parser.add_argument('--algorithm', type=str, choices=['PPO', 'PG', 'RS'], default='PPO')
#PPO
parser.add_argument('--ppo_epochs', type=int, default=10)
parser.add_argument('--clip_epsilon', type=float, default=0.2)

parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=2, help='random seed')

#验证方法
parser.add_argument('--valid_method', type=str, choices=['F1','Acc','Recall', 'Precision'],default='F1',help='F1 Acc Recall Precision')
#包长or字节
parser.add_argument('--packet_lengthORBytes', type=str, choices=['Bytes','Length'],default='Length',help='Length Bytes')
parser.add_argument('--model_save_path', type=str,default='/genmodel')
# parser.add_argument('--packet_file_path', type=str,default='../CICdata/CIC_comparesion_ZD_package_length_100.csv')
# parser.add_argument('--byte_file_path', type=str,default='../CICdata/cic_256_16.txt')
parser.add_argument('--Mapp_hdf5_filepath', type=str,default='/home/yintianan/python_workspace/new_autoMTC_MApp_embeded/MAppdata/preprocessed_data_Packet_Byte64_label.hdf5')
args = parser.parse_args()


def main():
    exp_dir = 'search_{}_{}'.format(args.algorithm, time.strftime("%Y%m%d-%H%M%S"))
    if not os.path.exists(exp_dir):
        os.mkdir(exp_dir)
    log_format = '%(asctime)s %(message)s'
    logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                        format=log_format, datefmt='%m/%d %I:%M:%S %p')
    fh = logging.FileHandler(os.path.join(exp_dir, 'log.txt'))
    fh.setFormatter(logging.Formatter(log_format))
    logging.getLogger().addHandler(fh)

    logging.info('args = %s', args)

    if args.algorithm == 'PPO' or args.algorithm == 'PG':
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        if torch.cuda.is_available():
            device = torch.device('cuda:{}'.format(str(args.gpu)))
            cudnn.benchmark = True
            cudnn.enable = True
            logging.info('using gpu : {}'.format(args.gpu))
            torch.cuda.manual_seed(args.seed)
        else:
            device = torch.device('cpu')
            logging.info('using cpu')

        if args.algorithm == 'PPO':
            ppo = PPO(args, device)
            ppo.multi_solve_environment()#全程都在训练控制器
        elif args.algorithm == 'PG':
            pg = PolicyGradient(args, device)
            pg.multi_solve_environment()

    else:
        rs = RandomSearch(args)
        rs.multi_solve_environment()



if __name__ == '__main__':
    main()