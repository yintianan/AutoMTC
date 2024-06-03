import matplotlib.pyplot as plt
import numpy as np

def get_acc(path):
    f = open(path, 'r')
    top1_acc = []
    top5_acc = []
    top20_acc = []
    for line in f:
        if 'arch_epoch' in line and 'top1_acc' in line:
            print(line)
            line = line.rstrip('\n').split(' ')
            top1_acc.append(float(line[line.index('top1_acc')+1]))
            top5_acc.append(float(line[line.index('top5_avg_acc') + 1]))
            top20_acc.append(float(line[line.index('top20_avg_acc') + 1]))
    return top1_acc[:100], top5_acc[:100], top20_acc[:100]

def draw_acc(PPO_path, out_path):

    ppo_top1, ppo_top5, ppo_top20 = get_acc(PPO_path)
    epochs = np.linspace(0, 59, 60)

    plt.figure()




    plt.plot(epochs, ppo_top1, label='top1 acc', color='red')
    plt.plot(epochs, ppo_top5, label='top5 acc', linestyle='-.', color='green')
    plt.plot(epochs, ppo_top20, label='top20 acc', linestyle='--', color='blue')

    plt.legend(loc='lower right')
    plt.title('Validation Accuracy')
    plt.xlabel('Search Epoch')
    plt.ylabel('Accuracy at 5 Epochs')
    plt.savefig(out_path)

draw_acc('search_PPO_20221108-161132/log.txt', 'search_PPO_20221108-161132/search.png')