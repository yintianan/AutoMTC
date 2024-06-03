import pickle

import torch

from nasnet.Prodataset.mydataset import mydataset

label_dic={}
label_count=0
label_digt=[]

flow_list=[]
label_list=[]

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





# print('ss')
def get_Trainqueue_Validqueue(train_portion=0.75,batch_size=16):
    load_file(path)
    dataset=mydataset(flow_list,label_list,label_dic)
    ll=dataset.__len__()
    test_portion=(1-train_portion)/2
    valid_portion=(1-train_portion)/2
    valid_set_length=int(valid_portion*len(dataset))
    test_set_length =int(test_portion*len(dataset))
    train_set_length=len(dataset)-test_set_length-valid_set_length

    train_dataset,valid_dataset,test_dataset=torch.utils.data.random_split(dataset,[train_set_length,valid_set_length,test_set_length],generator=torch.Generator().manual_seed(42))
    # print(list(test_dataset))

    # print('{} {}'.format(len(train_dataset),len(test_dataset)))


    train_queue = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,pin_memory=False,drop_last=True)
    valid_queue=torch.utils.data.DataLoader(valid_dataset,batch_size=batch_size,pin_memory=False,drop_last=True)
    test_queue = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, pin_memory=False, drop_last=True)

    # for flow,label in valid_queue:
    #
    #     # flow=torch.from_numpy(np.array(flow))
    #     # print(type(flow))
    #     print(flow.size())#([batch_size, 784])
    #
    #     flow_view=flow.view(batch_size,1,28,28)
    #     print(flow_view.size())#([batch_size, 1, 28, 28])
    #     # print(label)
    #     print('ss')
    #     break
    return train_queue,valid_queue,test_queue,label_dic


if __name__ =='__main__':
    # with open('Data_Processsed/label_dic.txt', 'rb') as f:
    #     label_dic = pickle.load(f)
    #
    # with open('Data_Processsed/flow_list.txt', 'rb') as f:
    #     flow_list = pickle.load(f)
    #
    # with open('Data_Processsed/label_list.txt', 'rb') as f:
    #     label_list = pickle.load(f)
    path= '../Data_Processsed'
    load_file(path)

    train_queue, valid_queue,_,label_dic=get_Trainqueue_Validqueue()
    print(label_dic)

    labelset=set( )
    for i,(flow,label) in enumerate(train_queue):
        # torch.set_printoptions(profile="full")
        # print(flow.size())
        # flow=torch.unsqueeze(flow,3)
        # print(flow.size())
        # print(flow.dtype)
        # print(label.dtype)
        # print(label)


        print(label)
        break

    print(sorted(label_dic))
    # print(labelset)
    # print(len(labelset))

