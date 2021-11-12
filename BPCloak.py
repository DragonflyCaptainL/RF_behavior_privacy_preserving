import numpy as np 
import torch
import torch.nn as nn 
import torch.nn.functional as F 
# import torchvision 
import torch.optim as optim 
# from torchvision import transforms
# import pickle
import random
import torch.utils.data as Data 
# import matplotlib.pyplot as plt
from torch.nn.modules.loss import CrossEntropyLoss

signal_path = r'signal_10sub_504_10.npy'
label_path = r'label_10sub_504_10.npy' # format [identity label, gesture label]

def getTrainData():
#     print((np.array([[1,2,3,4]])==np.array([[1,3,2,4]])).all())
    signal = np.load(signal_path,allow_pickle=True)
    label = np.load(label_path,allow_pickle=True)
    
    return signal, label


def notin_check(a,b):
    flag = 1
    for i in b:
        if (i==a).all() == True:
            flag = 0
            break
    return flag

#r = 0
def contrastDatasetFormlization(signals,labels):
    train_data = []  # signal samples used for BPCloak training
    train_label = []
    test_data =  []  # signal samples used for BPCloak testing
    test_label = []
    check_list = [] 
    
    i = 0
    while(i<1000):    # constructing a training set with 1000 training samples 
        a = random.randrange(0,len(signals))
        b = random.randrange(0,len(signals))
        if (labels[a][0] == labels[b][0]) and (labels[a][1] != labels[b][1]):
            tmp = []
            tmp.append(signals[a])
            tmp.append(signals[b])
            if notin_check(signals[a], check_list):
                check_list.append(signals[a])
            if notin_check(signals[b], check_list):
                check_list.append(signals[b])
            train_data.append(np.array(tmp))
            train_label.append(np.array([0,labels[a][0]])) # similar
            i += 1
        elif (labels[a][0] != labels[b][0]) and (labels[a][1] == labels[b][1]):
            tmp = []
            tmp.append(signals[a])
            tmp.append(signals[b])
            if notin_check(signals[a], check_list):
                check_list.append(signals[a])
            if notin_check(signals[b], check_list):
                check_list.append(signals[b])
            train_data.append(np.array(tmp))
            train_label.append(np.array([1,-1])) # dissimilar
            i += 1
    train_data = np.array(train_data)
    train_label = np.array(train_label)
#     np.save(r'1000_train_data.npy',train_data)
#     np.save(r'1000_train_label.npy',train_label)
    print('training set saved')
    
    
    print("test_data constructing...")
    
    for j in range(len(signals)):
        if notin_check(signals[j], check_list):
            test_data.append(signals[j])
            test_label.append(labels[j])
    test_data = np.array(test_data); test_label = np.array(test_label)
    
#     np.save(r'test_data.npy',test_data)
#     np.save(r'test_label.npy',test_label)  
    
    print('testing set saved')
          
    return np.array(train_data), np.array(train_label), np.array(test_data), np.array(test_label)

def loadData(train_batch_size, test_batch_size):
    
    signals, labels = getTrainData()
    train_data, train_label, test_data, test_label = contrastDatasetFormlization(signals,labels)

    train_data = torch.from_numpy(train_data).float()
    train_label = torch.from_numpy(train_label).float()
    
    torch_train = Data.TensorDataset(train_data,train_label)
#     torch_test = Data.TensorDataset(test_data,test_label)
    
    train_loader = Data.DataLoader(dataset = torch_train,
                             batch_size = train_batch_size,
                             shuffle=False)

    return train_loader,test_data, test_label


class BPCloak(nn.Module):
    def __init__(self):
        super(BPCloak, self).__init__()
        self.cnn1 = nn.Sequential(
            nn.Conv2d(1, 16,kernel_size=3, stride=(2,1),padding = 1),
            nn.BatchNorm2d(16),
             
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=.2),
            nn.Conv2d(16, 32,kernel_size=3, stride=(2,1),padding =1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=.2),
            nn.Conv2d(32, 64,kernel_size=3, stride=(2,1),padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=.2)
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64*63*10, 256),
            nn.ReLU(inplace=True),

            nn.Linear(256,64),
#             nn.ELU()
        )
        self.id = nn.Sequential(
            nn.Linear(64,128),
            nn.Sigmoid(),
            nn.Linear(128,10),
            nn.Sigmoid()
            )
    def forward_once(self, x):
        output = self.cnn1(x)
#         print(output.size())
        output = output.view(output.size()[0], -1)
        
        output = self.fc1(output)
        return output

    def forward(self, input1, input2):
#         print(input1.size())
        input1 = input1.view(-1,1,504,10)
        input2 = input2.view(-1,1,504,10)
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        id_out = self.id(output1)
        return output1, output2, id_out
    
    
class ContrastiveLoss(torch.nn.Module):
    
    def __init__(self, margin=2.5):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2) + (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        
#         loss_contrastive = torch.mean((1-label) * torch.pow(euclidean_distance, 2)  
#                                       (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))


        return loss_contrastive
    
if __name__ == "__main__": 
    device = torch.device('cuda')
    train_loader, test_data, test_label =  loadData(1,1) # two parameters: bactSize for training set and testing set
    
    net = BPCloak().to(device)
    criterion = ContrastiveLoss()
    id_criterion = CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
    
    counter = []
    loss_history = [] 
    iteration_number= 0
    
    for epoch in range(100):
        print('epoch =   :', epoch)
        for i,(train_data, train_label) in enumerate(train_loader):
#             print(train_data.size())
            img0, img1  = train_data[:,0:1].to(device), train_data[:,1:2].to(device)
            simi_label, id_label = train_label[:,0:1].to(device), train_label[:,1:2].view(-1).long().to(device)
            
#             print(img0.size())
#             train_label = train_label.view(-1)
            output1,output2, id_out = net(img0,img1)
            if id_label[0] == -1:
                optimizer.zero_grad()
                loss_contrastive = criterion(output1,output2,simi_label)

                loss_contrastive.backward()
                optimizer.step()
            else:
                optimizer.zero_grad()
                loss_contrastive = criterion(output1,output2,simi_label)
                loss_id = id_criterion(id_out,id_label) 
                            
                wei=0.5
                
                loss = loss_id * (1-wei)  + loss_contrastive * wei
                loss.backward()
                optimizer.step()  
            
            if (i+1) %5 == 0 :
#                 print("Epoch number {}\n Current loss {}\n".format(epoch,loss_contrastive.data[0]))
                iteration_number += 5
                counter.append(iteration_number)
                loss_history.append(loss_contrastive.item())
