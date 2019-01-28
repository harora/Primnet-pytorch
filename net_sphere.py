import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn import Parameter
import math



class AMLinear(nn.Module):
    def __init__(self, in_features, out_features, m = 0.35, phiflag=True):
        super(AMLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features,out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2,1,1e-5).mul_(1e5)
        self.phiflag = phiflag
        self.m = m
        self.s=10

    def forward(self, input):
        x = input   # size=(B,F)    F is feature len
        w = self.weight # size=(F,Classnum) F=in_features Classnum=out_features
        
        x=x.renorm(2,1,1e-5)
        
        xlen = x.pow(2).sum(1).pow(0.5) # size=B
        wlen = w.pow(2).sum(0).pow(0.5) # size=Classnum

        cos_theta = x.mm(w) # size=(B,Classnum)
        
        cos_theta = cos_theta / xlen.view(-1,1) / wlen.view(1,-1)

        cos_theta = cos_theta.clamp(-1,1)
        phi=cos_theta - self.m       
        

        output = (cos_theta,phi)

        return output # size=(B,Classnum,2)


class AMLoss(nn.Module):
    def __init__(self, gamma=0):
        super(AMLoss, self).__init__()
        self.gamma   = gamma
        self.it = 0
        self.LambdaMin = 5.0
        self.LambdaMax = 1500.0
        self.lamb = 1500.0
        self.nclass=72
        self.B=64
        self.S=30

    def forward(self, input, target):
        self.it += 1
        cos_theta,phi = input
        
        indices = torch.LongTensor(target.cpu()).view(-1,1)
        target_onehot = torch.zeros(self.B, self.nclass,dtype=torch.long)
        target_onehot = target_onehot.scatter(1,indices,1)

        adjusted_theta=self.S*torch.where(target_onehot.cuda()==1.,cos_theta,phi)

        m=nn.LogSoftmax()
        loss=nn.NLLLoss()
        inputs=m(adjusted_theta)

        los=loss(inputs,target)
        
        return los





class sphere4a(nn.Module):
    def __init__(self,classnum=72,feature=False):
        super(sphere4a, self).__init__()
        self.classnum = classnum
        self.feature = feature
        #input = B*3*112*112
        self.conv1_1 = nn.Conv2d(3,64,3,2,1)#,groups=32) #=>B*64*56*56
        self.relu1_1 = nn.PReLU(64)

        self.conv2_1 = nn.Conv2d(64,256,3,2,1)#,groups=32) #=>B*128*28*28
        self.relu2_1 = nn.PReLU(256)

        self.conv3_1 = nn.Conv2d(256,512,3,2,1)#,groups=32) #=>B*256*14*14
        self.relu3_1 = nn.PReLU(512)

        self.conv4_1 = nn.Conv2d(512,1024,3,2,1,groups=32) #=>B*512*7*7
        self.relu4_1 = nn.PReLU(1024)

        self.fc5 = nn.Linear(1024*7*7,512)
        self.fc6 = AMLinear(512,self.classnum)
        # self.fc7=nn.Linear(512,self.classnum)

    def forward(self, x):

        x = self.relu1_1(self.conv1_1(x))
        # x=self.conv1_1(x)

        x = self.relu2_1(self.conv2_1(x))
        # x=self.conv2_1(x)

        x = self.relu3_1(self.conv3_1(x))
        # x=self.conv3_1(x)

        x = self.relu4_1(self.conv4_1(x))
        # self.conv4_1(x)

        x = x.view(x.size(0),-1)
        x1 = self.fc5(x)
        # if self.feature: return x

        x = self.fc6(x1)
        return x,x1
