from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
torch.backends.cudnn.bencmark = True

import os,sys,cv2,random,datetime
import argparse
import numpy as np

from dataset import ImageDataset
from torch.utils import data
from matlab_cp2tform import get_similarity_transform_for_cv2
import net_sphere
from chimface import chimpface
# from sklearn.metrics.pairwise import euclidean_distances as ed


parser = argparse.ArgumentParser(description='PyTorch sphereface')
parser.add_argument('--net','-n', default='sphere4a', type=str)
parser.add_argument('--dataset', default='../../dataset/face/casia/casia.zip', type=str)
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--bs', default=64, type=int, help='')
args = parser.parse_args()
use_cuda = torch.cuda.is_available()

IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
def alignment(src_img,src_pts):
    of = 2
    ref_pts = [ [30.2946+of, 51.6963+of],[65.5318+of, 51.5014+of],
        [48.0252+of, 71.7366+of],[33.5493+of, 92.3655+of],[62.7299+of, 92.2041+of] ]
    crop_size = (96+of*2, 112+of*2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)

    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img





def printoneline(*argv):
    s = ''
    for arg in argv: s += str(arg) + ' '
    s = s[:-1]
    sys.stdout.write('\r'+s)
    sys.stdout.flush()

def save_model(model,filename):
    state = model.state_dict()
    for key in state: state[key] = state[key].clone().cpu()
    torch.save(state, filename)

def dt():
    return datetime.datetime.now().strftime('%H:%M:%S')



def train(epoch,args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    data_dir="chimface/chimpface/"
    data_list='chimface/chimpface/lists/split5train.txt'
    # ds = ImageDataset(args.dataset,dataset_load,'data/casia_landmark.txt',name=args.net+':train',
        # bs=args.bs,shuffle=True,nthread=6,imagesize=128)
    trainloader = data.DataLoader(
    chimpface(data_dir,data_list, max_iters=50000,
                crop_size=(112, 112), mean=IMG_MEAN),
    batch_size=args.bs, shuffle=True)

    trainloader_iter = enumerate(trainloader)
    
    for idx in range(750):
        _, batch = trainloader_iter.next()
        img,label,_=batch
        if img is None: break

        inputs=img
        targets=label

        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()

        optimizer.zero_grad()
        inputs, targets = Variable(inputs), Variable(targets)

        outputs,_ = net(inputs)
        loss = criterion(outputs, targets)
        lossd = loss.data[0]
        loss.backward()
        optimizer.step()

        train_loss += loss.data[0]
        outputs = outputs[0] # 0=cos_theta 1=phi_theta

        _, predicted = torch.max(outputs.data, 1)

        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()

        printoneline(str(idx),'Te=%d Loss=%.4f | AccT=%.4f%% (%d/%d) %.4f %.2f %d'
            % (epoch,train_loss/(batch_idx+1), 100.0*correct/total, correct, total, 
            lossd, criterion.lamb, criterion.it))
        batch_idx += 1
    print('')


from scipy.spatial import distance

def testcs(epoch,args):
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    batch_idx = 0
    data_dir="chimface/chimpface/"
    data_list='chimface/chimpface/lists/split5test.txt'
    
    trainloader = data.DataLoader(
    chimpface(data_dir,data_list, max_iters=50000,
                crop_size=(112, 112), mean=IMG_MEAN),
    batch_size=1, shuffle=False)

    trainloader_iter = enumerate(trainloader)
    features=[]
    names=[]
    for idx in range(1136):
        _, batch = trainloader_iter.next()
        img,label,name=batch
        if img is None: break

        inputs=img
        targets=label
        if use_cuda: inputs, targets = inputs.cuda(), targets.cuda()
        
        inputs, targets = Variable(inputs), Variable(targets)
        outputs,feature = net(inputs)
        
        features.append(feature.detach().cpu())
        names.append(name)

    correct=0
    
    for i in range(1136):
        fx=features[i]
        
        mindis=1000
        idx=19999
        for j in range(1136):  
            if j!=i:
                fy=features[j]
                d=distance.euclidean(fx,fy)
                # print(d)
                if d<mindis:
                    mindis=d
                    idx=j
        # print(names[i][0])       
        n1=names[i][0].strip().split(' ')[1]
        n2=names[idx][0].strip().split(' ')[1]
        if(n1==n2):
            correct+=1
            print(i,' ',float(correct)/(i+1))
            

    print(correct)
    # print(len(names))
        






net=net_sphere.sphere4a()
# net = getattr(net_sphere,args.net)()
net.load_state_dict(torch.load('sphere4a_12.pth'))
net.cuda()
criterion = net_sphere.AMLoss()


# print('start: time={}'.format(dt()))
# for epoch in range(0, 20):
#     if epoch in [0,10,15,18]:
#         if epoch!=0: args.lr *= 0.1
#         optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)

#     train(epoch,args)
#     save_model(net, '{}_{}.pth'.format(args.net,epoch))



testcs(1,args)

print('finish: time={}\n'.format(dt()))

