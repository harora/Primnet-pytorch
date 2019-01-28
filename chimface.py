import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import collections
import torch
import torchvision
from torch.utils import data
from PIL import Image
from matlab_cp2tform import get_similarity_transform_for_cv2


def alignment(src_img,src_pts):
    of = 2
    ref_pts = [ [30.2946+of, 51.6963+of],[65.5318+of, 51.5014+of],
        [48.0252+of, 71.7366+of],[33.5493+of, 92.3655+of],[62.7299+of, 92.2041+of] ]
    crop_size = (96+of*2, 112+of*2)

    s = np.array(src_pts).astype(np.float32)
    r = np.array(ref_pts).astype(np.float32)
    print s
    tfm = get_similarity_transform_for_cv2(s, r)
    face_img = cv2.warpAffine(src_img, tfm, crop_size)
    return face_img

class chimpface(data.Dataset):
    def __init__(self, root, list_path, max_iters=None, crop_size=(321, 321), mean=(128, 128, 128), scale=True, mirror=False, ignore_label=255,shuffle=True):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.scale = scale
        self.ignore_label = ignore_label
        self.mean = mean
        self.shuffle=shuffle
        self.is_mirror = mirror
        # print list_path
        self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id for i_id in open(list_path)]
        # print self.img_ids
        if not max_iters==None:
	    self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []


        for name in self.img_ids:
            nams=name.split(' ')
            nm=nams[0]
            img_file = osp.join(self.root, nm)
            label=nams[1]
            coords=[nams[3],nams[4],nams[6],nams[7],nams[9],nams[10].strip()]
            # print coordsx
            # print coordsy
            self.files.append({
                "img": img_file,
                "label": label,
                "name": name,
                "coords": coords
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        # label = self.nametoidx[datafiles["label"]]
        name = datafiles["name"]
        label= int(datafiles["label"])
        coords = datafiles["coords"]


        image = image.resize(self.crop_size, Image.BICUBIC)

        image = np.asarray(image, np.float32)

        size = image.shape

        image = image.transpose((2, 0, 1))
        image=image-127.5
        image/=128

        return image.copy(), label,name#, np.array(size), name


if __name__ == '__main__':
    dst = chimpface("chimface/chimpface/data_CZoo/",list_path='chimface/chimpface/data_CZoo/annotations_czoo.txt')

