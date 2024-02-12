import os
from os import listdir, mkdir, sep
from os.path import join, exists, splitext
import random
import numpy as np
import torch
from torch import nn
from PIL import Image
from torch.autograd import Variable
from args_fusion import args
import cv2    
import torch.nn.functional as F
import matplotlib as mpl
from torchvision import datasets, transforms

#./images/IV_images
#./images/IV_images
def getTranImagePatches(prepath):
    patchesIR = [];
    patchesVIS = [];
    picIdx = 0;
    for idx in range(0+1,21+1):
        print("Decomposing"+str(idx)+"-th images...");
        imageIR = cv2.imread(prepath+'/IR'+str(idx)+'.png', cv2.IMREAD_GRAYSCALE)        
        imageVIS = cv2.imread(prepath+'/VIS'+str(idx)+'.png', cv2.IMREAD_GRAYSCALE)        
        h = imageIR.shape[0];
        w = imageIR.shape[1];
        print(str(h)+","+str(w));
        for i in range(0,h-args.PATCH_SIZE+1,args.PATCH_STRIDE):
            for j in range(0,w-args.PATCH_SIZE+1,args.PATCH_STRIDE):
                picIdx+=1;
                patchImageIR = imageIR[i:i+args.PATCH_SIZE,j:j+args.PATCH_SIZE];
                patchImageVIS = imageVIS[i:i+args.PATCH_SIZE,j:j+args.PATCH_SIZE];
                cv2.imwrite('./images/IV_patches/IR'+str(picIdx)+'.png', patchImageIR)
                cv2.imwrite('./images/IV_patches/VIS'+str(picIdx)+'.png', patchImageVIS)
    return patchesIR,patchesVIS;

def gradient2(x):
    dim = x.shape;
    if (args.cuda):
        x = x.cuda(int(args.device));
    kernel = [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]];
    #kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]];
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(dim[1],dim[1],1,1);
    weight = nn.Parameter(data=kernel,requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(int(args.device));
    gradMap = F.conv2d(x,weight=weight,stride=1,padding=1);
    #showTensor(gradMap);
    return gradMap;     
    
def gradient(x):
    dim = x.shape;
    if (args.cuda):
        x = x.cuda(int(args.device));
    #kernel = [[0.,1.,0.],[1.,-4.,1.],[0.,1.,0.]];
    kernel = [[1 / 8, 1 / 8, 1 / 8], [1 / 8, -1, 1 / 8], [1 / 8, 1 / 8, 1 / 8]];
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(dim[1],dim[1],1,1);
    weight = nn.Parameter(data=kernel,requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(int(args.device));
    pad = nn.ReflectionPad2d(1);        
    gradMap = F.conv2d(pad(x),weight=weight,stride=1,padding=0);
    #showTensor(gradMap);
    return gradMap;         
    
def sumPatch(x,k):
    dim = x.shape;
    if (args.cuda):
        x = x.cuda(int(args.device));
    kernel = np.ones((2*k+1,2*k+1));
    kernel = kernel/(1.0*(2*k+1)*(2*k+1));
    kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
    kernel = kernel.repeat(dim[1],dim[1],1,1)    
    weight = nn.Parameter(data=kernel,requires_grad=False);
    if (args.cuda):
        weight = weight.cuda(int(args.device));
    gradMap = F.conv2d(x,weight=weight,stride=1,padding=k);
    #showTensor(gradMap);
    return gradMap;    

def loadPatchesPairPaths2(directory):
    imagePatchesIR = [];
    imagePatchesVIS = [];
    for i in range(0+1,args.trainNumber+1):
        irPatchPath = directory+"/IR/"+str(i)+".png";
        visPatchPath = directory+"/VIS/"+str(i)+".png";
        imagePatchesIR.append(irPatchPath);
        imagePatchesVIS.append(visPatchPath);
    return imagePatchesIR,imagePatchesVIS;
    
def generateTrainNumberIndex():
    imagePatches = [];
    for i in range(0+1,args.trainNumber+1):
        imagePatches.append(str(i));
    return imagePatches;    

def list_images(directory):
    images = []
    names = []
    dir = listdir(directory)
    dir.sort()
    for file in dir:
        name = file.lower()
        if name.endswith('.png'):
            images.append(join(directory, file))
        elif name.endswith('.jpg'):
            images.append(join(directory, file))
        elif name.endswith('.jpeg'):
            images.append(join(directory, file))
        name1 = name.split('.')
        names.append(name1[0])

    return images


def tensor_load_rgbimage(filename, size=None, scale=None, keep_asp=False):
    img = Image.open(filename).convert('RGB')
    if size is not None:
        if keep_asp:
            size2 = int(size * 1.0 / img.size[0] * img.size[1])
            img = img.resize((size, size2), Image.ANTIALIAS)
        else:
            img = img.resize((size, size), Image.ANTIALIAS)

    elif scale is not None:
        img = img.resize((int(img.size[0] / scale), int(img.size[1] / scale)), Image.ANTIALIAS)
    img = np.array(img).transpose(2, 0, 1)
    img = torch.from_numpy(img).float()
    return img


def tensor_save_rgbimage(tensor, filename, cuda=True):
    if cuda:
        # img = tensor.clone().cpu().clamp(0, 255).numpy()
        img = tensor.cpu().clamp(0, 255).data[0].numpy()
    else:
        # img = tensor.clone().clamp(0, 255).numpy()
        img = tensor.clamp(0, 255).numpy()
    img = img.transpose(1, 2, 0).astype('uint8')
    img = Image.fromarray(img)
    img.save(filename)


def tensor_save_bgrimage(tensor, filename, cuda=False):
    (b, g, r) = torch.chunk(tensor, 3)
    tensor = torch.cat((r, g, b))
    tensor_save_rgbimage(tensor, filename, cuda)


def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def matSqrt(x):
    U,D,V = torch.svd(x)
    return U * (D.pow(0.5).diag()) * V.t()

def load_datasetPair(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))
    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        image_path = image_path[:-mod];    
    num_imgs-=mod
    original_img_path = image_path[:num_imgs]

    # random
    random.shuffle(original_img_path)
    batches = int(len(original_img_path) // BATCH_SIZE)
    return original_img_path, batches

# load training images
def load_dataset(image_path, BATCH_SIZE, num_imgs=None):
    if num_imgs is None:
        num_imgs = len(image_path)
    original_imgs_path = image_path[:num_imgs]
    # random
    random.shuffle(original_imgs_path)
    mod = num_imgs % BATCH_SIZE
    print('BATCH SIZE %d.' % BATCH_SIZE)
    print('Train images number %d.' % num_imgs)
    print('Train images samples %s.' % str(num_imgs / BATCH_SIZE))

    if mod > 0:
        print('Train set has been trimmed %d samples...\n' % mod)
        original_imgs_path = original_imgs_path[:-mod]
    batches = int(len(original_imgs_path) // BATCH_SIZE)
    return original_imgs_path, batches


def get_image(path, height=256, width=256, mode='L'):
    if mode == 'L':
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)            
    elif mode == 'RGB':
        image = Image.open(path).convert('RGB')
    image = image/255;
    return image


def get_train_images_auto2(paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images

def get_train_images_auto(pre, paths, height=256, width=256, mode='RGB'):
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(pre+"/"+path+".png", height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            image = np.reshape(image, [image.shape[2], image.shape[0], image.shape[1]])
        images.append(image)

    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float();
    return images
    
def get_single_train_image(path):
    images = []
    mode = 'L';
    image = get_image(path, 0, 0, mode='L')
    if mode == 'L':
        image = np.reshape(image, [1, image.shape[0], image.shape[1]]);
        
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float();
    return images


def get_test_images(paths, height=None, width=None, mode='RGB'):
    ImageToTensor = transforms.Compose([transforms.ToTensor()])
    if isinstance(paths, str):
        paths = [paths]
    images = []
    for path in paths:
        image = get_image(path, height, width, mode=mode)
        if mode == 'L':
            image = np.reshape(image, [1, image.shape[0], image.shape[1]])
        else:
            # test = ImageToTensor(image).numpy()
            # shape = ImageToTensor(image).size()
            image = ImageToTensor(image).float().numpy()*255
    images.append(image)
    images = np.stack(images, axis=0)
    images = torch.from_numpy(images).float()
    return images


# colormap
def colormap():
    return mpl.colors.LinearSegmentedColormap.from_list('cmap', ['#FFFFFF', '#98F5FF', '#00FF00', '#FFFF00','#FF0000', '#8B0000'], 256)

