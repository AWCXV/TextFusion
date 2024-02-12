# test phase
import torch
import clip
from PIL import Image
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
from torch.autograd import Variable
from net import TextFusionNet_t
import utils
from args_fusion import args
import numpy as np
import torch.nn.functional as F
import time
import numpy as np    
import cv2    
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from PIL import Image

def main():

    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    test_path = "./samples/"

    in_c = 2
    out_c = 1
    model_path = args.model_path_gray

    with torch.no_grad():

        model = load_model(model_path, in_c, out_c)
        output_path = 'output/';    
        if os.path.exists(output_path) is False:
            os.mkdir(output_path)
        for i in range(1):

            index = i+1
            infrared_path = test_path + 'ir.png'
            visible_path = test_path + 'vis.png'
            text_path = test_path + "description.txt";
            with open(text_path, 'r') as f:
                description = f.readline().strip();
            #description = "You can also modify the description here!"
            run_demo(device, clip_model, model, infrared_path, visible_path, description, output_path)
    print('Done......')


def load_model(path, input_nc, output_nc):

    TextFusionNet_model = TextFusionNet_t()

    TextFusionNet_model.load_state_dict(torch.load(path))
    TextFusionNet_model = torch.nn.DataParallel(TextFusionNet_model,device_ids=[0]);

    para = sum([np.prod(list(p.size())) for p in TextFusionNet_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(TextFusionNet_model._get_name(), para / 1000/1000))

    TextFusionNet_model.eval()

    return TextFusionNet_model
   

def run_demo(device,clip_model, model, infrared_path, visible_path, description, output_path_root):

    ir_img = cv2.imread(infrared_path, cv2.IMREAD_GRAYSCALE)
    vi_img = cv2.imread(visible_path, cv2.IMREAD_GRAYSCALE)
    
    text = clip.tokenize([description]).to(device)                
    description_features = clip_model.encode_text(text)    
    
    ir_img = ir_img / 255.0;
    vi_img = vi_img / 255.0;
    
    h = vi_img.shape[0];
    w = vi_img.shape[1];
    
    ir_img_patches = np.resize(ir_img,[1,1,h,w]);    
    vi_img_patches = np.resize(vi_img,[1,1,h,w]);    
    
    ir_img_patches = torch.from_numpy(ir_img_patches).float();
    vi_img_patches = torch.from_numpy(vi_img_patches).float();
    
    if args.cuda:
        ir_img_patches = ir_img_patches.cuda(args.device)
        vi_img_patches = vi_img_patches.cuda(args.device)
        model = model.cuda(args.device);
        
    output = model(vis = vi_img_patches, ir = ir_img_patches, text_features = description_features);
    fuseImage = np.zeros((h,w));
      
    out = output.cpu().numpy();
    
    fuseImage = out[0][0];
    
    fuseImage = fuseImage*255;   
    fuseImage = np.round(fuseImage).astype(np.uint8)
    fuseImage = Image.fromarray(fuseImage, mode='L');
    
    file_name = 'fused_gray.jpg'
    if os.path.exists(output_path_root) is False:
        os.mkdir(output_path_root)
    output_path = output_path_root + "/"+ file_name

    fuseImage.save(output_path);

    print(output_path)

if __name__ == '__main__':
    main()
