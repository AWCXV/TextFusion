# Training TextFusion network

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import sys
import time
from utils import gradient, gradient2
import numpy as np
from tqdm import tqdm, trange
import scipy.io as scio
import random
import torch
from torch.optim import Adam
from testMat import showLossChart
from torch.autograd import Variable
from densefuseNet import DenseFuse_net
from torchvision.models import resnet50
import utils
from net import TextFusionNet_t
from args_fusion import args
from utils import sumPatch
import pytorch_msssim
import torchvision.models as models
import torch.nn.functional as F
import clip
from PIL import Image

def load_model(path):

    denseFuseNet_model = DenseFuse_net()
    denseFuseNet_model.load_state_dict(torch.load(path))

    para = sum([np.prod(list(p.size())) for p in denseFuseNet_model.parameters()])
    type_size = 4
    print('Model {} : params: {:4f}M'.format(denseFuseNet_model._get_name(), para / 1000/1000))

    denseFuseNet_model.eval()

    return denseFuseNet_model

def main():
    # load pre-train models -------------- begin
    
    #clip
    device = "cuda" if torch.cuda.is_available() else "cpu"
    clip_model, preprocess = clip.load("ViT-B/32", device=device)

    #vgg
    vgg_model = models.vgg19(pretrained=True)
    if (args.cuda):
        vgg_model = vgg_model.cuda(args.device);
    vggFeatures = [];
    vggFeatures.append(vgg_model.features[:3]);#64
    vggFeatures.append(vgg_model.features[:8]);#32
    vggFeatures.append(vgg_model.features[:17]);#16
    vggFeatures.append(vgg_model.features[:26]);#8
    vggFeatures.append(vgg_model.features[:35]);#4    
    for i in range(0,5):
        for parm in vggFeatures[i].parameters():
            parm.requires_grad = False;                    
    
    #autoencoder
    model_path = "./DenseFuse.model";
    densefuseModel = load_model(model_path)
    if (args.cuda):
        densefuseModel = densefuseModel.cuda(args.device);            
    # load pre-train models -------------- end
            
    patchPrePath = "../dataset/IVT_train/";
    PatchPaths = utils.generateTrainNumberIndex()
    batch_size = args.batch_size
        
    if os.path.exists(args.save_loss_dir) is False:
        os.mkdir(args.save_loss_dir)        

    TextFusionNet_model = TextFusionNet_t();
    print(TextFusionNet_model);
    
    optimizer_TextFusionNet_model = Adam(TextFusionNet_model.parameters(), args.lr)

    mse_loss = torch.nn.MSELoss(reduction="mean")
    l1_loss = torch.nn.L1Loss(reduction="mean");
    ssim_loss = pytorch_msssim.msssim
    if (args.cuda):
        TextFusionNet_model.cuda(int(args.device));

    tbar = trange(args.epochs)
    print('Start training.....')

    Record_matrix_total_loss = []
    Record_matrix_content_loss = []
    Record_matrix_decomposition_loss = []

    for e in tbar:
        TextFusionNet_model.train()
        print('Epoch %d.....' % e)
        patchesPaths, batches = utils.load_datasetPair(PatchPaths,batch_size);

        count = 0

        record_total_loss = 0.
        record_content_loss = 0.
        record_decomposition_loss = 0.

        batch = 0;        
        for i in range(batches):
            batch_inside = 0;        
            for textIndex in range(1,1+5):
                image_paths = patchesPaths[batch_inside * batch_size:(batch_inside * batch_size + batch_size)]
                batch_inside += 1;
                
                image_ir = utils.get_single_train_image(patchPrePath+"ir/"+image_paths[0]+".png");
                image_vi = utils.get_single_train_image(patchPrePath+"vis/"+image_paths[0]+".png");
                binaryInterestedRegions = utils.get_single_train_image(patchPrePath+"association/IVT_LLVIP_2000_imageIndex_"+image_paths[0]+"_textIndex_"+str(textIndex)+"/Final_Finetuned_BinaryInterestedMap.png");

                h = image_ir.shape[2];
                w = image_ir.shape[3];

                image_ir = F.interpolate(image_ir, scale_factor=0.5, mode='bilinear');
                image_vi = F.interpolate(image_vi, scale_factor=0.5, mode='bilinear');
                binaryInterestedRegions = F.interpolate(binaryInterestedRegions, scale_factor=0.5, mode='bilinear');
                
                ones = torch.ones_like(binaryInterestedRegions);
                binaryNonInterestedRegions = ones - binaryInterestedRegions;

                #batch_size equals to one
                text_path = patchPrePath + "text/" + image_paths[0] + "_" + str(textIndex) + ".txt";
                
                #load text content
                with open(text_path, 'r') as f:
                    description = f.readline().strip();
                    description = str(description);
                
                text = clip.tokenize([description]).to(device)                
                description_features = clip_model.encode_text(text)
                    
                h = image_ir.shape[2];
                w = image_ir.shape[3];

                optimizer_TextFusionNet_model.zero_grad();

                if args.cuda:
                    image_ir = image_ir.cuda(args.device)
                    image_vi = image_vi.cuda(args.device)
                    binaryInterestedRegions = binaryInterestedRegions.cuda(args.device);
                    binaryNonInterestedRegions = binaryNonInterestedRegions.cuda(args.device);
                
                #non-interested regions get gradient-based measurement  ---begin
                with torch.no_grad():
                    dup_ir = torch.cat([image_ir,image_ir,image_ir],1);
                    dup_vi = torch.cat([image_vi,image_vi,image_vi],1);
                    sum_g_ir = torch.zeros(1);
                    sum_g_vi = torch.zeros(1);
                    
                    if args.cuda:
                        sum_g_ir = sum_g_ir.cuda(args.device);
                        sum_g_vi = sum_g_vi.cuda(args.device);
                    depth_of_features = 5
                    myscale = 1;
                    num = 5;
                    tmpBinaryNonInterestedRegions = binaryNonInterestedRegions;
                    for j in range(depth_of_features):
                        g_ir = gradient(vggFeatures[j](dup_ir)).pow(2);
                        g_vi = gradient(vggFeatures[j](dup_vi)).pow(2);
                        
                        g_ir = g_ir.mean(dim = 1, keepdim=True);
                        g_vi = g_vi.mean(dim = 1, keepdim=True);
                        
                        g_ir = g_ir*tmpBinaryNonInterestedRegions;
                        g_vi = g_vi*tmpBinaryNonInterestedRegions;
                        
                        sum_non = torch.sum(tmpBinaryNonInterestedRegions);
                        if (sum_non.item()>0):                        
                            sum_g_ir = sum_g_ir + torch.sum(g_ir)/sum_non;
                            sum_g_vi = sum_g_vi + torch.sum(g_vi)/sum_non;
                        tmpBinaryNonInterestedRegions = F.interpolate(tmpBinaryNonInterestedRegions,scale_factor=0.5);
                    sum_g_ir /= depth_of_features;
                    sum_g_vi /= depth_of_features;

                sum_g_ir/=4000;
                sum_g_vi/=4000;
                
                weightNonInterestedIR = torch.exp(sum_g_ir)/(torch.exp(sum_g_ir)+torch.exp(sum_g_vi));
                weightNonInterestedVI = torch.exp(sum_g_vi)/(torch.exp(sum_g_ir)+torch.exp(sum_g_vi));
                #non-interested regions get gradient-based measurement  ---end
                    
                    
                #interested regions get pixel-based measurement  ---begin                    
                denseFeaturesIR = densefuseModel.encoder(image_ir)[0];
                denseFeaturesVIS = densefuseModel.encoder(image_vi)[0];
                
                almIR = denseFeaturesIR.sum(dim=1,keepdim = True);
                almVIS = denseFeaturesVIS.sum(dim=1,keepdim = True);
                
                weightInterestedIR = torch.exp(almIR)/(torch.exp(almIR)+torch.exp(almVIS));
                weightInterestedVIS = torch.exp(almVIS)/(torch.exp(almIR)+torch.exp(almVIS));
                
                #interested regions get pixel-based measurement  ---end
                
                fusedImage = TextFusionNet_model(vis = image_vi, ir = image_ir, text_features = description_features);
                
                #Loss function definition ---begin
                interestedLoss = mse_loss(binaryInterestedRegions*weightInterestedIR*fusedImage,binaryInterestedRegions*weightInterestedIR*image_ir)+\
                                mse_loss(binaryInterestedRegions*weightInterestedVIS*fusedImage,binaryInterestedRegions*weightInterestedVIS*image_vi);
                
                nonInterestedLoss = weightNonInterestedIR*mse_loss(binaryNonInterestedRegions*fusedImage,binaryNonInterestedRegions*image_ir)+\
                                weightNonInterestedVI*mse_loss(binaryNonInterestedRegions*fusedImage,binaryNonInterestedRegions*image_vi);            
                                
                totalLoss = interestedLoss + nonInterestedLoss;
                #Loss function definition ---end
                
                totalLoss.backward();
                optimizer_TextFusionNet_model.step();

                record_total_loss = totalLoss.item();
                record_content_loss += interestedLoss.item();
                record_decomposition_loss += nonInterestedLoss.item();            

                #Append loss matrix
                if (batch + 1) % args.log_loss_interval == 0:
                    mesg = "{}\tEpoch {}:\t[{}/{}]\t interestedLoss:{}\t noninterestedLoss:{}".format(
                        time.ctime(), e + 1, batch+1, batches*5, record_content_loss/args.log_loss_interval,record_decomposition_loss/args.log_loss_interval
                    )
                    tbar.set_description(mesg)
                    
                    Record_matrix_total_loss.append(record_total_loss/args.log_loss_interval);
                    Record_matrix_content_loss.append(record_content_loss/args.log_loss_interval);
                    Record_matrix_decomposition_loss.append(record_decomposition_loss/args.log_loss_interval);

                    record_total_loss = 0.
                    record_content_loss = 0.
                    record_decomposition_loss = 0.

                #Save loss data
                if (batch + 1) % args.log_model_interval == 0:
                    # save model
                    TextFusionNet_model.eval()
                    TextFusionNet_model.cpu()
                    save_model_filename = "SDNet_Epoch_" + str(e) + "_iters_" + str(batch+1) + ".model"
                    save_model_path = os.path.join(args.save_model_dir, save_model_filename)
                    torch.save(TextFusionNet_model.state_dict(), save_model_path)
                    
                    # Total loss
                    Record_matrix_nd_total_loss = np.array(Record_matrix_total_loss)
                    loss_filename_path = "TotalLoss_epoch_" + str(
                        args.epochs) + "_iters_" + str(batch+1) + ".mat"
                    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                    scio.savemat(save_loss_path, {'Loss': Record_matrix_nd_total_loss})
                    showLossChart(save_loss_path,args.save_loss_dir+'/totoal_loss.png')
                    
                    # Content loss
                    Record_matrix_nd_content_loss = np.array(Record_matrix_content_loss)
                    loss_filename_path = "ContentLoss_epoch_" + str(
                        args.epochs) + "_iters_" + str(batch+1) + ".mat"
                    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                    scio.savemat(save_loss_path, {'Loss': Record_matrix_nd_content_loss})
                    showLossChart(save_loss_path,args.save_loss_dir+'/interested_loss.png')

                    # Decomposition loss
                    Record_matrix_nd_decomposition_loss = np.array(Record_matrix_decomposition_loss)
                    loss_filename_path = "DecompositionLoss_epoch_" + str(
                        args.epochs) + "_iters_" + str(batch+1) + ".mat"
                    save_loss_path = os.path.join(args.save_loss_dir, loss_filename_path)
                    scio.savemat(save_loss_path, {'Loss': Record_matrix_nd_decomposition_loss})
                    showLossChart(save_loss_path,args.save_loss_dir+'/noninterested_loss.png')

                    TextFusionNet_model.train()
                    if (args.cuda):
                        TextFusionNet_model.cuda(int(args.device));
                    tbar.set_description("\nCheckpoint, trained model saved at", save_model_path)
            
                batch+=1;
    print("\nDone, trained model saved!")


if __name__ == "__main__":
    main()
