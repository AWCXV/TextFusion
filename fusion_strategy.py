import torch
import torch.nn.functional as F

EPSILON = 1e-10


# addition fusion strategy
def addition_fusion(tensor1, tensor2):
    return (tensor1 + tensor2)/2

def MAXFusion(tensor1, tensor2):
    return torch.max(tensor1,tensor2);

# attention fusion strategy, average based on weight maps
def L1Fusion(tensor1, tensor2):
    # avg, max, nuclear
    f_spatial = spatial_fusion(tensor1, tensor2)
    tensor_f = f_spatial
    return tensor_f
    
def SCFusion(tensor1,tensor2):
    f_spatial = spatial_fusion(tensor1, tensor2);
    f_channel = channel_fusion(tensor1, tensor2);
    a = 0;
    print("a="+str(a));
    tensor_f = a*f_spatial + (1-a)*f_channel;
    return tensor_f;
    
def channel_fusion(tensor1, tensor2):
    shape = tensor1.size()
    global_p1 = channel_attention(tensor1)
    global_p2 = channel_attention(tensor2)

    global_p_w1 = global_p1 / (global_p1+global_p2+EPSILON)
    global_p_w2 = global_p2 / (global_p1+global_p2+EPSILON)

    global_p_w1 = global_p_w1.repeat(1,1,shape[2],shape[3])
    global_p_w2 = global_p_w2.repeat(1,1,shape[2],shape[3])

    tensorf = global_p_w1 * tensor1 + global_p_w2 * tensor2

    return tensorf    

def channel_attention(tensor, pooling_type = 'avg'):
    shape = tensor.size()
    global_p = F.avg_pool2d(tensor,kernel_size=shape[2:])
    return global_p

def spatial_fusion(tensor1, tensor2, spatial_type='sum'):
    shape = tensor1.size()
    spatial1 = spatial_attention(tensor1, spatial_type)
    spatial2 = spatial_attention(tensor2, spatial_type)
    spatial_w1 = torch.exp(spatial1) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w2 = torch.exp(spatial2) / (torch.exp(spatial1) + torch.exp(spatial2) + EPSILON)
    spatial_w1 = spatial_w1.repeat(1, shape[1], 1, 1)
    spatial_w2 = spatial_w2.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1 * tensor1 + spatial_w2 * tensor2

    return tensor_f


def spatial_attention(tensor, spatial_type='sum'):
    if spatial_type == 'mean':
        spatial = tensor.mean(dim=1, keepdim=True)
    elif spatial_type == 'sum':
        spatial = tensor.sum(dim=1, keepdim=True)
    return spatial




