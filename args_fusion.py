
class args():

    # training args
    epochs = 3 #"number of training epochs, default is 2"
    batch_size = 1
    trainNumber = 2000
    
    HEIGHT = 256
    WIDTH = 256
    PATCH_SIZE = 128;
    PATCH_STRIDE = 4;

    save_model_dir = "models" #"path to folder where trained model will be saved."
    save_loss_dir = "models/loss"  # "path to folder where trained model will be saved."

    image_size = 256 
    cuda = 1 #"set it to 1 for running on GPU, 0 for CPU"

    lr = 1e-4 #"learning rate, default is 1e-4"
    lr_light = 1e-4  # "learning rate, default is 0.001"
    log_loss_interval = 5 #"number of images after which the training loss is logged, default is 500"
    log_model_interval = 100
    
    device = 0;

    model_path_gray = "./models/LLVIP_pretrained.model";
    


