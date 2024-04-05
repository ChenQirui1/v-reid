import yaml
import eval
import torch
import time
from tqdm import tqdm
from networks.vehiclenet.model import ft_net,ft_net_dense
from torch.autograd import Variable
from dataset.dataset import VeriDataset
from torchvision import transforms
from torch.utils.data import DataLoader
######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.
#

def get_dataset(dataset_name, query_dir, query_list, gallery_dir, gallery_list):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    # scale_size = args.scale_size
    # crop_size = args.crop_size

    if dataset_name == "veri":
        data_transform = transforms.Compose(
            [
                transforms.Resize((256, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(256, 128),
                transforms.ToTensor(),
                normalize,
            ]
        )

        query_set = VeriDataset(query_dir, query_list, data_transform, is_train=False)
        gallery_set = VeriDataset(
            gallery_dir, gallery_list, data_transform, is_train=False
        )

    query_loader = DataLoader(
        dataset=query_set,
        num_workers=4,
        batch_size=opt.batchsize,
        shuffle=False,
    )
    gallery_loader = DataLoader(
        dataset=gallery_set,
        num_workers=4,
        batch_size=opt.batchsize,
        shuffle=False,
    )

    image_datasets = {'query': query_set, 'gallery': gallery_set}

    dataloaders = {'query': query_loader, 'gallery': gallery_loader}

    return image_datasets, dataloaders

def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders):
    #features = torch.FloatTensor()
    # count = 0
    pbar = tqdm()
    if opt.linear_num <= 0:
        if opt.use_dense:
            opt.linear_num = 1024
        # elif opt.use_efficient:
        #     opt.linear_num = 1792
        # elif opt.use_NAS:
        #     opt.linear_num = 4032
        else:
            opt.linear_num = 2048

    for iter, data in enumerate(dataloaders):
        # img, label = data
        img, label, _ = data
        print(img.shape)
        n, _, _, _ = img.size()
        # count += n
        # print(count)
        pbar.update(n)
        # ff = torch.Tensor(n,opt.linear_num).zero_().cuda()
        # img = img
        ff = torch.zeros(n,opt.linear_num)

        # if opt.PCB:
        #     ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        # for i in range(2):
        #     if(i==1):
        #         img = fliplr(img)
        #     input_img = Variable(img.cuda())
        #     for scale in ms:
        #         if scale != 1:
        #             # bicubic is only  available in pytorch>= 1.1
        #             input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
        #         outputs = model(input_img) 
        #         ff += outputs

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            
            input_img = Variable(img.cuda())
            # print(img.shape)

            outputs = model(img) 
            # print(outputs)
            ff += outputs
 
        # # norm feature
        if opt.PCB:
            # feature size (n,2048,6)
            # 1. To treat every part equally, I calculate the norm for every 2048-dim part feature.
            # 2. To keep the cosine score==1, sqrt(6) is added to norm the whole feature (2048*6).
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True) * np.sqrt(6) 
            ff = ff.div(fnorm.expand_as(ff))
            ff = ff.view(ff.size(0), -1)
        else:
            fnorm = torch.norm(ff, p=2, dim=1, keepdim=True)
            ff = ff.div(fnorm.expand_as(ff))

        
        if iter == 0:
            features = torch.FloatTensor( len(dataloaders.dataset), ff.shape[1])
        #features = torch.cat((features,ff.data.cpu()), 0)
        start = iter*opt.batchsize
        end = min( (iter+1)*opt.batchsize, len(dataloaders.dataset))
        features[ start:end, :] = ff
    pbar.close()
    return features

######################################################################
# Load model
#---------------------------
def load_network(network):

    # name = 'ft_ResNet50'
    # save_path = os.path.join('./implementations/vehiclenet/model',name,args.weights)
    try:
        network.load_state_dict(torch.load(args.weights))
    except Exception as e: 
        print(e)
        # if torch.cuda.get_device_capability()[0]>6 and len(opt.gpu_ids)==1 and int(version[0])>1: # should be >=7
        #     print("Compiling model...")
        #     # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
        #     torch.set_float32_matmul_precision('high')
        #     network = torch.compile(network, mode="default", dynamic=True) # pytorch 2.0
        # network.load_state_dict(torch.load(save_path))

    return network

@eval.eval_loop
def main(args,**kwargs):
    # get yaml
    with open("./implementations/vehiclenet/model/ft_ResNet50/opts.yaml", 'r') as stream:
            config = yaml.load(stream, Loader=yaml.FullLoader) # for the new pyyaml via 'conda install pyyaml

    global opt
    opt = args
    opt.linear_num = 0
    # opt.use_swin = config['use_swin']
    opt.use_dense = config['use_dense']

    model_structure = ft_net(576)

    model = load_network(model_structure)

    dataloaders = get_dataset("VeRi", args.querypath, args.querylist, args.gallerypath, args.gallerylist)

    # Extract feature
    since = time.time()
    with torch.no_grad():
        gallery_feature = extract_feature(model,dataloaders=dataloaders['gallery_loader'])
        query_feature = extract_feature(model,dataloaders=dataloaders['query_loader'])



    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.2f}s'.format(
                time_elapsed // 60, time_elapsed % 60))

    return gallery_feature, query_feature

    
if __name__ == "__main__":
    main()