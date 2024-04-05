import eval
import torch
from eval_reid import extract_features
from networks.resnet import resnet50, resnet101
import os

@eval.eval_loop
def load_extract(args,**kwargs):
        # load network
    if args.backbone == "resnet50":
        model = resnet50(num_classes=args.num_classes)
    elif args.backbone == "resnet101":
        model = resnet101(num_classes=args.num_classes)

    print(args.weights)
    
    if args.weights != "":
        try:
            model = torch.nn.DataParallel(model)
            print("current path:", os.path.realpath(__file__))
            ckpt = torch.load(args.weights)
            print(ckpt["state_dict"])
            model.load_state_dict(ckpt["state_dict"])
            print("!!!load weights success !!! path is ", args.weights)
        except Exception as e:
            print("!!!load weights failed !!! path is ", args.weights)
            print(e)
            return
        
        
    return extract_features(args, model,query_loader=kwargs['query_loader'],gallery_loader=kwargs['gallery_loader'])

if __name__ == "__main__":
    load_extract()
    