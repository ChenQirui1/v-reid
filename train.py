import argparse
import time
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from datetime import datetime
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
import os.path as osp
from networks.resnet import resnet50, resnet101
# from tqdm import tqdm

# from dataset.dataset import VeriDataset, AicDataset


# from torch.utils.tensorboard import SummaryWriter

from dataset.dataset import VeriDataset

model_names = sorted(
    name
    for name in models.__dict__
    if name.islower() and not name.startswith("__") and callable(models.__dict__[name])
)

parser = argparse.ArgumentParser(description="PyTorch Relationship")

parser.add_argument("data", metavar="DIR", help="path to dataset")
parser.add_argument("trainlist", metavar="DIR", help="path to test list")
parser.add_argument(
    "--dataset",
    default="veri",
    type=str,
    help="dataset name veri or aic (default: veri)",
)
parser.add_argument(
    "-j",
    "--workers",
    default=4,
    type=int,
    metavar="N",
    help="number of data loading workers (defult: 4)",
)
parser.add_argument(
    "--batch_size",
    "--batch-size",
    default=100,
    type=int,
    metavar="N",
    help="mini-batch size (default: 1)",
)
parser.add_argument(
    "--start_epoch",
    default=0,
    type=int,
    metavar="N",
    help="mini-batch size (default: 1)",
)
parser.add_argument(
    "--backbone",
    default="resnet50",
    type=str,
    help="backbone network resnet50 or resnet101 (default: resnet50)",
)
parser.add_argument(
    "--weights",
    default="",
    type=str,
    metavar="PATH",
    help="path to weights (default: none)",
)
parser.add_argument("--scale-size", default=256, type=int, help="input size")
parser.add_argument(
    "-n",
    "--num_classes",
    default=16,
    type=int,
    metavar="N",
    help="number of classes / categories",
)
parser.add_argument(
    "--write-out", dest="write_out", action="store_true", help="write scores"
)
parser.add_argument("--crop_size", default=224, type=int, help="crop size")
parser.add_argument("--val_step", default=1, type=int, help="val step")
parser.add_argument("--epochs", default=200, type=int, help="epochs")
parser.add_argument(
    "--save_dir", default="./checkpoints/att/", type=str, help="save_dir"
)
parser.add_argument(
    "--num_gpu",
    default=4,
    type=int,
    metavar="PATH",
    help="path for saving result (default: none)",
)

best_prec1 = 0


def get_dataset(dataset_name, data_dir, train_list):

    # transformation

    # why normalise?
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    scale_size = args.scale_size
    crop_size = args.crop_size

    if dataset_name == "veri":
        train_data_transform = transforms.Compose(
            [
                transforms.Resize((336, 336)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((crop_size, crop_size)),
                transforms.ToTensor(),
                normalize,
            ]
        )
        veri_dataset = VeriDataset(
            data_dir, train_list, train_data_transform, is_train=True
        )
        train_set, val_set = torch.utils.data.random_split(veri_dataset, [0.8, 0.2])

    # elif dataset_name == "aic":
    #     train_data_transform = transforms.Compose(
    #         [
    #             transforms.Scale((336, 336)),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.RandomCrop((crop_size, crop_size)),
    #             transforms.ToTensor(),
    #             normalize,
    #         ]
    #     )
    #     train_set = AicDataset(
    #         data_dir, train_list, train_data_transform, is_train=True
    #     )
    else:
        print("!!!dataset error!!!")
        return

    training_loader = DataLoader(
        dataset=train_set,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = DataLoader(
        dataset=val_set,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=True,
    )

    return training_loader, validation_loader


def main():
    global args, best_prec1
    args = parser.parse_args()
    print(args)

    # Create dataloader
    print("====> Creating dataloader...")

    data_dir = args.data
    train_list = args.trainlist
    dataset_name = args.dataset

    training_loader, validation_loader = get_dataset(dataset_name, data_dir, train_list)

    # load network
    if args.backbone == "resnet50":
        model = resnet50(num_classes=args.num_classes)
    elif args.backbone == "resnet101":
        model = resnet101(num_classes=args.num_classes)

    if args.weights != "":
        try:
            ckpt = torch.load(args.weights)
            model.module.load_state_dict(ckpt["state_dict"])
            print("!!!load weights success !! path is ", args.weights)
        except Exception as e:
            model_init(args.weights, model)

    model = torch.nn.DataParallel(model)
    model.cuda()
    mkdir_if_missing(args.save_dir)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=10e-3
    )
    criterion = nn.CrossEntropyLoss().cuda()

    cudnn.benchmark = True

    train(training_loader, validation_loader, model, criterion, optimizer)


def model_init(weights, model):
    """
    print ('attention!!!!!!! load model fail and go on init!!!')
    ckpt = torch.load(weights)
    pretrained_dict=ckpt['state_dict']
    model_dict = model.module.state_dict()
    model_pre_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    model_dict.update(model_pre_dict)
    model.module.load_state_dict(model_dict)
    for v ,val in model_pre_dict.items() :
        print ('update',v)
    """
    saved_state_dict = torch.load(weights)
    new_params = model.state_dict().copy()
    for i in saved_state_dict:
        i_parts = i.split(".")
        # print(i_parts)
        if not i_parts[0] == "fc":
            new_params[".".join(i_parts[0:])] = saved_state_dict[i]
        else:
            print("Not Load", i)
    model.load_state_dict(new_params)

    print("-------Load Weight", weights)


def adjust_lr(optimizer, ep):
    if ep < 10:
        lr = 1e-4 * (ep + 1) / 2
    elif ep < 40:
        lr = 1e-3
    elif ep < 70:
        lr = 1e-4
    elif ep < 100:
        lr = 1e-5
    elif ep < 130:
        lr = 1e-6
    elif ep < 160:
        lr = 1e-4
    else:
        lr = 1e-5
    for p in optimizer.param_groups:
        p["lr"] = lr

    print("lr is ", lr)


def train(
    training_loader: DataLoader,
    validation_loader: DataLoader,
    model: torch.nn.Module,
    criterion,
    optimizer,
):

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # writer = SummaryWriter("runs/vreid_trainer_{}".format(timestamp))

    def train_one_epoch(epoch_index: int, tb_writer=None):

        running_loss = 0
        last_loss = 0

        # print(len(training_loader.dataset))
        for i, (image, target, camid) in enumerate(training_loader):

            batch_size = image.shape[0]
            target = target.cuda()

            # generate predictions for this batch
            outputs, res = model(image)

            # compute gradient and do SGD step
            optimizer.zero_grad()

            # print(outputs)
            # compute loss and its gradients
            loss = criterion(outputs, target)
            loss.backward()

            # adjust learning weights
            optimizer.step()

            # Gather data and report
            running_loss += loss.item()

            if i % 100 == 0:
                last_loss = running_loss / 100  # loss per batch
                print("  batch {} loss: {}".format(i + 1, last_loss))
                tb_x = epoch_index * len(training_loader) + i + 1
                # tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

        # last_loss = running_loss / (i + 1)
        return last_loss

    for epoch in range(args.start_epoch, args.epochs + 1):

        model.train(True)

        # adjust lr by epoch
        adjust_lr(optimizer, epoch)

        # train_one_epoch(epoch)
        avg_loss = train_one_epoch(epoch)

        if epoch % args.val_step == 0:
            save_checkpoint(model, epoch, optimizer)

        running_vloss = 0.0

        model.eval()
        # Disable gradient computation and reduce memory consumption.
        # statistics for batch normalization.
        print(len(validation_loader.dataset))
        with torch.no_grad():
            for vbatch_idx, vdata in enumerate(validation_loader):
                # print(len(vdata))
                vinputs, vlabels, vcamid = vdata
                vlabels = vlabels.cuda()
                voutputs, res = model(vinputs)
                vloss = criterion(voutputs, vlabels)
                # print(vloss)
                running_vloss += vloss

                # list_of_topk = accuracy(voutputs, vlabels, topk=(1, 5))

                # print("TOP1: {} TOP5: {}".format(list_of_topk[0], list_of_topk[1]))
                # print("mAP: ", )
        # print(j)
        avg_vloss = running_vloss / (vbatch_idx + 1)

        print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

        # # Log the running loss averaged per batch
        # # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss }, epoch + 1)
        # # Log the running loss averaged per batch
        # # for both training and validation
        # writer.add_scalars('Training vs. Validation Loss',
        #                 { 'Training' : avg_loss, 'Validation' : avg_vloss },
        #                 epoch + 1)
        # writer.flush()

        """
        if epoch% args.val_step == 0:
            acc = validate(test_loader, model, criterion)
            is_best = acc > best_acc
            best_acc = max(acc, best_acc)
            save_checkpoint({
                    'state_dict': model.module.state_dict(),
                    'epoch': epoch,
                }, is_best=is_best,train_batch=60000, save_dir=args.save_dir, filename='checkpoint_ep' + str(epoch) + '.pth.tar')
        """

    return


def validate(val_loader, model, criterion):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    end = time.time()
    lated = 0
    val_label = []
    val_pre = []

    model.eval()

    end = time.time()
    tp = {}  # precision
    p = {}  # prediction
    r = {}  # recall

    for i, (
        union,
        obj1,
        obj2,
        bpos,
        target,
        full_im,
        bboxes_14,
        categories,
    ) in enumerate(val_loader):

        batch_size = bboxes_14.shape[0]
        cur_rois_sum = categories[0, 0]
        bboxes = bboxes_14[0, 0 : categories[0, 0], :]
        for b in range(1, batch_size):
            bboxes = torch.cat((bboxes, bboxes_14[b, 0 : categories[b, 0], :]), 0)
            cur_rois_sum += categories[b, 0]
        assert bboxes.size(0) == cur_rois_sum, "Bboxes num must equal to categories num"

        target = target.cuda()
        union_var = torch.autograd.Variable(union, volatile=True).cuda()
        obj1_var = torch.autograd.Variable(obj1, volatile=True).cuda()
        obj2_var = torch.autograd.Variable(obj2, volatile=True).cuda()
        bpos_var = torch.autograd.Variable(bpos, volatile=True).cuda()
        full_im_var = torch.autograd.Variable(full_im, volatile=True).cuda()
        bboxes_var = torch.autograd.Variable(bboxes, volatile=True).cuda()
        categories_var = torch.autograd.Variable(categories, volatile=True).cuda()

        target_var = torch.autograd.Variable(target, volatile=True)

        output = model(
            union_var,
            obj1_var,
            obj2_var,
            bpos_var,
            full_im_var,
            bboxes_var,
            categories_var,
        )

        # compute output
        loss = criterion(output, target)

        prec1 = accuracy(output, target, topk=(1,))
        batch_size = target.size(0)
        _, pred = output.topk(1, 1, True, True)
        pred = pred.t()
        val_label[lated : lated + batch_size] = target
        val_pre[lated : lated + batch_size] = pred.data.cpu().numpy().tolist()[:]
        lated = lated + batch_size

        losses.update(loss.item(), obj1.size(0))
        top1.update(prec1[0], obj1.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 8 == 0:
            print(
                "Test: [{0}/{1}]\t"
                "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                "Prec@1 {top1.val:.3f} ({top1.avg:.3f})".format(
                    i, len(val_loader), batch_time=batch_time, loss=losses, top1=top1
                )
            )

    print("----------------------------------------------------------")
    count = [0] * 16
    acc = [0] * 16
    pre_new = []
    for i in val_pre:
        for j in i:
            pre_new.append(j)
    for idx in range(len(val_label)):
        count[val_label[idx]] += 1
        if val_label[idx] == pre_new[idx]:
            acc[val_label[idx]] += 1
    classaccuracys = []
    for i in range(16):
        if count[i] != 0:
            classaccuracy = (acc[i] * 1.0 / count[i]) * 100.0
        else:
            classaccuracy = 0
        classaccuracys.append(classaccuracy)

    print(
        (
            "Testing Results: Prec@1 {top1.avg:.3f} classacc {classaccuracys} Loss {loss.avg:.5f}".format(
                top1=top1, classaccuracys=classaccuracys, loss=losses
            )
        )
    )

    return top1.avg[0]


def save_checkpoint(model, epoch, optimizer):
    state = {
        "epoch": epoch,
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    filepath = osp.join(args.save_dir, "Car_epoch_" + str(epoch) + ".pth")
    torch.save(state, filepath)


def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


# def accuracy_original(output: torch.Tensor, target: torch.Tensor, topk=(1,)):
#     print(output.shape)
#     print(target.shape)
#     maxk = max(topk)
#     batch_size = target.size(0)

#     _, pred = output.topk(maxk, 1, True, True)
#     pred = pred.t()
#     correct = pred.eq(target.reshape(1, -1).expand_as(pred))

#     res = []
#     for k in topk:
#         correct_k = correct[:k].reshape(-1).float().sum(0)
#         res.append(correct_k.mul_(100.0 / batch_size))
#     return res


def accuracy(
    output: torch.Tensor, target: torch.Tensor, topk=(1,)
) -> list[torch.FloatTensor]:
    """
    Computes the accuracy over the k top predictions for the specified values of k
    In top-5 accuracy you give yourself credit for having the right answer
    if the right answer appears in your top five guesses.

    ref:
    - https://pytorch.org/docs/stable/generated/torch.topk.html
    - https://discuss.pytorch.org/t/imagenet-example-accuracy-calculation/7840
    - https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b
    - https://discuss.pytorch.org/t/top-k-error-calculation/48815/2
    - https://stackoverflow.com/questions/59474987/how-to-get-top-k-accuracy-in-semantic-segmentation-using-pytorch

    :param output: output is the prediction of the model e.g. scores, logits, raw y_pred before normalization or getting classes
    :param target: target is the truth
    :param topk: tuple of topk's to compute e.g. (1, 2, 5) computes top 1, top 2 and top 5.
    e.g. in top 2 it means you get a +1 if your models's top 2 predictions are in the right label.
    So if your model predicts cat, dog (0, 1) and the true label was bird (3) you get zero
    but if it were either cat or dog you'd accumulate +1 for that example.
    :return: list of topk accuracy [top1st, top2nd, ...] depending on your topk input
    """
    with torch.no_grad():
        # ---- get the topk most likely labels according to your model
        # get the largest k \in [n_classes] (i.e. the number of most likely probabilities we will use)
        maxk = max(
            topk
        )  # max number labels we will consider in the right choices for out model
        batch_size = target.size(0)

        # get top maxk indicies that correspond to the most likely probability scores
        # (note _ means we don't care about the actual top maxk scores just their corresponding indicies/labels)
        _, y_pred = output.topk(k=maxk, dim=1)  # _, [B, n_classes] -> [B, maxk]
        y_pred = (
            y_pred.t()
        )  # [B, maxk] -> [maxk, B] Expects input to be <= 2-D tensor and transposes dimensions 0 and 1.

        # - get the credit for each example if the models predictions is in maxk values (main crux of code)
        # for any example, the model will get credit if it's prediction matches the ground truth
        # for each example we compare if the model's best prediction matches the truth. If yes we get an entry of 1.
        # if the k'th top answer of the model matches the truth we get 1.
        # Note: this for any example in batch we can only ever get 1 match (so we never overestimate accuracy <1)
        target_reshaped = target.view(1, -1).expand_as(
            y_pred
        )  # [B] -> [B, 1] -> [maxk, B]
        # compare every topk's model prediction with the ground truth & give credit if any matches the ground truth
        correct = (
            y_pred == target_reshaped
        )  # [maxk, B] were for each example we know which topk prediction matched truth
        # original: correct = pred.eq(target.view(1, -1).expand_as(pred))

        # -- get topk accuracy
        list_topk_accs = []  # idx is topk1, topk2, ... etc
        for k in topk:
            # get tensor of which topk answer was right
            ind_which_topk_matched_truth = correct[:k]  # [maxk, B] -> [k, B]
            # flatten it to help compute if we got it correct for each example in batch
            flattened_indicator_which_topk_matched_truth = (
                ind_which_topk_matched_truth.reshape(-1).float()
            )  # [k, B] -> [kB]
            # get if we got it right for any of our top k prediction for each example in batch
            tot_correct_topk = flattened_indicator_which_topk_matched_truth.float().sum(
                dim=0, keepdim=True
            )  # [kB] -> [1]
            # compute topk accuracy - the accuracy of the mode's ability to get it right within it's top k guesses/preds
            topk_acc = tot_correct_topk / batch_size  # topk accuracy for entire batch
            list_topk_accs.append(topk_acc)
        return list_topk_accs  # list of topk accuracies for entire batch [topk1, topk2, ... etc]


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


if __name__ == "__main__":
    main()
