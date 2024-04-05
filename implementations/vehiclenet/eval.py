import torch
from tqdm import tqdm
import argparse
import os
import time
import numpy as np
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import matplotlib.pyplot as plt
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.datasets import FakeData
from torch.utils.data import DataLoader
from tqdm import tqdm
from networks.resnet import resnet49, resnet101
from dataset.dataset import VeriDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
######################################################################
# Extract feature
# ----------------------
#
# Extract feature from  a trained model.


def fliplr(img):
    '''flip horizontal'''
    inv_idx = torch.arange(img.size(3)-1,-1,-1).long()  # N x C x H x W
    img_flip = img.index_select(3,inv_idx)
    return img_flip

def extract_feature(model,dataloaders, opt):
    #features = torch.FloatTensor()
    # count = 0
    pbar = tqdm()
    if opt.linear_num <= 0:
        if opt.use_swin or opt.use_swinv2 or opt.use_dense or opt.use_convnext:
            opt.linear_num = 1024
        elif opt.use_efficient:
            opt.linear_num = 1792
        elif opt.use_NAS:
            opt.linear_num = 4032
        else:
            opt.linear_num = 2048

    for iter, data in enumerate(dataloaders):
        # img, label = data
        img, label, _ = data
        n, c, h, w = img.size()
        # count += n
        # print(count)
        pbar.update(n)
        ff = torch.FloatTensor(n,opt.linear_num).zero_().cuda()

        if opt.PCB:
            ff = torch.FloatTensor(n,2048,6).zero_().cuda() # we have six parts

        for i in range(2):
            if(i==1):
                img = fliplr(img)
            input_img = Variable(img.cuda())
            for scale in ms:
                if scale != 1:
                    # bicubic is only  available in pytorch>= 1.1
                    input_img = nn.functional.interpolate(input_img, scale_factor=scale, mode='bicubic', align_corners=False)
                outputs = model(input_img) 
                ff += outputs
        # norm feature
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



parser = argparse.ArgumentParser(description="PyTorch Relationship")

parser.add_argument("querypath", metavar="DIR", help="path to query set")
parser.add_argument("querylist", metavar="DIR", help="path to query list")
parser.add_argument("gallerypath", metavar="DIR", help="path to gallery set")
parser.add_argument("gallerylist", metavar="DIR", help="path to gallery list")
parser.add_argument(
    "--dataset", default="veri", type=str, help="dataset name (default: veri)"
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
    default=1,
    type=int,
    metavar="N",
    help="mini-batch size (default: 1)",
)
parser.add_argument(
    "-n",
    "--num_classes",
    default=576,
    type=int,
    metavar="N",
    help="number of classes / categories",
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
parser.add_argument("--scale-size", default=224, type=int, help="input size")
parser.add_argument("--crop_size", default=224, type=int, help="crop size")
parser.add_argument("--save_dir", default="./results/", type=str, help="save_dir")
parser.add_argument(
    "--TopK",
    default=100,
    type=int,
    help="save top K indexes of results for each query (default: 100)",
)
parser.add_argument(
    "--nocalc",
    default=False,
    help="no calculation of features and distance matrix",
    action=argparse.BooleanOptionalAction,
)


def get_dataset(dataset_name, query_dir, query_list, gallery_dir, gallery_list):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    scale_size = args.scale_size
    crop_size = args.crop_size

    if dataset_name == "veri":
        data_transform = transforms.Compose(
            [
                transforms.Resize((scale_size, scale_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop((crop_size, crop_size)),
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
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=False,
    )
    gallery_loader = DataLoader(
        dataset=gallery_set,
        num_workers=args.workers,
        batch_size=args.batch_size,
        shuffle=False,
    )

    return query_loader, gallery_loader


def fake_dataset(image_size):
    query_set = FakeData(size=1000, image_size=(3, image_size, image_size))
    gallery_set = FakeData(size=1000, image_size=(3, image_size, image_size))
    query_loader = DataLoader(query_set)
    gallery_loader = DataLoader(gallery_set)

    return query_loader, gallery_loader


def main():
    global args

    torch.multiprocessing.set_sharing_strategy("file_system")

    args = parser.parse_args()
    print(args)
    # Create dataloader
    print("====> Creating dataloader...")

    query_dir = args.querypath
    query_list = args.querylist
    gallery_dir = args.gallerypath
    gallery_list = args.gallerylist
    dataset_name = args.dataset

    query_loader, gallery_loader = get_dataset(
        dataset_name, query_dir, query_list, gallery_dir, gallery_list
    )

    # ! swap out this implemention in real runs
    # query_loader, gallery_loader = fake_dataset(args.crop_size)

    print("Cuda available: ", torch.cuda.is_available())

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
            # print(ckpt["state_dict"])
            model.load_state_dict(ckpt["state_dict"])
            print("!!!load weights success !!! path is ", args.weights)
        except Exception as e:
            print("!!!load weights failed !!! path is ", args.weights)
            print(e)
            return
    # else:
    #     print("!!!Load Weights PATH ERROR!!!")
    #     return
    model.cuda()
    mkdir_if_missing(args.save_dir)

    cudnn.benchmark = True
    evaluate(query_loader, gallery_loader, model)

    return


######################################################################
# Load model
#---------------------------
def load_network(network):
    save_path = os.path.join('./model',name,'net_%s.pth'%opt.which_epoch)
    try:
        network.load_state_dict(torch.load(save_path))
    except: 
        if torch.cuda.get_device_capability()[0]>6 and len(opt.gpu_ids)==1 and int(version[0])>1: # should be >=7
            print("Compiling model...")
            # https://huggingface.co/docs/diffusers/main/en/optimization/torch2.0
            torch.set_float32_matmul_precision('high')
            network = torch.compile(network, mode="default", dynamic=True) # pytorch 2.0
        network.load_state_dict(torch.load(save_path))

    return network

def evaluate(query_loader, gallery_loader, model):

    if args.nocalc:

        query_feats = np.load(args.save_dir + "queryFeat.npy")
        query_pids = np.load(args.save_dir + "queryPID.npy")
        query_camids = np.load(args.save_dir + "queryCamID.npy")
        gallery_feats = np.load(args.save_dir + "galleryFeat.npy")
        gallery_pids = np.load(args.save_dir + "galleryPID.npy")
        gallery_camids = np.load(args.save_dir + "galleryCamID.npy")

        distmat = np.load(args.save_dir + "distmat.npy")
    else:
        # extract features
        (
            query_feats,
            query_pids,
            query_camids,
            gallery_feats,
            gallery_pids,
            gallery_camids,
        ) = extract_features(model, query_loader, gallery_loader)
        # compute distmat
        distmat = compute_distmat(query_feats, gallery_feats)

    print("Computing CMC and mAP...")
    # compute cmc and mAP
    cmc, mAP = eval_func(
        distmat, query_pids, gallery_pids, query_camids, gallery_camids
    )
    print(
        "mAP = " + "%.4f" % mAP + "\tRank-1 = " + "%.4f" % cmc[0],
        "\tRank-5 = " + "%.4f" % cmc[4],
    )

    plot_cmc(cmc)

    pred_clusters = eval_cluster(gallery_feats, gallery_pids)

    plot_cluster(gallery_feats, gallery_pids, pred_clusters)

    return


def extract_features(model, query_loader, gallery_loader):
    print("Start evaluation...")
    query_feats = []
    query_pids = []
    query_camids = []

    gallery_feats = []
    gallery_pids = []
    gallery_camids = []

    end = time.time()
    # switch to eval mode
    model.eval()

    print("Processing query set...")
    queryN = 0
    for image, pid, camid in tqdm(query_loader):
        # if i == 10:
        #     break
        # print("Extracting feature of image " + "%d:" % i)
        query_pids.append(pid)
        query_camids.append(camid)

        image = image.cuda()
        # image = torch.autograd.Variable(image).cuda()
        output, feat = model(image)

        query_feats.append(feat.data.cpu())
        queryN = queryN + 1

    query_time = time.time() - end
    end = time.time()
    print("Processing query set... \tTime[{0:.3f}]".format(query_time))

    print("Processing gallery set...")
    galleryN = 0
    for image, pid, camid in tqdm(gallery_loader):
        # if i == 20:
        #     break
        # print("Extracting feature of image " + "%d:" % i)
        gallery_pids.append(pid)
        gallery_camids.append(camid)
        image = torch.autograd.Variable(image).cuda()
        output, feat = model(image)
        gallery_feats.append(feat.data.cpu())
        galleryN = galleryN + 1

    gallery_time = time.time() - end
    print("Processing gallery set... \tTime[{0:.3f}]".format(gallery_time))

    # query
    qf = torch.cat(query_feats, dim=0)

    q_pids = np.asarray(query_pids)
    q_camids = np.asarray(query_camids).T

    # gallery
    gf = torch.cat(gallery_feats, dim=0)
    g_pids = np.asarray(gallery_pids)
    g_camids = np.asarray(gallery_camids).T

    m, n = qf.shape[0], gf.shape[0]
    qf = qf.view(m, -1)
    gf = gf.view(n, -1)

    q_camids = np.squeeze(q_camids)
    g_camids = np.squeeze(g_camids)

    print("Saving feature mat...")

    np.save(args.save_dir + "queryFeat.npy", qf)
    np.save(args.save_dir + "queryPID.npy", q_pids)
    np.save(args.save_dir + "queryCamID.npy", q_camids)

    np.save(args.save_dir + "galleryFeat.npy", gf)
    np.save(args.save_dir + "galleryPID.npy", g_pids)
    np.save(args.save_dir + "galleryCamID.npy", g_camids)

    return qf, q_pids, q_camids, gf, g_pids, g_camids


def compute_distmat(query_feats, gallery_feats):

    # pairwise l2 distance between query vector and gallery
    # similar to https://scikit-learn.org/0.16/modules/generated/sklearn.metrics.pairwise.euclidean_distances.html
    # but this calculates for a batch
    # can use cdist from pytorch also

    # distmat = (
    #     torch.pow(qf, 2).sum(dim=1, keepdim=True).expand(m, n)
    #     + torch.pow(gf, 2).sum(dim=1, keepdim=True).expand(n, m).t()
    # )
    # distmat.addmm_(1, -2, qf, gf.t())
    # distmat = distmat.cpu().numpy()

    distmat = torch.cdist(query_feats, gallery_feats, p=2)
    distmat = distmat.cpu().numpy()

    print("Saving distmat...")
    np.save(args.save_dir + "distmat.npy", np.asarray(distmat))
    np.savetxt(args.save_dir + "distmat.txt", np.asarray(distmat), fmt="%.4f")

    return distmat


def eval_func(
    distmat: np.ndarray,
    q_pids: np.ndarray,
    g_pids: np.ndarray,
    q_camids: np.ndarray,
    g_camids: np.ndarray,
    max_rank=100,
):
    """Evaluation with market1501 metric
    Key: for each query identity, its gallery images from the same camera view are discarded.
    """
    num_q, num_g = distmat.shape
    max_rank = args.TopK
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    # sorted index of distmat in accending for ranking
    indices = np.argsort(distmat, axis=1)

    # check for every query, which gallery labels match the query label
    # returns a binary matrix of shape: (no. of queries, no. of gallery)
    # g_pids[indices] broadcasts the gallery labels by no, of queries
    # matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    matches = np.array(g_pids[indices] == q_pids[:, np.newaxis])

    print("Saving resulting indexes...", indices.shape)
    np.save(args.save_dir + "result.npy", indices[:, : args.TopK] + 1)
    np.savetxt(args.save_dir + "result.txt", indices[:, : args.TopK] + 1, fmt="%d")

    all_cmc = []
    # print(type(all_cmc))
    all_AP = []
    num_valid_q = 0.0  # number of valid query
    for q_idx in tqdm(range(num_q)):
        # get query pid and camid
        q_pid = q_pids[q_idx]
        q_camid = q_camids[q_idx]

        # remove gallery samples that have the same pid and camid with query
        order = indices[q_idx]
        remove = np.squeeze(g_pids[order] == q_pid) & np.squeeze(
            g_camids[order] == q_camid
        )
        keep = np.invert(remove)

        # compute cmc curve
        # binary vector, positions with value 1 are correct matches
        rectified_matches = matches[q_idx][keep]
        if not np.any(rectified_matches):
            # this condition is true when query identity does not appear in gallery
            continue

        # the cmc will be [0,0,...1,1,...1]
        # denoting the rank at which the match is found
        cmc = rectified_matches.cumsum()
        # creating a step function, https://cysu.github.io/open-reid/notes/evaluation_metrics.html
        cmc[cmc > 1] = 1

        # print(type(all_cmc))
        # print(type(cmc[:max_rank].tolist()))
        all_cmc.append(cmc[:max_rank])
        num_valid_q += 1.0

        # compute average precision
        # reference: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Average_precision

        # num of relevant docs
        num_rel = rectified_matches.sum()
        # print(num_rel)

        cum_matches = rectified_matches.cumsum()
        # calculate the precision @ k for all k ranks
        # precision_at_k = [x / (i + 1.0) for i, x in enumerate(cum_matches)]
        # print(precision_at_k)
        precision_at_k = cum_matches / (np.arange(len(cum_matches)) + 1.0)

        # print("precision at k", precision_at_k)
        # print("rectified_matches", rectified_matches)
        # print("num_rel", num_rel)
        # p @ k * relevance @ k
        tmp_cmc = precision_at_k * rectified_matches.ravel()
        # print(tmp_cmc)
        AP = tmp_cmc.sum() / num_rel
        # print("AP", AP)
        all_AP.append(AP)

        assert num_valid_q > 0, "Error: all query identities do not appear in gallery"

    all_cmc = np.asarray(all_cmc).astype(np.float32)
    all_cmc = all_cmc.sum(axis=0) / num_valid_q
    mAP = np.mean(all_AP)

    return all_cmc, mAP


def eval_cluster(gallery_feat, true_labels):

    transformed_feat = PCA(n_components=2).fit_transform(gallery_feat)

    kmeans = KMeans(n_clusters=576, random_state=0).fit(transformed_feat)

    predict = kmeans.fit_predict(transformed_feat)

    silhouette_avg = silhouette_score(transformed_feat, predict)

    adjusted_rand = adjusted_rand_score(
        labels_true=true_labels.ravel(), labels_pred=predict
    )

    print("Silhouette Score: ", silhouette_avg)
    print("Adjusted Rand Score: ", adjusted_rand)

    return predict


def plot_cmc(cmc_values: np.ndarray, topk=20):
    # print(all_cmc.shape)
    # Plot CMC curve for the query
    # cmc_values = np.mean(all_cmc, axis=0)
    # print(cmc_values)
    cmc_values = cmc_values[:topk]

    # Plot CMC Curve
    plt.figure(figsize=(8, 5))
    plt.plot(
        range(1, len(cmc_values) + 1),
        cmc_values,
        marker="o",
        fillstyle="none",
        label="CMC Curve",
    )
    plt.xticks(range(1, len(cmc_values) + 1))
    plt.xlabel("Rank k")
    plt.ylabel("Recognition Percentage %")
    plt.title("CMC Curve")
    plt.legend()
    plt.grid(True)
    # plt.show()
    plt.savefig(args.save_dir + "cmc_curve.png", format="png")


def plot_cluster(reduced_feat, true_labels, pred_labels):

    # Plot the reduced features
    plt.figure(figsize=(8, 5))
    plt.scatter(reduced_feat[:, 0], reduced_feat[:, 1], c=true_labels, cmap="viridis")
    plt.title("True Labels")
    plt.colorbar()
    plt.savefig(args.save_dir + "true_labels.png", format="png")

    plt.figure(figsize=(8, 5))
    plt.scatter(reduced_feat[:, 0], reduced_feat[:, 1], c=pred_labels, cmap="viridis")
    plt.title("Predicted Labels")
    plt.colorbar()
    plt.savefig(args.save_dir + "pred_labels.png", format="png")

    return


def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


if __name__ == "__main__":
    main()