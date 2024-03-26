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
from networks.resnet import resnet50, resnet101
from dataset.dataset import VeriDataset
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from eval_reid import compute_distmat, eval_func, extract_features
from plot import plot_cluster, plot_cmc

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


def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def eval_loop(func):
    def wrapper():
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
        parser.add_argument(
            "--save_dir", default="./results/", type=str, help="save_dir"
        )
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

                query_set = VeriDataset(
                    query_dir, query_list, data_transform, is_train=False
                )
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

        # Pass loaders to func
        kwargs = {"query_loader": query_loader, "gallery_loader": gallery_loader}

        # query_loader, gallery_loader = fake_dataset(args.crop_size)
        print("Cuda available: ", torch.cuda.is_available())

        mkdir_if_missing(args.save_dir)

        cudnn.benchmark = True

        print(args.weights)

        if args.nocalc:
            query_feats = np.load(args.save_dir + "queryFeat.npy")
            query_pids = np.load(args.save_dir + "queryPID.npy")
            query_camids = np.load(args.save_dir + "queryCamID.npy")
            gallery_feats = np.load(args.save_dir + "galleryFeat.npy")
            gallery_pids = np.load(args.save_dir + "galleryPID.npy")
            gallery_camids = np.load(args.save_dir + "galleryCamID.npy")

            distmat = np.load(args.save_dir + "distmat.npy")
        else:
            # load model and extract features
            result = func(args, **kwargs)

            (
                query_feats,
                gallery_feats,
            ) = result

            STANDARD_PID_CAMID = "./results/standard" 

            query_pids = np.load( STANDARD_PID_CAMID + "queryPID.npy")
            query_camids = np.load( STANDARD_PID_CAMID + "queryCamID.npy")

            gallery_pids = np.load(STANDARD_PID_CAMID + "galleryPID.npy")
            gallery_camids = np.load(STANDARD_PID_CAMID + "galleryCamID.npy")

            np.save(args.save_dir + "queryFeat.npy", query_feats)
            np.save(args.save_dir + "queryPID.npy", query_pids)
            np.save(args.save_dir + "queryCamID.npy", query_camids)

            np.save(args.save_dir + "galleryFeat.npy", gallery_feats)
            np.save(args.save_dir + "galleryPID.npy", gallery_pids)
            np.save(args.save_dir + "galleryCamID.npy", gallery_camids)

            # compute distmat
            distmat = compute_distmat(query_feats, gallery_feats)

            np.save(args.save_dir + "distmat.npy", np.asarray(distmat))
            np.savetxt(args.save_dir + "distmat.txt", np.asarray(distmat), fmt="%.4f")

        print("Computing CMC and mAP...")
        # compute cmc and mAP
        cmc, mAP = eval_func(
            args, distmat, query_pids, gallery_pids, query_camids, gallery_camids
        )
        print(
            "mAP = " + "%.4f" % mAP + "\tRank-1 = " + "%.4f" % cmc[0],
            "\tRank-5 = " + "%.4f" % cmc[4],
        )

        # plotting
        plot_cmc(cmc)

        pred_clusters = eval_cluster(gallery_feats, gallery_pids)

        plot_cluster(gallery_feats, gallery_pids, pred_clusters)

        return result

    return wrapper


# def eval_loop(func):
#     print('test')
#     def wrapper(**kwargs):
#         global args
#         parser = argparse.ArgumentParser(description="PyTorch Relationship")

#         parser.add_argument("querypath", metavar="DIR", help="path to query set")
#         parser.add_argument("querylist", metavar="DIR", help="path to query list")
#         parser.add_argument("gallerypath", metavar="DIR", help="path to gallery set")
#         parser.add_argument("gallerylist", metavar="DIR", help="path to gallery list")
#         parser.add_argument(
#             "--dataset", default="veri", type=str, help="dataset name (default: veri)"
#         )
#         parser.add_argument(
#             "-j",
#             "--workers",
#             default=4,
#             type=int,
#             metavar="N",
#             help="number of data loading workers (defult: 4)",
#         )
#         parser.add_argument(
#             "--batch_size",
#             "--batch-size",
#             default=1,
#             type=int,
#             metavar="N",
#             help="mini-batch size (default: 1)",
#         )
#         parser.add_argument(
#             "-n",
#             "--num_classes",
#             default=576,
#             type=int,
#             metavar="N",
#             help="number of classes / categories",
#         )
#         parser.add_argument(
#             "--backbone",
#             default="resnet50",
#             type=str,
#             help="backbone network resnet50 or resnet101 (default: resnet50)",
#         )
#         parser.add_argument(
#             "--weights",
#             default="",
#             type=str,
#             metavar="PATH",
#             help="path to weights (default: none)",
#         )
#         parser.add_argument("--scale-size", default=224, type=int, help="input size")
#         parser.add_argument("--crop_size", default=224, type=int, help="crop size")
#         parser.add_argument("--save_dir", default="./results/", type=str, help="save_dir")
#         parser.add_argument(
#             "--TopK",
#             default=100,
#             type=int,
#             help="save top K indexes of results for each query (default: 100)",
#         )
#         parser.add_argument(
#             "--nocalc",
#             default=False,
#             help="no calculation of features and distance matrix",
#             action=argparse.BooleanOptionalAction,
#         )


#         torch.multiprocessing.set_sharing_strategy("file_system")

#         args = parser.parse_args()
#         print(args)

#         # Create dataloader
#         print("====> Creating dataloader...")

#         query_dir = args.querypath
#         query_list = args.querylist
#         gallery_dir = args.gallerypath
#         gallery_list = args.gallerylist
#         dataset_name = args.dataset

#         query_loader, gallery_loader = get_dataset(
#             dataset_name, query_dir, query_list, gallery_dir, gallery_list
#         )

#         kwargs['query_loader'] = query_loader
#         kwargs['gallery_loader'] = gallery_loader

#         # query_loader, gallery_loader = fake_dataset(args.crop_size)
#         print("Cuda available: ", torch.cuda.is_available())

#         mkdir_if_missing(args.save_dir)

#         cudnn.benchmark = True

#         print(args.weights)


#         if args.nocalc:
#             query_feats = np.load(args.save_dir + "queryFeat.npy")
#             query_pids = np.load(args.save_dir + "queryPID.npy")
#             query_camids = np.load(args.save_dir + "queryCamID.npy")
#             gallery_feats = np.load(args.save_dir + "galleryFeat.npy")
#             gallery_pids = np.load(args.save_dir + "galleryPID.npy")
#             gallery_camids = np.load(args.save_dir + "galleryCamID.npy")

#             distmat = np.load(args.save_dir + "distmat.npy")
#         else:

#             # load model and extract features
#             (
#                 query_feats,
#                 query_pids,
#                 query_camids,
#                 gallery_feats,
#                 gallery_pids,
#                 gallery_camids,
#             ) = func(*args,**kwargs)

#             # compute distmat
#             distmat = compute_distmat(query_feats, gallery_feats)

#         print("Computing CMC and mAP...")
#         # compute cmc and mAP
#         cmc, mAP = eval_func(
#             distmat, query_pids, gallery_pids, query_camids, gallery_camids
#         )
#         print(
#             "mAP = " + "%.4f" % mAP + "\tRank-1 = " + "%.4f" % cmc[0],
#             "\tRank-5 = " + "%.4f" % cmc[4],
#         )

#         #plotting
#         plot_cmc(cmc)

#         pred_clusters = eval_cluster(gallery_feats, gallery_pids)

#         plot_cluster(gallery_feats, gallery_pids, pred_clusters)

#     return wrapper


if __name__ == "__main__":

    @eval_loop
    def load_extract():
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
