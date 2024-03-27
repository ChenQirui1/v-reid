"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import numpy as np
import torch
from tqdm import tqdm
import time
import shutil

def extract_features(args, model, query_loader, gallery_loader):
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
    # max_rank = args.TopK
    if num_g < max_rank:
        max_rank = num_g
        print("Note: number of gallery samples is quite small, got {}".format(num_g))

    # sorted index of distmat in accending for ranking
    indices = np.argsort(distmat, axis=1)


    # check for every query, which gallery labels match the query label
    # returns a binary matrix of shape: (no. of queries, no. of gallery)
    # g_pids[indices] broadcasts the gallery labels by no, of queries
    # matches = (g_pids[indices] == q_pids[:, np.newaxis]).astype(np.int32)
    print(g_pids)
    matches = np.array(g_pids[indices] == q_pids[:, np.newaxis])

    print("Saving resulting indexes...", indices.shape)
    # np.save(args.save_dir + "result.npy", indices[:, : args.TopK] + 1)
    # np.savetxt(args.save_dir + "result.txt", indices[:, : args.TopK] + 1, fmt="%d")

    # np.save(args.save_dir + "result.npy", indices[:, : max_rank] + 1)
    # np.savetxt(args.save_dir + "result.txt", indices[:, : max_rank] + 1, fmt="%d")



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


