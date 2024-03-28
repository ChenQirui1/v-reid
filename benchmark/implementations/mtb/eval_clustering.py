from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, adjusted_rand_score
import shutil
import numpy as np
import os

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

def gen_clustered_data(size,pred_labels,no_of_classes, img_dir, save_dir, filepath_list_path):

    def write_clusters(file_list, group):
        mkdir_if_missing(f"{save_dir}/{group}")
        for filename in file_list:
            shutil.copy(img_dir + filename, f"{save_dir}/{group}/{filename}")

    # shutil.rmtree(save_dir)

    with open(filepath_list_path,"r") as f:
        file_content = f.read()
        img_paths = np.array(file_content.split("\n"))
    
    for i in range(no_of_classes):
        selection = np.argwhere([pred_labels == i])
        selection = selection[:,1]
        try:
            selection = np.random.choice(selection,size,replace=False)
        except:
            continue
        file_list = img_paths[selection]

        write_clusters(file_list,i)


def mkdir_if_missing(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)