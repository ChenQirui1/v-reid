import numpy as np
import matplotlib.pyplot as plt


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