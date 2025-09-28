import matplotlib.pyplot as plt

def plot_top_importances(imp, k=10, title="Top permutation importances"):
    topk = imp.head(k)
    plt.bar([str(n) for n in topk.index], topk.values)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel("delta MAE from permutation")
    plt.title(title)
    plt.tight_layout()
