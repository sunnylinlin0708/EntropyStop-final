
import numpy as np
from sklearn.metrics import silhouette_score


def cluster_losses(losses):
    """將一維 loss 分成兩群。"""

    x = np.asarray(losses, dtype=np.float64).reshape(-1)

    if not np.isfinite(x).all():
        raise ValueError("Loss contains NaN or Inf")

    n = len(x)

    if n < 3 or np.unique(x).size < 2:
        return x, None

    # 排序
    order = np.argsort(x)
    z = x[order]

    # 中心化，降低數值誤差
    z = z - z.mean()

    # 計算累積和
    prefix = np.cumsum(z)
    prefix_sq = np.cumsum(z ** 2)

    sizes = np.arange(1, n)

    # 左群 WSS
    left_sum = prefix[:-1]
    left_sq = prefix_sq[:-1]

    left_wss = (
        left_sq - left_sum ** 2 / sizes
    )

    # 右群 WSS
    right_sum = prefix[-1] - left_sum
    right_sq = prefix_sq[-1] - left_sq

    right_wss = (
        right_sq - right_sum ** 2 / (n - sizes)
    )

    # 找出使總 WSS 最小的切點
    wss = left_wss + right_wss

    # 不切開相同的 loss
    wss[z[:-1] == z[1:]] = np.inf

    split = int(np.argmin(wss)) + 1

    labels = np.zeros(n, dtype=int)
    labels[order[split:]] = 1

    return x, labels


def cal_bss(losses):
    """計算 BSS / TSS，越大越好。"""

    x, labels = cluster_losses(losses)

    if labels is None:
        return 0.0

    tss = np.sum((x - x.mean()) ** 2)

    if tss <= 0:
        return 0.0

    wss = 0.0

    for c in range(2):
        group = x[labels == c]
        wss += np.sum((group - group.mean()) ** 2)

    bss = tss - wss

    return float(np.clip(bss / tss, 0, 1))


def cal_silhouette(losses):
    """計算平均 Silhouette，越大越好。"""

    x, labels = cluster_losses(losses)

    if labels is None:
        return -1.0

    return float(
        silhouette_score(
            x.reshape(-1, 1),
            labels,
            metric="euclidean"
        )
    )


# B 的停止框架是越小越好，因此取負號
def bss_metric(losses):
    return -cal_bss(losses)


def silhouette_metric(losses):
    return -cal_silhouette(losses)


if __name__ == "__main__":
    losses = np.array([1, 2, 3, 8, 9, 10])

    print("BSS/TSS:", cal_bss(losses))
    print("Silhouette:", cal_silhouette(losses))

    print("BSS stop metric:", bss_metric(losses))
    print("Silhouette stop metric:", silhouette_metric(losses))
