EntropyStop 期末專題



論文：EntropyStop: Unsupervised Deep Outlier Detection with Loss Entropy (KDD '24)



程式碼改自作者的 repo：

https://github.com/goldenNormal/EntropyStop-KDD2024

作者原本的 readme 改名為 AUTHOR\_README.md。



環境

Windows、RTX 5060 Laptop、Python 3.14、PyTorch 2.14.1 (CUDA 13.0)、PyG 2.6.1，

其他套件版本在 requirements.txt



安裝

git clone https://github.com/sunnylinlin0708/EntropyStop-final.git

cd EntropyStop-final

python -m venv .venv

.venv\\Scripts\\activate

python -m pip install -r requirements.txt

(PyG 要用 2.6.1，新版會多裝 xxhash，在我們電腦上會被擋。)



資料集

用作者放在 data/ 的 ADBench 資料集，我們從 47 個裡挑了 12 個，寫在 configs/datasets.txt。

6 個是論文裡效果正常的，6 個是論文 Table 6 說效果不好的。而5 萬筆以上的資料太大所以沒選。



執行

python run\_baseline.py

會跑 VanillaAE 和 EntropyAE，12 個資料集各跑 5 個 seed，結果存在 results/baseline/baseline\_results.csv。

