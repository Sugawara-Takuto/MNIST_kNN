import tensorflow as tf
import numpy as np
import time
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from tqdm import tqdm
from PIL import Image
import numpy as np
# from sklearn import neighbors, datasets, metrics
import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

accuracies_list = []    # 正確度保存用　ｋが異なるデータ群を二次元で格納
proc_time_list = []     # 実行時間保存用　ｋが異なるデータ群を二次元で格納
    
# MNIST手書き数字データの読み込み
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.reshape(60000, 28*28)
X_test  = X_test.reshape(10000, 28*28)

# 全60,000件の学習データのうち，1%を検証用データにとっておく
trData, valData, trLabels, valLabels = train_test_split(np.array(X_train), y_train, test_size=0.01, random_state=60)

# 何%を検証用にするか
nVals = np.arange(0.9, 0.0, -0.1)

k_list = [3, 5, 7, 9, 11, 13, 15] # ｋの指定

for k in k_list:

    accuracies = []  # 正確度保存用
    proc_time = []   # 実行時間保存用

    for per in tqdm( nVals ):  # tqdmを使うとき
        
        # 59,400件のデータのうち，per％をテスト用に，残りを学習用として確保
        trainData, testData, trainLabels, testLabels = train_test_split(trData, trLabels, test_size=per, random_state=42)

        start = time.time()  # 時間計測開始

        # KNNモデル
        model =  KNeighborsClassifier(n_neighbors=k)
        model.fit(trainData, trainLabels)#学習

        # 検証用データを使って実行
        score = model.score(valData, valLabels)    

        duration = time.time() - start  # 計測終了

        print("train size=%d, accuracy=%.2f%%, time=%.2f[s]" % (len(trainLabels), score * 100, duration))
        # 正確度と処理時間を保存
        accuracies.append(score)
        proc_time.append(duration)
    # 正確度と処理時間を保存（ｋ別に）
    accuracies_list.append(accuracies)
    proc_time_list.append(proc_time)

# 最大識別率となる学習データ個数は？
i = np.argmax(accuracies)
print("train size=%5d achieved highest accuracy of %.2f%% on validation data" % (len(trLabels) * (1 - nVals[i]), accuracies[i] * 100))

# グラフ

#figure()でグラフを表示する領域をつくり，figというオブジェクトにする．
fig = plt.figure()

#add_subplot()でグラフを描画する領域を追加する．引数は行，列，場所
ax1 = fig.add_subplot(3, 3, 1)
ax2 = fig.add_subplot(3, 3, 2)
ax3 = fig.add_subplot(3, 3, 3)
ax4 = fig.add_subplot(3, 3, 4)
ax5 = fig.add_subplot(3, 3, 5)
ax6 = fig.add_subplot(3, 3, 6)
ax7 = fig.add_subplot(3, 3, 7)

# グラフのパラメータ設定
x = len(trLabels)*(1-nVals)

y1 = np.array(accuracies_list[0])
y2 = np.array(accuracies_list[1])
y3 = np.array(accuracies_list[2])
y4 = np.array(accuracies_list[3])
y5 = np.array(accuracies_list[4])
y6 = np.array(accuracies_list[5])
y7 = np.array(accuracies_list[6])

y2_1 = np.array(proc_time_list[0])
y2_2 = np.array(proc_time_list[1])
y2_3 = np.array(proc_time_list[2])
y2_4 = np.array(proc_time_list[3])
y2_5 = np.array(proc_time_list[4])
y2_6 = np.array(proc_time_list[5])
y2_7 = np.array(proc_time_list[6])

c1,c2 = "blue","green"     # 各プロットの色

# グラフにプロットする関数
def makeplot (ax, y, y2_, k_num):
    ax.plot(x, y, color=c1, label= 'accuracies')
    # 1軸と2軸の関連づけ
    ax.set_ylim([0.9, 1])
    plt.legend()
    ax_2 = ax.twinx()
    # 2軸に処理時間
    ax_2.plot(x,y2_, color=c2)
    plt.ylim([0, 1.3])
    plt.title("K = " + k_num)
    return ax_2

# 関数の実行
makeplot(ax1, y1, y2_1, '3')
makeplot(ax2, y2, y2_2, '5')
makeplot(ax3, y3, y2_3, '7')
makeplot(ax4, y4, y2_4, '9')
makeplot(ax5, y5, y2_5, '11')
makeplot(ax6, y6, y2_6, '13')
makeplot(ax7, y7, y2_7, '15')

plt.show()

#False画像取り出し
output = model.predict(valData)#結果の呼び出し
data_len = (len(valLabels))
a=1
for i in range(data_len):
    if output[i] != valLabels[i]:
        f_image = valData[i]
        ax = plt.subplot(1, 10, a)
        ax.set_axis_off()
        ax.set_title( "class: {}:{}".format(valLabels[i],output[i]))#ラベル：予測
        plt.imshow(f_image.reshape(28, 28).astype("uint8"))
        a += 1
        if a > 10:
            break

plt.show()