import tensorflow as tf
tf.random.set_seed(0)
import numpy as np
import matplotlib.pyplot as plt
from nn.model_base import edl_classifier4psedo_data, EDLModelBase
from nn.losses import EDLLoss
import os, sys
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# 仮データの作成
def psedo_data(row, col, x_bias, y_bias):
    # 極座標で考える
    row, col = (100, 2)
    r_class0 = tf.random.uniform(shape=(row,), minval=0, maxval=0.5)
    theta_class0 = tf.random.uniform(shape=(row,), minval=0, maxval=np.pi*2)
    r_class1 = tf.random.uniform(shape=(row,), minval=0.5, maxval=1)
    theta_class1 = tf.random.uniform(shape=(row,), minval=0, maxval=np.pi*2)
    input_class0 = (x_bias+r_class0*np.cos(theta_class0), y_bias+r_class0*np.sin(theta_class0))
    input_class1 = (x_bias+r_class1*np.cos(theta_class1), y_bias+r_class1*np.sin(theta_class1))
    x_train = tf.concat([input_class0, input_class1], axis=1)
    x_train = tf.transpose(x_train)
    y_train_0 = [0 for _ in range(row)]
    y_train_1 = [1 for _ in range(row)]
    y_train = y_train_0 + y_train_1
    x_test = None
    y_test = None
    return (x_train, x_test), (y_train, y_test)


(x_train, x_test), (y_train, y_test) = psedo_data(row=100, col=2, x_bias=0, y_bias=0)

# カスタムトレーニング
inputs = tf.keras.Input(shape=(2,))
outputs = edl_classifier4psedo_data(x=inputs, use_bias=True, hidden_dim=8)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
# 最適化関数の設定
optimizer = tf.keras.optimizers.Adam()
# メトリクスの作成
# true side : カテゴリカルな状態，pred side : クラスの次元数（ソフトマックスをかける前)
# CategoricalAccuracy : one-hotに対して計算してくれる
train_acc_metric = tf.keras.metrics.CategoricalAccuracy()
epochs = 200
time_hparam = 1
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(buffer_size=x_train.shape[0]).batch(batch_size=32)

for epoch in range(epochs):
    loss_fn = EDLLoss(K=2, annealing=min(1.0, epoch/epochs*time_hparam))
    print(f"エポック:{epoch}")
    # エポック内のバッチサイズごとのループ
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        # 勾配を計算
        with tf.GradientTape() as tape:
            # NOTE : x_batch_trainの次元は3である
            #assert np.ndim(x_batch_train) == 3
            # dataset.shuffleを入れることによってバッチサイズを設定できる
            evidence = model(x_batch_train, training=True)
            alpha = evidence+1
            y_pred = alpha/tf.reduce_sum(alpha, axis=1, keepdims=True)
            # NOTE : ここでone-hotにする
            y_batch_train = tf.one_hot(y_batch_train, depth=2)
            loss_value = loss_fn.call(y_batch_train, alpha)  # NOTE : one-hotの形で入れるべき？
            
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # NOTE : categoricalに戻す <= いる？
        # y_batch_train = tf.keras.utils.to_categorical(y_batch_train,
        #                                               num_classes=2,)
        # y_pred = tf.keras.utils.to_categorical(y_pred,
        #                                        num_classes=2,)
        train_acc_metric.update_state(y_batch_train, y_pred)
    
    # エポックの終わりにメトリクスを表示する
    train_acc = train_acc_metric.result()
    print(f"訓練一致率：{train_acc:.2%}")
    # エポックの終わりに訓練メトリクスを初期化
    train_acc_metric.reset_states()

# 結果の出力(訓練データそのまま)
evidence = model(x_train)
alpha = evidence + 1
y_pred = alpha/tf.reduce_sum(alpha, axis=1, keepdims=True)
unc = 2/tf.reduce_sum(alpha, axis=1, keepdims=True)
# 2次元空間での不確かさの出力
# グラデーションを作る
# 正解の散布図
figure = plt.figure(figsize=(12,4))
ax = figure.add_subplot(131)
ax.scatter(x_train[:100,0], x_train[:100,1], c="r")
ax.scatter(x_train[100:,0], x_train[100:,1], c="b")
ax.set_title("true")
# 予測の散布図
ax = figure.add_subplot(132)
ax.set_title("pred")
y_pred_ctg = np.argmax(y_pred, axis=1)
for x, label in zip(x_train, y_pred_ctg):
    if label == 0:
        ax.scatter(x[0], x[1], c="r")
    elif label == 1:
        ax.scatter(x[0], x[1], c="b")
    else:
        print("exception has occured")
        sys.exit(1)
# 不確かさの分布
ax = figure.add_subplot(133)
ax.set_title("unc")
im = ax.scatter(x_train[:,0], x_train[:,1], c=unc, cmap='Blues')
cbar = figure.colorbar(im)
plt.legend()
plt.show()