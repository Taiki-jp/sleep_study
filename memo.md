2021/11/03

# 実験内容

NREM2 の F 値が高いので、NREM2 のバイアスを抜いた実験を行ってみる

# 比較実験

- NREM2 バイアスありのもの
- NREM2 バイアスなしのもの

# 結果

- 若干 NREM2 バイアスなしのほうが良い

# random_seed の実験

- os.environ["PYTHONHASHSEED"] = str(seed)のコメントアウトの結果
  -0.05913474,-0.05913474
- tf.random.set_seed(seed)のコメントアウト
  -0.05913474,-0.11651658,-0.06454095
  tf.random.set_seed によってシードを固定することができる

# TODO

- 各睡眠段階の acc, pre, rec, f を全ての提案手法について計算する
  - cnn(attention)
  - cnn(no-attention)
  - ENN
  - AECNN
  - ECNN
  <!-- - CCNN -->
  - EENN
  - DENN
