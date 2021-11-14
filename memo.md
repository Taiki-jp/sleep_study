2021/11/03
# 実験内容
NREM2のF値が高いので、NREM2のバイアスを抜いた実験を行ってみる
# 比較実験
- NREM2バイアスありのもの
- NREM2バイアスなしのもの
# 結果
- 若干NREM2バイアスなしのほうが良い

# random_seedの実験

- os.environ["PYTHONHASHSEED"] = str(seed)のコメントアウトの結果
 -0.05913474,-0.05913474
- tf.random.set_seed(seed)のコメントアウト
 -0.05913474,-0.11651658,-0.06454095
 tf.random.set_seedによってシードを固定することができる
