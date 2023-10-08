# sleep study

2020.4 - 2022.3 : code for master thethis
# Introduction

![net_alt](gallery/my_network.png "my_network")
ENN の構造は上記の通り（InceptionV3 の一部を利用）

![git_alt](gallery/out.gif "my_psedo")
仮のデータを用いた実験結果

# Features

- <strong>nn による睡眠段階推定手法</strong>

## TODO
| DATA / MODEL| dnn | enn | proposed1(negative) | proposed2(aggressive) | proposed3(hierarichical) |
----|----|----|----|----|----|
| psedo | code | code | code | code | code |
| sleep | code | code | code | code | code |

# wandb
## wandb のコード名
- code_CS : seed の比較を行うコード
- code_dnn : Determinstic NN の実験（比較のため）を行うコード
- code_enn : ENN の実験を行うコード

# Requirement

## windows

```bash
pip install -r requirements.txt
```

# Usage

```bash
git clone https://github.com/taiki-jp/sleep_study
cd sleep_study
python main.py
```

# Note

- windows ubuntu 環境で動作確認済み
- gpu の設定をしていない場合エラーが出るかも
