# sleep study

2020.4 - 2022.3 : code for master thethis

# Introduction

![net_alt](gallery/my_network.png "my_network")
ENN の構造は上記の通り（InceptionV3 の一部を利用）

![git_alt](gallery/out.gif "my_psedo")
仮のデータを用いた実験結果

# Features

- <strong>nn による睡眠段階推定手法</strong>

# Branch

- master : 無駄なファイルを除いた一番軽量なブランチ（基本ここから切る）
- feature/psedo_data : 仮のデータを使った実験ブランチ（マージ・削除済み）
- feature/deep_learning : EU が原因で間違っているように見えるので、多層化により一致率向上を目指す
- feature/rnn : 時系列を考慮した NN の構築
- feature/sep_learning : 何のブランチか忘れた
- feature/data_selecting : 不確かさの低いデータを拾ってきて分類する
- feature/dnn : determinstic nn の学習用ブランチ
- feature/how2show_merged_result : マージ後の結果をどのように見せるか変更するブランチ（master ベース）

# Environment

前処理後のダンプデータのファイル特定のための情報や保存したモデルの情報などは json ファイルにまとめている．以下に各種 json ファイルでの構造をまとめる．

## model_id.json

- 階層 1(model type)
  - prev_datasets(削除予定)
  - dnn
  - enn
  - unused_set(削除予定)
- 階層 2(data type; only dnn and enn)
  - spectrum
  - spectrogram
- 階層 3(sleep stage position of window)
  - bottom
  - middle
  - top
- 階層 4(stride size)
  - stride_1
  - stride_4
  - stride_16
  - stride_480
  - stride_1024
- 階層 5(kernel size)
  - kernel_512
  - kernel_1024
- 階層 6(cleansing type of model)
  - no_cleansing
  - positive_cleansing
  - negative_cleansing

## pre_processed_id.json

- 階層 1(dataset type)
  - normal_prev
  - normal_follow
  - sas_prev
  - sav_normal
- 階層 2(dataset type)
  - spectrum
  - spectrogram
  - spectrogram
- 階層 3(sleep stage position of window)
  - bottom
  - middle
  - top
- 階層 4（stride size）
  - stride_1
  - stride_4
  - stride_16
  - stride_480
  - stride_1024
- 階層 5（kernel size）
  - kernel_512
  - kernel_1024

## TODO

| DATA / MODEL | dnn  | enn  | proposed1(negative) | proposed2(aggressive) | proposed3(hierarichical) |
| ------------ | ---- | ---- | ------------------- | --------------------- | ------------------------ |
| psedo        | code | code | code                | code                  | code                     |
| sleep        | code | code | code                | code                  | code                     |

# Prefix

- feat: 新機能実装
- fix: バグの修正
- docs: ドキュメントのみの変更
- style: コード内のスタイルの変更（改行やフォーマットなどの機能以外の変更）
- refactor: 修正や新機能以外のコードの修正
- perf: パフォーマンス(実行速度)の向上
- test: 機能テストの追加
- chore: makefile,ライブラリ,その他の補足ツールの変更
  > （引用：[takadamalab](https://github.com/takadamalab)）

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
# set your local environment path of "git" and "sleep"
# example
# os.environ["git"] = "%userprofile%/git"
# os.environ["sleep"] = "%userprofile%/sleep_study"
python main.py
```

# Note

- windows ubuntu 環境で動作確認済み
- gpu の設定をしていない場合エラーが出るかも

# Author

作成情報を列挙する

- 千住太希
- 電気通信大学
- s2030060@uec.ac.jp

# License

- under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
