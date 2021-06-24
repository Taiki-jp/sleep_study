# sleep study

2020.4 - 2022.3 までの睡眠の研究まとめ
 
# DEMO
 ![my_alt](gallery/my_network.png "my_title")
ENNの構造は上記の通り（InceptionV3の一部を利用’）
 
# Features
 
"hoge"のセールスポイントや差別化などを説明する
 
# Requirement
 
"hoge"を動かすのに必要なライブラリなどを列挙する
 
* huga 3.5.2
* hogehuga 1.0.2
 
# Installation
 
Requirementで列挙したライブラリなどのインストール方法を説明する
 
```bash
pip install huga_package
```
 
# Usage
 
DEMOの実行方法など、"hoge"の基本的な使い方を説明する
 
```bash
git clone https://github.com/hoge/~
cd examples
python demo.py
```
 
# Note
 
注意点などがあれば書く
 
# Author
 
作成情報を列挙する
 
* 作成者
* 所属
* E-mail
 
# License
ライセンスを明示する
 
"hoge" is under [MIT license](https://en.wikipedia.org/wiki/MIT_License).
 
社内向けなら社外秘であることを明示してる
 
"hoge" is Confidential.

# sleep_study

<strong>nnによる睡眠段階推定手法</strong>

## 各ブランチの説明
- master : 無駄なファイルを除いた一番軽量なブランチ
- feature/edl : evidential deep learning による分類（要削除）
- feature/edl_1d : evidential deep learning のメイン開発ブランチ（のちにedlに吸収）
- feature/psedo_data : 仮のデータを使った実験ブランチ
- fix/feauture/edl : feature/edl の修正用ブランチ
## 各ブランチ名の頭にはブランチの役割を表すプレフィクスをつける
- feat: 新機能実装
- fix: バグの修正
- docs: ドキュメントのみの変更
- style: コード内のスタイルの変更（改行やフォーマットなどの機能以外の変更）
- refactor: 修正や新機能以外のコードの修正
- perf: パフォーマンス(実行速度)の向上
- test: 機能テストの追加
- chore: makefile,ライブラリ,その他の補足ツールの変更
>（引用：[takadamalab](https://github.com/takadamalab)）