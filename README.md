# sleep study

2020.4 - 2022.3 までの睡眠の研究まとめ
 u
# DEMO
 ![net_alt](gallery/my_network.png "my_network")
ENNの構造は上記の通り（InceptionV3の一部を利用）
 
 ![git_alt](gallery/out.gif "my_psedo")
仮のデータを用いた実験結果

# Features

* <strong>nnによる睡眠段階推定手法</strong>
 
# Requirement
 * requirements.txtに記載の通り

# Installation 
 
```bash
pip install -r requirements.txt
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


<strong>nnによる睡眠段階推定手法</strong>

## 各ブランチの説明
- master : 無駄なファイルを除いた一番軽量なブランチ
- feature/edl : evidential deep learning による分類（要削除）
- feature/edl_1d : evidential deep learning のメイン開発ブランチ（のちにedlに吸収）
- feature/psedo_data : 仮のデータを使った実験ブランチ
- fix/feauture/edl : feature/edl の修正用ブランチ
- feature/each_ss_unc : 各睡眠段階の不確かさを表示するように改良（masterにマージ済み）
- feature/shift_window_in_any_length : 窓のずらし方を任意の幅にできるように改良（masterにマージ済み）
- feature/sep_learning : 不確かさの大きさに応じてデータセットを作成

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