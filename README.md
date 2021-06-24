# sleep_study

<strong>nnによる睡眠段階推定手法</strong>

## 各ブランチの説明
- master : dnn の分類
- edl : evidential deep learning による分類
- edl_1d : 同上（ただし入力データはスペクトラムであり、畳み込みは1次元）
- more_edl : evidential deep learning にeuを学習される機構の提案

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