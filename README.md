# Machine Learning Algorithms & Experiments

機械学習・強化学習・深層学習について、**アルゴリズムの理解 → Pythonでの実装 → テスト → 数値実験 → 結果の考察**を記録するリポジトリです。講義で扱った問題を出発点とした実装と、今後の自主的な再実装・検証を追加していきます。

> This repository collects educational implementations and reproducible experiments in machine learning. It is a learning and experimentation portfolio, not a production ML framework.

## 実装済みの成果物

| Project | 分野 | 内容 | 実装・実験 |
| --- | --- | --- | --- |
| [Multi-Armed Bandit](multi-armed-bandit/) | 強化学習の基礎 | ε-greedy、行動価値の逐次更新、予算制約付き多腕バンディット | [コード](multi-armed-bandit/bandit.py) · [実験結果](multi-armed-bandit/results/summary.csv) · [詳細](multi-armed-bandit/README.md) |

## 今後の拡張方針

Softmax、Attention、最適化アルゴリズム（Adamなど）、ニューラルネットワーク、Transformer等のテーマを、**実際に実装・検証した順に**追加する予定です。未着手のものを実装済みとは記載しません。規模の大きいモデル訓練や独立したアプリケーションは、別リポジトリとして管理する方針です。

新しい成果物の記録形式は [追加用チェックリスト](docs/PROJECT_TEMPLATE.md) にまとめています。

## 実行環境

Python 3.10+。リポジトリのルートで以下を実行してください。

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
python multi-armed-bandit/experiment.py
```

実験スクリプトは `multi-armed-bandit/results/` に集計CSVとグラフを出力します。

## 成果物の位置づけ

ここに掲載するコードは、アルゴリズム実装・分析能力の学習記録です。実サービスの開発・運用経験や、独自の学術的研究成果を示すものではありません。講義課題を出発点とした場合は各プロジェクトにその旨を明記します。

## 公開・再利用について

授業配布資料や第三者のコードは無断転載しません。授業課題の公開条件がある場合はその条件に従ってください。現時点ではライセンスを設定していません。外部コードの再利用、OSSへの転用などを想定する場合は後日適切に設定します。

開発環境の準備と更新方法は [Development Guide](docs/DEVELOPMENT.md) を参照してください。