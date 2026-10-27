# New implementation checklist / 新規実装チェックリスト

Create a new directory for each cohesive small algorithm or experiment. Split a new repository when the work becomes an independent model-training system, paper reproduction, API/service, or mature library.

- [ ] State the problem and assumptions / 問題設定・仮定
- [ ] Explain the algorithm or mathematical equations / 数式・アルゴリズム
- [ ] Describe what is written from scratch and what uses external libraries / 自作・外部ライブラリの範囲
- [ ] Add reproducible run commands and dependencies / 再現手順
- [ ] Add unit tests and edge cases / テスト
- [ ] Define baseline methods, metrics and dataset / ベースライン・評価指標
- [ ] Record seeds and experiment settings / シード・条件
- [ ] Show results, limitations and next steps / 結果・限界・今後の課題
- [ ] Confirm sharing permissions and remove credentials, personal data, and restricted course material / 公開可否・秘匿情報
- [ ] Keep source labels concise (e.g. 「講義スライド」「グループ発表スライド」); distinguish original derivations from source summaries / 出典は簡潔にし、追加の導出と資料内容を区別

Do not add empty implementation directories merely to suggest completed work. Update the top-level README when adding an actual implementation.
