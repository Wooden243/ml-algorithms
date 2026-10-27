
"""有限予算付き多腕バンディット問題のシミュレーション。

10本の腕のうち、各試行で1本だけが良い腕になる。
報酬は成功 +1、失敗 -1。
予算が0になるか、最大T回の選択が終わると終了する。

ε-greedyによる行動選択と、Q値の逐次更新を実装する。
"""

# 型ヒント（型注釈）の評価方法を調整するPythonの機能。
# 強化学習アルゴリズム自体とは直接関係しない。
from __future__ import annotations

# 数値計算ライブラリNumPyをnpという名前で使用する。
import numpy as np


# ===== ① シミュレーション関数の定義 =====

def simulate_budget_path(
    alpha: float,              # 学習率。新しい報酬をQ値に反映する割合
    epsilon: float,            # 探索率。ランダムに腕を選ぶ確率
    initial_budget: int = 20,  # 各試行の初期予算
    K: int = 10,               # 腕の本数
    T: int = 100,              # 1試行の最大選択回数
    n_runs: int = 5_000,       # 独立したシミュレーションの数
    p_good: float = 0.65,      # 良い腕が成功する確率
    p_bad: float = 0.44,       # 普通の腕が成功する確率
    seed: int = 0,             # 擬似乱数生成器のシード
) -> tuple[np.ndarray, dict[str, float]]:
    # 戻り値：
    # ① 各時点の平均予算を格納したNumPy配列
    # ② 実験の集計指標を格納した辞書

    """平均予算の推移と集計結果を返す。

    good_arm_rateは、各試行内での良い腕の選択率を
    求めた後、その値を試行間で平均する。
    """

    # ===== ② 入力値の検証 =====

    # 学習率αは0より大きく、1以下でなければならない。
    if not 0 < alpha <= 1:
        # 不正な値が指定された場合は例外を発生させる。
        raise ValueError("alpha must satisfy 0 < alpha <= 1")

    # 探索率εは0以上、1以下でなければならない。
    if not 0 <= epsilon <= 1:
        raise ValueError("epsilon must satisfy 0 <= epsilon <= 1")

    # 初期予算、腕の本数、最大試行回数、実験回数を確認する。
    # それぞれ正の整数である必要がある。
    if any(
        not isinstance(v, (int, np.integer))
        or isinstance(v, (bool, np.bool_))
        or v < 1
        for v in (initial_budget, K, T, n_runs)
    ):
        raise ValueError(
            "initial_budget, K, T, and n_runs must be positive integers"
        )

    # 成功確率は必ず0以上、1以下である必要がある。
    if not (0 <= p_bad <= 1 and 0 <= p_good <= 1):
        raise ValueError("probabilities must be between 0 and 1")

    # 乱数シードは0以上の整数である必要がある。
    if not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")

    # ===== ③ 環境・学習変数の初期化 =====

    # 指定したシードで乱数生成器を作成する。
    rng = np.random.default_rng(seed)

    # 5,000試行それぞれについて、良い腕を1本ランダムに決める。
    # 例：[3, 8, 0, 3, ...]
    # 数値は腕のインデックス（0～K-1）。
    good_arms = rng.integers(0, K, size=n_runs)

    # 各試行の各腕に対する行動価値Qを0で初期化する。
    # 形状：(n_runs, K) → 通常は(5000, 10)。
    q_values = np.zeros((n_runs, K), dtype=float)

    # すべての試行の現在予算をinitial_budgetに設定する。
    # 形状：(5000,)
    wealth = np.full(n_runs, float(initial_budget))

    # 各試行の時刻0～Tの予算推移を保存する配列を作る。
    # 形状：(5000, 101)。
    budget_paths = np.zeros((n_runs, T + 1), dtype=float)

    # 時刻0の予算を初期予算として記録する。
    # ":" は「すべての行」を意味する。
    budget_paths[:, 0] = wealth

    # 各試行の累積報酬。最初はすべて0。
    total_reward = np.zeros(n_runs)

    # 各試行で腕を何回選択したか。最初は0。
    steps_played = np.zeros(n_runs, dtype=int)

    # 各試行が破産したか。最初は全員False。
    bankrupt = np.zeros(n_runs, dtype=bool)

    # 各試行で良い腕を何回選んだか。最初は0。
    good_arm_count = np.zeros(n_runs)

    # 各試行が現在も継続可能か。
    # 最初は全員True（予算が残っている）。
    alive = np.ones(n_runs, dtype=bool)

    # ===== ④ 時間ステップごとの処理 =====

    # t=1からt=Tまで繰り返す。
    # T=100なら100ステップ分。
    for t in range(1, T + 1):

        # aliveがTrueの試行番号だけ取得する。
        # 破産した試行は今後の行動から除外する。
        active = np.where(alive)[0]

        # 継続中の試行が1つも残っていないか確認する。
        if active.size == 0:

            # 残りの時刻の予算はすべて0と記録する。
            budget_paths[:, t:] = 0

            # 全員破産したためループを終了する。
            break

        # ===== ⑤ ε-greedyによる行動選択 =====

        # Q値が同点の場合に、選択が特定の腕へ偏るのを避ける。
        # 各Q値に極めて小さいランダムな値を加える。
        jitter = rng.uniform(
            0, 1e-12, size=(len(active), K)
        )

        # 各試行について、Q値が最大の腕を選ぶ。
        # axis=1は、各行の中で最大値の列番号を探す指定。
        greedy_actions = np.argmax(
            q_values[active] + jitter, axis=1
        )

        # 各試行について、腕を1本ランダムに選ぶ。
        # 探索用の候補を生成する。
        random_actions = rng.integers(
            0, K, size=len(active)
        )

        # 0以上1未満の乱数を生成し、
        # epsilon未満ならTrue（探索）、それ以外はFalse（活用）。
        explore = rng.random(len(active)) < epsilon

        # exploreがTrueならrandom_actionsを採用。
        # Falseならgreedy_actionsを採用する。
        actions = np.where(
            explore, random_actions, greedy_actions
        )

        # ===== ⑥ 報酬の生成 =====

        # 各試行で、選択した腕が実際の良い腕か判定する。
        # Trueなら良い腕、Falseなら普通の腕。
        selected_is_good = actions == good_arms[active]

        # 良い腕なら成功確率p_good、
        # 普通の腕なら成功確率p_badを設定する。
        p_hit = np.where(
            selected_is_good, p_good, p_bad
        )

        # 乱数が成功確率未満なら成功(True)。
        # そうでなければ失敗(False)。
        hits = rng.random(len(active)) < p_hit

        # 成功なら報酬+1、失敗なら報酬-1。
        rewards = np.where(hits, 1.0, -1.0)

        # ===== ⑦ Q値の更新（学習の中心） =====

        # 各試行で今回選択した腕の古いQ値を取得する。
        old_q = q_values[active, actions]

        # 選択した腕のQ値だけを更新する。
        # 新Q = 旧Q + α × (観測報酬 - 旧Q)
        q_values[active, actions] = (
            old_q + alpha * (rewards - old_q)
        )

        # ===== ⑧ 予算・記録の更新 =====

        # 今回の報酬を現在予算に加える。
        wealth[active] += rewards

        # 今回の報酬を累積報酬にも加える。
        total_reward[active] += rewards

        # 腕を1回選んだので、行動回数を1増やす。
        steps_played[active] += 1

        # 今回良い腕を選んだ試行はカウントを1増やす。
        # Trueは数値演算では1、Falseは0として扱われる。
        good_arm_count[active] += selected_is_good

        # ===== ⑨ 破産判定 =====

        # 現在予算が0以下になった試行を探す。
        newly_bankrupt = wealth[active] <= 0

        # 今回新たに破産した試行が1件以上あるか確認する。
        if np.any(newly_bankrupt):

            # 破産した試行の元のインデックスを取り出す。
            bankrupt_indices = active[newly_bankrupt]

            # 破産済みの記録をTrueに変更する。
            bankrupt[bankrupt_indices] = True

            # 今後の行動選択から除外する。
            alive[bankrupt_indices] = False

            # 破産した試行の予算を0に確定する。
            wealth[bankrupt_indices] = 0.0

        # 全試行の現在予算を時刻tの列に記録する。
        # 破産済みの試行も予算0のまま記録される。
        budget_paths[:, t] = wealth

    # ===== ⑩ 結果の集計 =====

    # 5,000試行の予算を時刻ごとに平均する。
    # axis=0は「行方向に集計し、列ごとの平均を求める」。
    # 結果は時刻0～100の長さ101の配列。
    expected_budget_path = budget_paths.mean(axis=0)

    # 0による除算を防止するための処理。
    # steps_playedと1のうち大きい方を採用する。
    safe_steps = np.maximum(steps_played, 1)

    # 集計結果を辞書形式でまとめる。
    summary = {

        # 使用した学習率。
        "alpha": float(alpha),

        # 使用した探索率。
        "epsilon": float(epsilon),

        # 初期予算。
        "initial_budget": initial_budget,

        # 最終時点における予算の平均値。
        "final_expected_budget": float(
            expected_budget_path[-1]
        ),

        # 各試行の累積報酬の平均値。
        "mean_total_reward": float(
            total_reward.mean()
        ),

        # 破産した試行の割合。
        "bankruptcy_rate": float(
            bankrupt.mean()
        ),

        # 各試行の行動回数の平均。
        "mean_steps": float(
            steps_played.mean()
        ),

        # 試行ごとに良い腕の選択率を求め、その平均を取る。
        "good_arm_rate": float(
            (good_arm_count / safe_steps).mean()
        ),
    }

    # ① 平均予算の推移
    # ② 実験結果の集計辞書
    # を呼び出し元のexperiment.pyに返す。
    return expected_budget_path, summary
