"""
#字棋 (Tic-Tac-Toe) 棋局识别与最优策略决策

约束：AI 只能下在第二行 (row index 1)，即中间行的三格。
对手可以在任意位置落子。

与 nine_grid_realsense.py 协同：
  - nine_grid_realsense 通过 RealSense 相机识别 3×3 九宫格中的 R/B 棋子
  - 本模块读取 jiu 数组，以 Minimax 算法决策 AI 的最优落子列号

用法:
    from tic_tac_toe_ai import NineGrid

    ng = NineGrid(ai_label='B')          # AI 执蓝方
    ng.from_array(jiu)                    # 从外部数组同步棋局
    col = ng.best_move()                  # 获取最优列号 (0/1/2)，无空位返回 None
    if col is not None:
        ng.jiu[1, col] = ng.ai_label     # 落子
"""

import numpy as np


class NineGrid:
    """
    #字棋 (Tic-Tac-Toe) 棋局识别与最优策略决策

    规则：
      - 只有竖线（列）和斜线（对角线）算胜利，横线不算。
      - AI 只能下在第二行 (row index 1)，即中间行的三格。
      - 对手可以在任意位置落子。
    """

    def __init__(self, ai_label='B'):
        """
        Parameters
        ----------
        ai_label : str
            AI 的棋子标签，'R' 或 'B'。默认 'B'（蓝色方）。
        """
        # dtype=object 以同时兼容整数 0 和字符串 'R'/'B'
        self.jiu = np.array([[0, 0, 0],
                             [0, 0, 0],
                             [0, 0, 0]], dtype=object)
        self.ai_label = ai_label
        self.opponent_label = 'R' if ai_label == 'B' else 'B'

    # ------------------------------------------------------------------
    # 棋盘读写
    # ------------------------------------------------------------------

    def from_array(self, arr):
        """
        从外部数组更新棋局状态。

        Parameters
        ----------
        arr : array-like, shape (3, 3)
            棋盘状态，空位为 0 / '' / '0'，棋子为 'R' / 'B'。
        """
        self.jiu = np.array(arr, dtype=object)

    def _empty(self, r, c, board=None):
        """判断 (r, c) 是否为空位。"""
        if board is None:
            board = self.jiu
        val = board[r, c]
        return val == 0 or val == '' or val == '0'

    # ------------------------------------------------------------------
    # 棋局状态
    # ------------------------------------------------------------------

    def current_winner(self):
        """返回当前胜者 ('R'/'B') 或 None。"""
        return self._check_winner(self.jiu)

    def is_game_over(self):
        """游戏是否结束（有人获胜或棋盘满了）。"""
        if self.current_winner() is not None:
            return True
        return self._is_full(self.jiu)

    def _check_winner(self, board):
        """
        在给定棋盘上检查胜负。

        规则：只有竖线（列）和斜线（对角线）算胜利，横线不算。
        """
        # 检查三列（竖线）
        for c in range(3):
            v = board[0, c]
            if v == board[1, c] == board[2, c] and not self._empty(0, c, board):
                return v
        # 主对角线
        v = board[1, 1]
        if not self._empty(1, 1, board):
            if board[0, 0] == v and board[2, 2] == v:
                return v
            # 副对角线
            if board[0, 2] == v and board[2, 0] == v:
                return v
        return None

    def _is_full(self, board):
        """棋盘是否已满（无空位）。"""
        for r in range(3):
            for c in range(3):
                if self._empty(r, c, board):
                    return False
        return True

    # ------------------------------------------------------------------
    # 可行落子 & 最优决策
    # ------------------------------------------------------------------

    def get_available_moves(self):
        """
        获取 AI 可下的位置。

        Returns
        -------
        list of (int, int)
            空位列表，仅限第二行 (row=1)，如 [(1,0), (1,2)]。
        """
        moves = []
        for c in range(3):
            if self._empty(1, c):
                moves.append((1, c))
        return moves

    def best_move(self):
        """
        Minimax 搜索最优落子列号。

        AI 层只搜索第二行 (row=1)，对手层搜索全部空位。

        Returns
        -------
        int or None
            最优落子列号 (0/1/2)，第二行无空位时返回 None。
        """
        valid = self.get_available_moves()
        if not valid:
            return None

        # 单选项直接返回
        if len(valid) == 1:
            return valid[0][1]

        best_score = float('-inf')
        best_col = valid[0][1]

        # 列优先级：分数相同时优先选两边 (0, 2)，再选中间列 (1)
        col_order = [0, 2, 1]  # 两侧优先的遍历顺序

        for c in col_order:
            if not self._empty(1, c):
                continue
            self.jiu[1, c] = self.ai_label
            score = self._minimax(self.jiu, 0, False)
            self.jiu[1, c] = 0

            if score > best_score:
                best_score = score
                best_col = c

        return best_col

    def _minimax(self, board, depth, is_max):
        """
        Minimax 递归搜索。

        Parameters
        ----------
        board : np.ndarray
            当前棋盘镜像。
        depth : int
            搜索深度（用于偏好更快的胜利）。
        is_max : bool
            True = AI 层（最大化分数），False = 对手层（最小化分数）。

        Returns
        -------
        int
            当前局面的评估分数。
        """
        # 终局判断
        winner = self._check_winner(board)
        if winner == self.ai_label:
            return 10 - depth
        if winner == self.opponent_label:
            return depth - 10
        if self._is_full(board):
            return 0

        if is_max:
            # AI 层：只搜索第二行 (row=1)
            best = float('-inf')
            for c in range(3):
                if self._empty(1, c, board):
                    board[1, c] = self.ai_label
                    score = self._minimax(board, depth + 1, False)
                    board[1, c] = 0
                    best = max(best, score)
            # AI 在第二行无可下位置时，跳过回合让对手继续
            if best == float('-inf'):
                return self._minimax(board, depth, False)
            return best
        else:
            # 对手层：搜索所有空位
            best = float('inf')
            for r in range(3):
                for c in range(3):
                    if self._empty(r, c, board):
                        board[r, c] = self.opponent_label
                        score = self._minimax(board, depth + 1, True)
                        board[r, c] = 0
                        best = min(best, score)
            return best

    # ------------------------------------------------------------------
    # 可视化
    # ------------------------------------------------------------------

    def __str__(self):
        """以文本形式打印当前棋局。"""
        symbol = {0: '.', '': '.', '0': '.'}
        lines = []
        for r in range(3):
            row = []
            for c in range(3):
                val = self.jiu[r, c]
                row.append(symbol.get(val, str(val)))
            lines.append(' '.join(row))
        return '\n'.join(lines)

    def print_board(self):
        """打印当前棋局到控制台。"""
        print(str(self))


# ------------------------------------------------------------------
# 独立测试
# ------------------------------------------------------------------
if __name__ == '__main__':
    print("=" * 40)
    print("#字棋 AI —— 约束: 只能下第二行")
    print("=" * 40)

    ng = NineGrid(ai_label='B')

    # 模拟一局：棋盘上已有若干棋子
    test_boards = [
        # (棋盘状态, 描述)
        ([[0,    0,   0],
          [0,    0,   0],
          [0,    0,   0]],   "空棋盘 → 应选中间列 (col=1)"),

        ([[0,    'R', 0],
          [0,    0,   0],
          [0,    0,   0]],   "对手占第一行中 → 应防守中间"),

        ([[0,    0,   0],
          [0,    'B', 0],
          [0,    0,   0]],   "AI 已占中间 → 继续扩大优势"),

        ([['B',  0,   0],
          [0,    0,   0],
          [0,    0,   'R']], "混合局面"),

        ([[0,    0,   'R'],
          ['B',  0,   'R'],
          [0,    0,   0]],   "对手快赢 → 封堵 col=0"),
    ]

    for board, desc in test_boards:
        ng.from_array(board)
        col = ng.best_move()
        print(f"\n{desc}")
        ng.print_board()
        print(f"→ 推荐落子列: {col}  (cell [1, {col}])")
        if col is not None:
            ng.jiu[1, col] = ng.ai_label
            print("落子后:")
            ng.print_board()
            if ng.current_winner():
                print(f">>> 胜者: {ng.current_winner()}")
