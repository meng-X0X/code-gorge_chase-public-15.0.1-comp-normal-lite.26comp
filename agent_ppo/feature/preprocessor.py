#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Feature preprocessor and reward design for Gorge Chase PPO.
峡谷追猎 PPO 特征预处理与奖励设计（优化版 v2）。

优化要点（相较官方简易版）：
  1. 奖励拆分更合理：存活 + 距离delta + 绝对距离小奖励 + 宝箱奖励 + 边界惩罚 + 动作平滑惩罚
  2. 修复 smooth_penalty bug（原 AI 版用 env_obs.get("action") 永远取不到当前动作）
  3. 地图 center 改为动态计算，避免地图尺寸变化时特征系统性偏移
  4. 奖励做 clip 防止极端值破坏 PPO value 估计
  5. 新增宝箱特征（最近宝箱距离 + 宝箱数量）+ 宝箱拾取即时奖励
  6. 保持与框架完全兼容的接口：
       类名         → Preprocessor
       核心方法     → feature_process(self, env_obs, last_action)
       返回格式     → (feature: np.ndarray[42], legal_action: list[int], reward: list[float])

特征向量维度说明（总 42 维）：
  [0:4]   英雄特征   (x_norm, z_norm, flash_cd_norm, buff_norm)            →  4
  [4:9]   怪物0特征  (is_in_view, x_norm, z_norm, speed_norm, dist_norm)   →  5
  [9:14]  怪物1特征  (同上)                                                 →  5
  [14:30] 局部地图   (4×4 障碍二值)                                         → 16
  [30:38] 合法动作   (8 维 0/1 掩码)                                        →  8
  [38:40] 进度特征   (step_norm, step_norm)                                 →  2
  [40:42] 宝箱特征   (nearest_chest_dist_norm, chest_count_norm)            →  2
                                                                      总计 = 42
"""

import numpy as np

# 地图尺寸（128×128）
MAP_SIZE = 128.0
# 最大对角距离
MAX_DISTANCE = MAP_SIZE * 1.41
# 最大怪物速度
MAX_MONSTER_SPEED = 5.0
# 最大闪现冷却步数
MAX_FLASH_CD = 2000.0
# 最大 buff 持续时间
MAX_BUFF_DURATION = 50.0
# 地图中最大宝箱数量
MAX_CHEST_COUNT = 10.0

# ── 奖励超参数 ──────────────────────────────────────────────
SURVIVE_REWARD      =  0.01   # 每步存活基础奖励
DELTA_DIST_WEIGHT   =  0.20   # 距离变化奖励权重（官方 0.1 → 提升至 0.20）
ABS_DIST_WEIGHT     =  0.05   # 绝对距离小奖励（鼓励保持远离怪物）
CHEST_PICKUP_REWARD =  1.00   # 拾取宝箱即时奖励（对应游戏中 100 积分，缩放到奖励尺度）
CHEST_DIST_WEIGHT   =  0.03   # 靠近宝箱的方向性小奖励权重（鼓励在安全时收集）
BOUNDARY_THRESHOLD  =  0.06   # 归一化坐标边界阈值（< 该值或 > 1-该值 触发惩罚）
BOUNDARY_PENALTY    = -0.05   # 靠近边界惩罚（防止 agent 困在角落）
SMOOTH_PENALTY_TURN =  {      # 动作抖动惩罚（按角度差分档）
    4: -0.04,   # 掉头（180°）
    3: -0.02,   # 大转弯（135°）
    2: -0.01,   # 中转弯（90°）
    1:  0.00,   # 小转弯（45°），不惩罚
}
REWARD_CLIP         = (-1.0, 1.0)   # 奖励截断范围
# ────────────────────────────────────────────────────────────


def _norm(v, v_max, v_min=0.0):
    """将值归一化到 [0, 1]。"""
    v = float(np.clip(v, v_min, v_max))
    return (v - v_min) / (v_max - v_min) if (v_max - v_min) > 1e-6 else 0.0


class Preprocessor:
    """
    特征预处理器（优化版 v2），接口与框架官方版完全兼容。
    总特征维度 = 42。
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """重置所有跨帧状态（每局开始时调用）。"""
        self.step_no = 0
        self.max_step = 1000
        self.last_min_monster_dist_norm = 0.5   # 上一步最近怪物归一化距离
        self.prev_action = 0                    # 上上步动作（用于 smooth_penalty）
        self.last_chest_count = 0               # 上一步的宝箱数量（用于检测拾取事件）
        self.last_nearest_chest_dist_norm = 1.0 # 上一步最近宝箱归一化距离

    # ------------------------------------------------------------------
    # 核心接口（框架调用入口）
    # ------------------------------------------------------------------
    def feature_process(self, env_obs, last_action):
        """
        将 env_obs 转换为特征向量、合法动作掩码和即时奖励。

        Args:
            env_obs:     框架传入的环境观测字典
            last_action: 上一步实际执行的动作编号（0~15，前8移动后8闪现）

        Returns:
            feature      (np.ndarray, float32, shape=[42])
            legal_action (list[int], 长度 8)
            reward       (list[float], 长度 1)
        """
        observation = env_obs["observation"]
        frame_state = observation["frame_state"]
        env_info    = observation["env_info"]
        map_info    = observation["map_info"]
        legal_act_raw = observation["legal_action"]

        self.step_no  = observation["step_no"]
        self.max_step = env_info.get("max_step", 1000)

        # ── 1. 英雄自身特征（4 维） ───────────────────────────────────
        hero     = frame_state.get("heroes") or {}
        hero_pos = hero.get("pos") or {}
        hero_x_norm      = _norm(hero_pos.get("x", 0),                    MAP_SIZE)
        hero_z_norm      = _norm(hero_pos.get("z", 0),                    MAP_SIZE)
        flash_cd_norm    = _norm(hero.get("flash_cooldown", 0),    MAX_FLASH_CD)
        buff_remain_norm = _norm(hero.get("buff_remaining_time", 0), MAX_BUFF_DURATION)

        hero_feat = np.array(
            [hero_x_norm, hero_z_norm, flash_cd_norm, buff_remain_norm],
            dtype=np.float32
        )

        # ── 2. 怪物特征（5 维 × 2） ──────────────────────────────────
        monsters = frame_state.get("monsters", [])
        monster_feats = []
        for i in range(2):
            if i < len(monsters):
                m = monsters[i]
                is_in_view = float(m.get("is_in_view", 0))
                m_pos = m.get("pos") or {}   # 防止 pos 字段为 None
                if is_in_view and m_pos:
                    m_x_norm     = _norm(m_pos.get("x", 0),  MAP_SIZE)
                    m_z_norm     = _norm(m_pos.get("z", 0),  MAP_SIZE)
                    m_speed_norm = _norm(m.get("speed", 1),  MAX_MONSTER_SPEED)
                    raw_dist     = np.sqrt(
                        (hero_pos.get("x", 0) - m_pos.get("x", 0)) ** 2 +
                        (hero_pos.get("z", 0) - m_pos.get("z", 0)) ** 2
                    )
                    dist_norm = _norm(raw_dist, MAX_DISTANCE)
                else:
                    m_x_norm = m_z_norm = m_speed_norm = 0.0
                    dist_norm = 1.0   # 不可见 → 视为最远
                monster_feats.append(
                    np.array([is_in_view, m_x_norm, m_z_norm, m_speed_norm, dist_norm],
                             dtype=np.float32)
                )
            else:
                monster_feats.append(np.zeros(5, dtype=np.float32))

        # ── 3. 局部地图特征（4×4 = 16 维） ───────────────────────────
        map_feat = np.zeros(16, dtype=np.float32)
        if map_info is not None and len(map_info) >= 13:
            center  = len(map_info) // 2
            row_len = len(map_info[0]) if map_info else 0
            flat_idx = 0
            for row in range(center - 2, center + 2):
                for col in range(center - 2, center + 2):
                    if 0 <= row < len(map_info) and 0 <= col < row_len:
                        map_feat[flat_idx] = float(map_info[row][col] != 0)
                    flat_idx += 1

        # ── 4. 合法动作掩码（8 维） ───────────────────────────────────
        legal_action = [1] * 8
        if isinstance(legal_act_raw, list) and legal_act_raw:
            if isinstance(legal_act_raw[0], bool):
                for j in range(min(8, len(legal_act_raw))):
                    legal_action[j] = int(legal_act_raw[j])
            else:
                valid_set = {int(a) for a in legal_act_raw if int(a) < 8}
                legal_action = [1 if j in valid_set else 0 for j in range(8)]
        if sum(legal_action) == 0:
            legal_action = [1] * 8   # 全部动作均不合法时回退到全 1

        # ── 5. 进度特征（2 维） ──────────────────────────────────────
        step_norm     = _norm(self.step_no, self.max_step)
        progress_feat = np.array([step_norm, step_norm], dtype=np.float32)

        # ── 6. 宝箱特征（2 维） ──────────────────────────────────────
        # treasure_boxes: 列表，每个元素含 {"pos": {"x":..,"z":..}, ...}
        # 字段名以实际 env_obs 为准，这里同时兼容 "treasure_boxes" 和 "chests"
        chests = frame_state.get("treasure_boxes") or frame_state.get("chests") or []

        cur_chest_count = len(chests)
        nearest_chest_dist_norm = 1.0   # 默认最远（无宝箱或全不可见）

        if chests:
            min_dist = float("inf")
            for chest in chests:
                if not isinstance(chest, dict):
                    continue
                c_pos = chest.get("pos") or {}   # 防止 pos 为 None
                cx = c_pos.get("x", hero_pos.get("x", 0))
                cz = c_pos.get("z", hero_pos.get("z", 0))
                d = np.sqrt(
                    (hero_pos.get("x", 0) - cx) ** 2 +
                    (hero_pos.get("z", 0) - cz) ** 2
                )
                if d < min_dist:
                    min_dist = d
            nearest_chest_dist_norm = _norm(min_dist, MAX_DISTANCE)

        chest_count_norm = _norm(cur_chest_count, MAX_CHEST_COUNT)
        chest_feat = np.array(
            [nearest_chest_dist_norm, chest_count_norm],
            dtype=np.float32
        )

        # ── 7. 拼接特征向量（总 42 维） ──────────────────────────────
        feature = np.concatenate([
            hero_feat,                                      #  4
            monster_feats[0],                               #  5
            monster_feats[1],                               #  5
            map_feat,                                       # 16
            np.array(legal_action, dtype=np.float32),       #  8
            progress_feat,                                  #  2
            chest_feat,                                     #  2
        ])                                                  # = 42

        # ── 8. 计算即时奖励 ──────────────────────────────────────────
        reward = self._compute_reward(
            hero_x_norm, hero_z_norm,
            monster_feats,
            last_action,
            cur_chest_count,
            nearest_chest_dist_norm,
        )

        # 更新宝箱跨帧状态
        self.last_chest_count = cur_chest_count
        self.last_nearest_chest_dist_norm = nearest_chest_dist_norm

        return feature, legal_action, [reward]

    # ------------------------------------------------------------------
    # 内部奖励计算
    # ------------------------------------------------------------------
    def _compute_reward(self, hero_x_norm, hero_z_norm,
                        monster_feats, last_action,
                        cur_chest_count, nearest_chest_dist_norm):
        """
        计算即时奖励（float），综合以下 7 项：

        (1) 存活奖励        +0.01 / step              → 鼓励活得更久
        (2) 距离 delta      ΔDist × 0.20              → 核心：增大与怪物距离
        (3) 绝对距离        MinDist_norm × 0.05        → 辅助：保持远离怪物
        (4) 宝箱拾取        +1.00 / 个                → 拾取宝箱即时激励
        (5) 宝箱方向        ΔChestDist × 0.03          → 靠近宝箱的小奖励
        (6) 边界惩罚        -0.05（靠近地图边缘）     → 防止困在角落
        (7) 动作平滑惩罚    按角度差分档惩罚           → 减少策略抖动
        最终 clip 到 [-1.0, 1.0]
        """
        # (1) 存活奖励
        reward = SURVIVE_REWARD

        # (2) 怪物距离 delta 奖励（主要信号）
        cur_min_dist_norm = 1.0
        for m_feat in monster_feats:
            if m_feat[0] > 0:   # is_in_view
                cur_min_dist_norm = min(cur_min_dist_norm, float(m_feat[4]))

        delta_reward = DELTA_DIST_WEIGHT * (
            cur_min_dist_norm - self.last_min_monster_dist_norm
        )
        self.last_min_monster_dist_norm = cur_min_dist_norm
        reward += delta_reward

        # (3) 绝对距离小奖励（辅助信号，权重小，不抢主导梯度）
        reward += ABS_DIST_WEIGHT * cur_min_dist_norm

        # (4) 宝箱拾取即时奖励
        #     通过比较前后帧宝箱数量判断是否拾取（数量减少 → 被拾取）
        #     注意：宝箱只会被拾取减少，不会自然消失，因此差值 > 0 即为拾取
        chests_collected = max(0, self.last_chest_count - cur_chest_count)
        reward += CHEST_PICKUP_REWARD * chests_collected

        # (5) 宝箱方向奖励：靠近最近宝箱给小奖励（负值 = 远离，正值 = 靠近）
        #     当距离最近怪物较远（相对安全）时，该奖励鼓励 agent 去捡宝箱
        #     为避免 agent 为捡宝箱冒险靠近怪物，只在怪物较远时给正向宝箱方向奖励
        #     cur_min_dist_norm 越大 → 怪物越远 → 宝箱激励权重越高
        safety_factor = cur_min_dist_norm  # [0,1]：越安全，宝箱方向奖励越强
        chest_delta = self.last_nearest_chest_dist_norm - nearest_chest_dist_norm
        reward += CHEST_DIST_WEIGHT * safety_factor * chest_delta

        # (6) 边界惩罚
        if (hero_x_norm < BOUNDARY_THRESHOLD or
                hero_x_norm > 1.0 - BOUNDARY_THRESHOLD or
                hero_z_norm < BOUNDARY_THRESHOLD or
                hero_z_norm > 1.0 - BOUNDARY_THRESHOLD):
            reward += BOUNDARY_PENALTY

        # (7) 动作平滑惩罚
        #     比较"上上步动作 prev_action"与"上步动作 last_action"之间的角度差
        #     动作空间 0~7 为移动，8~15 为闪现；统一折叠到 0~7 做环形比较
        move_action = last_action % 8
        prev_move   = self.prev_action % 8
        diff = abs(move_action - prev_move)
        diff = min(diff, 8 - diff)   # 环形距离（0~7 成圆）
        reward += SMOOTH_PENALTY_TURN.get(diff, 0.0)
        self.prev_action = last_action   # 滚动更新

        # Clip 防止极端值破坏 PPO value 估计
        reward = float(np.clip(reward, REWARD_CLIP[0], REWARD_CLIP[1]))
        return reward
