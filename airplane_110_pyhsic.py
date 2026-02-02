import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Rectangle, Circle, Polygon, Wedge
from collections import deque
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import torch
import os

# ==========================================
# 3D 战术引导灭火环境 V3.2 (解决“不洒水”问题)
# ==========================================
class WildfireCellularEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)

        # --- 核心物理与战术参数 ---
        self.dt = 0.1
        self.max_thrust = 170000.0    # 增加推力，增强爬升能力
        self.drop_height_min = 25.0   # 稍微降低门槛
        self.drop_height_max = 130.0  
        self.refill_dist = 700.0      
        self.airport_pos = np.array([-2000.0, -2000.0], dtype=np.float32)
        
        # 火场系统
        self.num_cells = 90
        self.fire_cells_pos = None     
        self.fire_cells_health = None  
        self.fire_center_proxy = np.zeros(2) 
        
        # 地形预计算
        self.terrain_grid = np.linspace(-3500, 3500, 40)
        self.mesh_x, self.mesh_y = np.meshgrid(self.terrain_grid, self.terrain_grid)
        self.mesh_z = np.vectorize(self._get_terrain_z)(self.mesh_x, self.mesh_y)

        self.trace_alt = deque(maxlen=100)
        self.trace_terrain = deque(maxlen=100)
        self.fig = None

    def _get_terrain_z(self, x, y):
        dist_air = np.sqrt((x - self.airport_pos[0])**2 + (y - self.airport_pos[1])**2)
        if dist_air < 800: return 0.0
        z = 200 * np.sin(x/1100) * np.cos(y/1100) + 70 * np.cos((x-y)/600)
        return max(0.0, z) * (1.0 if dist_air > 1200 else (dist_air-800)/400.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        angles = np.linspace(0, 2*np.pi, self.num_cells, endpoint=False)
        radii = 350.0 + np.random.uniform(-60, 60, self.num_cells)
        # 火场随机位置，防止过拟合
        self.fire_center_proxy = np.array([np.random.uniform(500, 1200), np.random.uniform(500, 1200)])
        self.fire_cells_pos = self.fire_center_proxy + np.stack([radii*np.cos(angles), radii*np.sin(angles)], axis=1)
        self.fire_cells_health = np.ones(self.num_cells, dtype=np.float32)
        self.fire_cells_wetness = np.zeros(self.num_cells, dtype=np.float32)

        # 飞机初始化：确保有时出生在低空，强迫学习灭火
        start_z = np.random.choice([80.0, 150.0, 300.0]) 
        self.pos = np.array([self.airport_pos[0], self.airport_pos[1], start_z], dtype=np.float32)
        self.velocity, self.heading, self.pitch, self.bank = 110.0, 0.0, 0.0, 0.0
        self.current_water = 6000.0
        self.step_count = 0
        return self._get_obs(), {}

    def _get_obs(self):
        g_z = self._get_terrain_z(self.pos[0], self.pos[1])
        vec_f = self.fire_center_proxy - self.pos[:2]
        dist_f = np.linalg.norm(vec_f)
        ang_f = math.atan2(vec_f[1], vec_f[0]) - self.heading
        vec_a = self.airport_pos - self.pos[:2]
        dist_a = np.linalg.norm(vec_a)
        ang_a = math.atan2(vec_a[1], vec_a[0]) - self.heading
        
        return np.array([
            (self.pos[2]-g_z)/500.0, self.velocity/200.0, self.pitch, self.bank,
            dist_f/5000.0, np.sin(ang_f), np.cos(ang_f),
            dist_a/5000.0, np.sin(ang_a), np.cos(ang_a),
            self.current_water/6000.0, 
            (self._get_terrain_z(self.pos[0]+400*math.cos(self.heading), self.pos[1]+400*math.sin(self.heading))-self.pos[2])/100.0,
            np.mean(self.fire_cells_health), 1.0 if self.current_water < 500 else 0.0,
            np.sin(self.heading), np.cos(self.heading)
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        # 物理模拟
        self.bank = np.clip(self.bank + action[0]*0.4, -np.deg2rad(75), np.deg2rad(75))
        self.pitch = np.clip(self.pitch + action[1]*0.15, -np.deg2rad(40), np.deg2rad(40))
        thrust = ((action[2]+1)/2) * self.max_thrust
        mass = 12000.0 + self.current_water
        accel = (thrust - 0.015*self.velocity**2 - mass*9.81*math.sin(self.pitch)) / mass
        self.velocity = np.clip(self.velocity + accel*0.1, 45.0, 280.0)
        self.heading += (9.81 * math.tan(self.bank) / self.velocity) * 0.1
        self.pos += np.array([self.velocity*math.cos(self.pitch)*math.cos(self.heading), 
                              self.velocity*math.cos(self.pitch)*math.sin(self.heading), 
                              self.velocity*math.sin(self.pitch)]) * 0.1
        
        # 核心逻辑与奖励
        curr_z = self._get_terrain_z(self.pos[0], self.pos[1])
        agl = self.pos[2] - curr_z
        dist_f = np.linalg.norm(self.pos[:2] - self.fire_center_proxy)
        reward = 0.0

        # 1. 坠机惩罚
        if self.pos[2] <= curr_z + 2.0:
            return self._get_obs(), -2000.0, True, False, {}

        # 2. 引导性奖励 (The "Carrot")
        if self.current_water > 0:
            # 越靠近火场奖励越高 (指数级)
            reward += 0.5 * np.exp(-dist_f / 1500.0)
            # 进入火场范围后，如果高度合适，给额外的“瞄准奖励”
            if dist_f < 1500:
                reward += 1.0 * np.exp(-abs(agl - 60.0) / 40.0)
        else:
            # 没水了，离基地越近奖励越高
            dist_a = np.linalg.norm(self.pos[:2] - self.airport_pos)
            reward += 1.0 * np.exp(-dist_a / 1500.0)

        # 3. 灭火交互 (The "Big Win")
        self.is_dropping = False
        in_height_window = (self.drop_height_min < agl < self.drop_height_max)
        if in_height_window and self.current_water > 0 and dist_f < 1500:
            # 检查是否有元胞在下方
            diffs = self.fire_cells_pos - self.pos[:2]
            hits = np.sum(np.sum(diffs**2, axis=1) < 250.0**2) # 增大溅射半径到250m
            if hits > 0:
                self.is_dropping = True
                self.current_water = max(0, self.current_water - 180.0) # 加大排水
                # 只有真正灭到火才给巨额分，且分数与命中数成正比
                reward += hits * 15.0 
                # 更新元胞状态
                hit_mask = np.sum(diffs**2, axis=1) < 250.0**2
                self.fire_cells_health[hit_mask] = np.maximum(0, self.fire_cells_health[hit_mask] - 0.4)
                self.fire_cells_wetness[hit_mask] = 1.0

        # 4. 补水逻辑
        self.is_refilling = False
        dist_a = np.linalg.norm(self.pos[:2] - self.airport_pos)
        if dist_a < self.refill_dist and self.pos[2] < 60 and self.velocity < 110:
            if self.current_water < 6000:
                self.current_water = min(6000, self.current_water + 250)
                self.is_refilling = True
                reward += 5.0 # 补水奖励

        # 演化与结束
        # 火势蔓延速度随健康度变化
        self.fire_cells_pos += (self.fire_cells_pos - self.fire_center_proxy) * 0.0005 * np.expand_dims(self.fire_cells_health, 1)
        self.trace_alt.append(self.pos[2]); self.trace_terrain.append(curr_z)
        
        win = np.mean(self.fire_cells_health) < 0.1
        if win: reward += 5000.0
        
        return self._get_obs(), reward, win, self.step_count > 1600, {}

    def render(self):
        if self.fig is None:
            plt.ion(); self.fig = plt.figure(figsize=(15, 8)); gs = gridspec.GridSpec(2, 2, width_ratios=[2, 1])
            self.ax_m = self.fig.add_subplot(gs[:, 0]); self.ax_p = self.fig.add_subplot(gs[0, 1]); self.ax_s = self.fig.add_subplot(gs[1, 1])
        
        self.ax_m.clear(); self.ax_m.contourf(self.mesh_x, self.mesh_y, self.mesh_z, levels=10, cmap='terrain', alpha=0.4)
        alive = self.fire_cells_health > 0.1
        if np.any(alive): self.ax_m.scatter(self.fire_cells_pos[alive, 0], self.fire_cells_pos[alive, 1], c='red', s=20)
        self.ax_m.add_patch(Circle(self.airport_pos, self.refill_dist, color='lime', alpha=0.2))
        
        h = self.heading; p = self.pos[:2]
        poly = np.array([p + [80*math.cos(h), 80*math.sin(h)], p + [50*math.cos(h+2.6), 50*math.sin(h+2.6)], p + [50*math.cos(h-2.6), 50*math.sin(h-2.6)]])
        self.ax_m.add_patch(Polygon(poly, color='cyan' if self.is_dropping else 'white'))
        if self.is_dropping: self.ax_m.add_patch(Wedge(p, 250, np.rad2deg(h)+150, np.rad2deg(h)+210, color='cyan', alpha=0.4))
        self.ax_m.set_xlim(p[0]-1500, p[0]+1500); self.ax_m.set_ylim(p[1]-1500, p[1]+1500)

        self.ax_p.clear(); self.ax_p.fill_between(range(len(self.trace_alt)), 0, self.trace_terrain, color='#5c4033', alpha=0.7)
        self.ax_p.plot(self.trace_alt, color='cyan', label="ALT"); self.ax_p.set_ylim(0, 600); self.ax_p.legend()
        
        self.ax_s.clear(); self.ax_s.axis('off')
        self.ax_s.text(0.1, 0.8, f"WATER: {self.current_water/6000:.1%}", color='cyan', fontsize=15, fontweight='bold')
        self.ax_s.text(0.1, 0.5, f"FIRE: {np.mean(self.fire_cells_health):.1%}", color='red', fontsize=15, fontweight='bold')
        self.ax_s.text(0.1, 0.2, "ACTION: " + ("DROPPING" if self.is_dropping else "REFILLING" if self.is_refilling else "SCANNING"), color='white')
        plt.draw(); plt.pause(0.001)

# ==========================================
# 训练与运行逻辑 (完全兼容之前的调用)
# ==========================================
def train_dashboard_model():
    from stable_baselines3 import PPO
    print(">>> 启动 V3.2 强化训练 (解决不洒水问题)...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # 增加并行环境，提高探索效率
    env = make_vec_env(WildfireCellularEnv, n_envs=12, vec_env_cls=SubprocVecEnv)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=2e-4, n_steps=2048, batch_size=512, gamma=0.995, device=device)
    model.learn(total_timesteps=4_000_000) # 建议至少 4M 步
    model.save("ppo_fire_v3_2")
    env.close()

def visualize_dashboard_model():
    from stable_baselines3 import PPO
    env = WildfireCellularEnv(render_mode="human")
    model = PPO.load("ppo_fire_v3_2") if os.path.exists("ppo_fire_v3_2.zip") else None
    for _ in range(5):
        obs, _ = env.reset(); done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True) if model else (env.action_space.sample(), None)
            obs, reward, done, trunc, _ = env.step(action); env.render()
            if done or trunc: break
    env.close()

if __name__ == "__main__":
    train_dashboard_model() 
    visualize_dashboard_model()