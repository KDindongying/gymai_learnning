import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.colors as mcolors
import os

class WildfireSmoothEnv(gym.Env):
    """
    最终版：平滑动态火场 + 真实灭火逻辑
    1. 形状：平滑的变形虫形状（正弦叠加+平滑滤波），无尖刺。
    2. 判定：必须飞入图形内部（上方）才能灭火。
    3. 视觉：连续的渐变火圈，无散点。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # 动作：[转向, 油门]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 观测：[距离偏差, 相对角度cos, 相对角度sin, 速度, 距离中心, 局部火势]
        high = np.array([np.inf] * 6, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # === 物理参数 ===
        self.dt = 0.1
        self.min_speed = 0.05
        self.max_speed = 0.40
        self.max_turn_rate = np.deg2rad(20)

        # === 火场生成参数 ===
        self.num_vertices = 80          # 增加顶点数以保证平滑度
        self.base_radius = 2.0          # 基础大小
        self.growth_rate = 0.003        # 基础生长速度
        self.smooth_factor = 0.2        # 平滑系数 (0~1)，越大越圆
        
        self.max_steps = 1000
        
        # 运行时状态
        self.plane_pos = None
        self.plane_theta = None
        self.plane_speed = None
        
        self.fire_angles = None    # 极坐标角度
        self.fire_radii = None     # 极坐标半径
        self.fire_health = None    # 0.0(灭) ~ 1.0(旺)
        self.fire_vertices = None  # 笛卡尔坐标缓存

        # 渲染资源
        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # === 1. 生成平滑的不规则形状 (正弦波叠加法) ===
        self.fire_angles = np.linspace(0, 2*np.pi, self.num_vertices, endpoint=False)
        self.fire_radii = np.full(self.num_vertices, self.base_radius)
        
        # 叠加 3 个不同频率的正弦波，制造“变形虫”效果
        for freq in [2, 3, 5]: 
            phase = np.random.uniform(0, 2*np.pi)
            amplitude = np.random.uniform(0.2, 0.5)
            self.fire_radii += amplitude * np.sin(freq * self.fire_angles + phase)
        
        self.fire_health = np.ones(self.num_vertices, dtype=np.float32)
        self._update_vertices()

        # === 2. 初始化飞机 ===
        # 初始在火场外围一段距离
        start_idx = np.random.randint(0, self.num_vertices)
        start_angle = self.fire_angles[start_idx]
        start_dist = self.fire_radii[start_idx] + 1.0 
        
        self.plane_pos = np.array([
            start_dist * np.cos(start_angle),
            start_dist * np.sin(start_angle)
        ], dtype=np.float32)
        
        self.plane_theta = start_angle + np.pi/2 # 切线方向
        self.plane_speed = 0.2
        self.step_count = 0
        
        return self._get_obs(), {}

    def _update_vertices(self):
        xs = self.fire_radii * np.cos(self.fire_angles)
        ys = self.fire_radii * np.sin(self.fire_angles)
        self.fire_vertices = np.stack([xs, ys], axis=1)

    def _smooth_radii(self):
        """物理平滑：模拟表面张力，防止某个点生长过快变成尖刺"""
        # r[i] = 0.1*r[i-1] + 0.8*r[i] + 0.1*r[i+1]
        r = self.fire_radii
        r_prev = np.roll(r, 1)
        r_next = np.roll(r, -1)
        self.fire_radii = 0.1 * r_prev + 0.8 * r + 0.1 * r_next

    def _spread_fire(self):
        # 1. 随机生长 (受健康度限制)
        noise = np.random.normal(0.001, 0.001, self.num_vertices) # 小幅波动
        growth = (self.growth_rate + noise) * self.fire_health
        self.fire_radii += np.maximum(0, growth)
        
        # 2. 强制平滑 (关键步骤：保持形状圆润)
        self._smooth_radii()
        
        # 3. 扩散灭火效果 (如果一个点灭了，旁边也会稍微变弱)
        h = self.fire_health
        h_prev = np.roll(h, 1)
        h_next = np.roll(h, -1)
        # 简单的热传导模型
        self.fire_health = 0.05 * h_prev + 0.9 * h + 0.05 * h_next
        
        self._update_vertices()

    def _get_geometry_info(self):
        """计算飞机相对于火场边缘的位置"""
        # 极坐标判定
        plane_dist = np.linalg.norm(self.plane_pos)
        plane_angle = math.atan2(self.plane_pos[1], self.plane_pos[0])
        if plane_angle < 0: plane_angle += 2*np.pi
        
        # 找到角度最近的索引
        # 由于 angles 是均匀分布的，可以直接计算索引
        idx = int((plane_angle / (2*np.pi)) * self.num_vertices) % self.num_vertices
        
        fire_r = self.fire_radii[idx]
        
        # signed_dist: 正=在外面，负=在里面(上方)
        signed_dist = plane_dist - fire_r
        
        # 切线向量
        p_curr = self.fire_vertices[idx]
        p_next = self.fire_vertices[(idx+1)%self.num_vertices]
        tangent = p_next - p_curr
        tangent /= (np.linalg.norm(tangent) + 1e-6)
        
        return idx, signed_dist, tangent, self.fire_health[idx]

    def _get_obs(self):
        idx, signed_dist, tangent, health = self._get_geometry_info()
        
        plane_vec = np.array([math.cos(self.plane_theta), math.sin(self.plane_theta)])
        dot = np.dot(plane_vec, tangent)
        cross = plane_vec[0]*tangent[1] - plane_vec[1]*tangent[0]
        
        return np.array([
            signed_dist, dot, cross, self.plane_speed, 
            np.linalg.norm(self.plane_pos), health
        ], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        
        # === 动力学更新 ===
        turn = np.clip(action[0], -1.0, 1.0) * self.max_turn_rate
        accel = np.clip(action[1], -1.0, 1.0) * 0.02
        
        self.plane_theta += turn
        self.plane_speed = np.clip(self.plane_speed + accel, self.min_speed, self.max_speed)
        
        vx = self.plane_speed * math.cos(self.plane_theta)
        vy = self.plane_speed * math.sin(self.plane_theta)
        self.plane_pos += np.array([vx, vy])

        # === 灭火逻辑 (更新需求：必须在上方) ===
        idx, signed_dist, tangent, local_health = self._get_geometry_info()
        
        extinguished = 0.0
        # 判定条件：
        # 1. signed_dist < 0.1 (意味着在内部，或者刚好贴着边缘)
        # 2. signed_dist > -0.5 (也不能太深，那是火场中心，太热了)
        if -0.5 < signed_dist < 0.1:
            # 只有在内部才能灭火
            efficiency = 1.0 - (self.plane_speed / self.max_speed) # 飞得慢灭火快
            drop_amount = 0.15 * efficiency
            
            # 影响周围一圈
            impact_indices = [idx, (idx-1)%self.num_vertices, (idx+1)%self.num_vertices]
            for i in impact_indices:
                if self.fire_health[i] > 0:
                    self.fire_health[i] = max(0.0, self.fire_health[i] - drop_amount)
                    extinguished += drop_amount

        # === 环境自然演化 ===
        self._spread_fire()

        # === 奖励 ===
        reward = 0.0
        # 1. 灭火奖励 (大权重)
        reward += extinguished * 20.0
        
        # 2. 距离保持奖励 (希望它贴着边缘飞，稍微靠里一点点)
        target = -0.1 # 目标是在边缘内侧 0.1
        err = abs(signed_dist - target)
        reward += 1.0 * np.exp(-5.0 * err)
        
        # 3. 惩罚
        terminated = False
        truncated = False
        
        if signed_dist > 2.0: # 飞太远
            reward -= 10.0
            terminated = True
        if signed_dist < -1.5: # 飞太深(烧毁)
            reward -= 50.0
            terminated = True
            
        if self.step_count >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_facecolor('#111111') # 深色背景
        
        self.ax.clear()
        
        # === 1. 绘制连续火圈 (Color Line) ===
        # 构造线段点集
        points = self.fire_vertices.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        # 闭合环
        last_seg = np.array([[self.fire_vertices[-1], self.fire_vertices[0]]])
        segments = np.concatenate([segments, last_seg], axis=0)
        
        # 颜色映射：根据 health 决定颜色
        # 0.0 (灭) -> 灰色/蓝色, 1.0 (旺) -> 红色/橙色
        cmap = plt.get_cmap('RdYlBu_r') 
        norm = plt.Normalize(0.0, 1.0)
        colors = cmap(norm(self.fire_health))
        
        # 绘制边缘 (LineCollection 性能高且支持渐变)
        lc = LineCollection(segments, colors=colors, linewidths=3, alpha=0.9)
        self.ax.add_collection(lc)
        
        # 绘制内部填充 (简单的半透明红)
        # 用 fill 填充整体，表现“火场”
        self.ax.fill(self.fire_vertices[:, 0], self.fire_vertices[:, 1], color='#ff3300', alpha=0.2)

        # === 2. 绘制飞机 ===
        # 简单的三角形
        self.ax.scatter(self.plane_pos[0], self.plane_pos[1], s=120, c='white', marker='^', zorder=10)
        
        # === 3. 洒水视效 ===
        # 如果正在灭火区域内，画一个以飞机为中心的淡蓝色圆圈，表示洒水范围
        idx, signed_dist, _, _ = self._get_geometry_info()
        if -0.5 < signed_dist < 0.1:
            water_circle = plt.Circle((self.plane_pos[0], self.plane_pos[1]), 0.3, color='cyan', alpha=0.3)
            self.ax.add_artist(water_circle)
            self.ax.text(self.plane_pos[0], self.plane_pos[1]+0.4, "DROP!", color='cyan', fontsize=8, ha='center')

        # 镜头与信息
        limit = 6.0
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        self.ax.set_title(f"Fire Health: {np.mean(self.fire_health):.2%}", color='white')
        
        plt.draw()
        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)

# ==========================================
# 演示与训练入口
# ==========================================
def train_smooth_wildfire():
    from stable_baselines3 import PPO
    env = WildfireSmoothEnv()
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=None)
    print(">>> 开始训练 (500,000 steps)...")
    model.learn(total_timesteps=500_000)
    model.save("ppo_smooth_fire")
    return model

def visualize_smooth_wildfire():
    from stable_baselines3 import PPO
    
    # 尝试加载模型，没有则随机跑
    if os.path.exists("ppo_smooth_fire.zip"):
        print(">>> 加载模型...")
        model = PPO.load("ppo_smooth_fire")
    else:
        print(">>> 无模型，随机演示...")
        model = None

    env = WildfireSmoothEnv(render_mode="human")
    
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        print(f"Episode {ep+1}")
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # !!! 必须手动调用 render 才有画面 !!!
            env.render()
            
            if terminated or truncated:
                done = True
    env.close()

if __name__ == "__main__":
    train_smooth_wildfire()
    
    # 直接运行看效果（无模型时是随机乱飞，有模型时看智能表现）
    visualize_smooth_wildfire()