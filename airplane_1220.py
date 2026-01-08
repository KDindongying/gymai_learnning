import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import os

# ==========================================
# 核心环境类：WildfireFinalEnv
# ==========================================
class WildfireFinalEnv(gym.Env):
    """
    终极版环境：
    1. 物理：平滑变形虫形状，被浇灭的边缘锁定不再生长。
    2. 胜利：全场火势(Health)平均值 < 5% 判定胜利。
    3. 视觉：红蓝渐变火线，无散点。
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # 动作：[转向, 油门]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        # 观测空间
        high = np.array([np.inf] * 6, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # 物理参数
        self.dt = 0.1
        self.min_speed = 0.05
        self.max_speed = 0.40
        self.max_turn_rate = np.deg2rad(22) 

        # 火场参数
        self.num_vertices = 90          
        self.base_radius = 2.0          
        self.growth_rate = 0.0025       
        self.max_steps = 1500           
        
        # 状态
        self.plane_pos = None
        self.plane_theta = None
        self.plane_speed = None
        self.fire_angles = None    
        self.fire_radii = None     
        self.fire_health = None    
        self.fire_vertices = None  

        self.fig = None
        self.ax = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 生成不规则形状
        self.fire_angles = np.linspace(0, 2*np.pi, self.num_vertices, endpoint=False)
        self.fire_radii = np.full(self.num_vertices, self.base_radius)
        for freq in [3, 4, 6]: 
            phase = np.random.uniform(0, 2*np.pi)
            amp = np.random.uniform(0.15, 0.4)
            self.fire_radii += amp * np.sin(freq * self.fire_angles + phase)
        
        self.fire_health = np.ones(self.num_vertices, dtype=np.float32)
        self._update_vertices()

        # 2. 初始化飞机
        start_idx = np.random.randint(0, self.num_vertices)
        start_angle = self.fire_angles[start_idx]
        start_dist = self.fire_radii[start_idx] + 1.5 
        
        self.plane_pos = np.array([
            start_dist * np.cos(start_angle),
            start_dist * np.sin(start_angle)
        ], dtype=np.float32)
        
        self.plane_theta = start_angle + np.pi/2 
        self.plane_speed = 0.2
        self.step_count = 0
        
        return self._get_obs(), {}

    def _update_vertices(self):
        xs = self.fire_radii * np.cos(self.fire_angles)
        ys = self.fire_radii * np.sin(self.fire_angles)
        self.fire_vertices = np.stack([xs, ys], axis=1)

    def _smooth_radii(self):
        """物理平滑 + 焦土锁定逻辑"""
        r = self.fire_radii
        r_prev = np.roll(r, 1)
        r_next = np.roll(r, -1)
        smoothed_r = 0.1 * r_prev + 0.8 * r + 0.1 * r_next
        
        # 只有还在燃烧(health>0.1)的点才允许变化形状，灭掉的点锁定形状
        burning_mask = self.fire_health > 0.1
        self.fire_radii = np.where(burning_mask, smoothed_r, self.fire_radii)

    def _spread_fire(self):
        # 随机生长，受Health控制
        noise = np.random.normal(0, 0.0005, self.num_vertices)
        delta = (self.growth_rate + noise) * self.fire_health
        self.fire_radii += np.maximum(0, delta)
        
        self._smooth_radii()
        
        # 热传导 (让灭火效果稍微扩散一点点)
        h = self.fire_health
        h_smoothed = 0.05 * np.roll(h, 1) + 0.9 * h + 0.05 * np.roll(h, -1)
        self.fire_health = h_smoothed
        
        self._update_vertices()

    def _get_geometry_info(self):
        """计算飞机与火场的几何关系"""
        plane_dist = np.linalg.norm(self.plane_pos)
        plane_angle = math.atan2(self.plane_pos[1], self.plane_pos[0])
        if plane_angle < 0: plane_angle += 2*np.pi
        
        idx = int((plane_angle / (2*np.pi)) * self.num_vertices) % self.num_vertices
        fire_r = self.fire_radii[idx]
        
        signed_dist = plane_dist - fire_r
        
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
        
        # 1. 运动
        turn = np.clip(action[0], -1.0, 1.0) * self.max_turn_rate
        accel = np.clip(action[1], -1.0, 1.0) * 0.02
        self.plane_theta += turn
        self.plane_speed = np.clip(self.plane_speed + accel, self.min_speed, self.max_speed)
        
        vx = self.plane_speed * math.cos(self.plane_theta)
        vy = self.plane_speed * math.sin(self.plane_theta)
        self.plane_pos += np.array([vx, vy])

        # 2. 灭火交互
        idx, signed_dist, tangent, local_health = self._get_geometry_info()
        extinguished = 0.0
        
        # 必须切入火线内部 (-0.5 ~ 0.1) 才能洒水
        if -0.5 < signed_dist < 0.1:
            efficiency = 1.0 - (self.plane_speed / self.max_speed)
            drop_amount = 0.15 * efficiency 
            
            # 作用于邻近点
            impact_indices = [idx, (idx-1)%self.num_vertices, (idx+1)%self.num_vertices]
            for i in impact_indices:
                if self.fire_health[i] > 0:
                    self.fire_health[i] = max(0.0, self.fire_health[i] - drop_amount)
                    extinguished += drop_amount

        # 3. 环境演化
        self._spread_fire()

        # 4. 奖励机制
        reward = 0.0
        terminated = False
        truncated = False
        
        reward += extinguished * 20.0 # 灭火主奖励
        
        # 距离保持 (Target=-0.1, 稍微往里一点)
        err = abs(signed_dist - (-0.1))
        reward += 0.5 * np.exp(-5.0 * err)

        # 胜利判定：平均火势 < 5%
        avg_health = np.mean(self.fire_health)
        if avg_health < 0.05:
            terminated = True
            reward += 500.0 # 胜利大奖
            print(f"!!! MISSION COMPLETE at Step {self.step_count} !!!")

        # 失败判定
        if signed_dist > 2.5: # 逃逸
            reward -= 20.0
            terminated = True
        elif signed_dist < -2.0: # 烧死
            reward -= 50.0
            terminated = True
            
        if self.step_count >= self.max_steps:
            truncated = True
            
        return self._get_obs(), reward, terminated, truncated, {}

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
            self.ax.set_facecolor('#111111')
        
        self.ax.clear()
        
        # 画彩色火线
        points = self.fire_vertices.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        last_seg = np.array([[self.fire_vertices[-1], self.fire_vertices[0]]])
        segments = np.concatenate([segments, last_seg], axis=0)
        
        cmap = plt.get_cmap('RdYlBu_r') 
        norm = plt.Normalize(0.0, 1.0)
        colors = cmap(norm(self.fire_health))
        
        lc = LineCollection(segments, colors=colors, linewidths=4, alpha=0.9)
        self.ax.add_collection(lc)
        
        # 内部填充 (随平均火势变暗)
        avg_health = np.mean(self.fire_health)
        self.ax.fill(self.fire_vertices[:, 0], self.fire_vertices[:, 1], color='#ff3300', alpha=0.3*avg_health)

        # 飞机
        self.ax.scatter(self.plane_pos[0], self.plane_pos[1], s=120, c='white', marker='^', zorder=10)
        
        # 洒水光圈
        idx, signed_dist, _, _ = self._get_geometry_info()
        if -0.5 < signed_dist < 0.1:
            water = plt.Circle((self.plane_pos[0], self.plane_pos[1]), 0.4, color='cyan', alpha=0.4)
            self.ax.add_artist(water)

        limit = 6.0
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        
        status = f"Fire Left: {avg_health:.1%}"
        if avg_health < 0.05:
            self.ax.text(0, 0, "SUCCESS", color='lime', fontsize=20, ha='center', fontweight='bold')
            
        self.ax.set_title(status, color='white')
        
        plt.draw()
        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)

# ==========================================
# 训练与运行逻辑 (保持简单结构)
# ==========================================

def train_final_model():
    """训练逻辑：参数已针对高难度任务优化"""
    from stable_baselines3 import PPO
    
    # 1. 创建环境
    env = WildfireFinalEnv()
    
    # 2. 增强模型参数
    # policy_kwargs: 加深网络 [256, 256]，因为火场形状很复杂
    # ent_coef=0.01: 增加随机探索，防止飞机学会“装死”
    policy_kwargs = dict(net_arch=[256, 256])
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,     
        n_steps=2048,           
        batch_size=64,
        ent_coef=0.01,           # <--- 关键：防止陷入局部最优
        policy_kwargs=policy_kwargs,
        # tensorboard_log="./logs/" # 如果不想装tensorboard，这行保持注释或删除
    )
    
    # 建议步数：1,000,000 (100万步)，因为要学会清理全场比较难
    print(">>> 开始训练终极灭火任务 (1,000,000 steps)...")
    try:
        model.learn(total_timesteps=1_000_000)
    except KeyboardInterrupt:
        print("训练被手动打断，正在保存...")
    
    model.save("ppo_firefighter_final")
    print("模型已保存: ppo_firefighter_final.zip")
    return model

def visualize_final_model():
    """演示逻辑：加载模型并渲染"""
    from stable_baselines3 import PPO
    
    if os.path.exists("ppo_firefighter_final.zip"):
        print(">>> 加载训练好的模型...")
        model = PPO.load("ppo_firefighter_final")
    else:
        print(">>> 未找到模型，将使用【随机动作】演示...")
        model = None

    # 必须 render_mode="human"
    env = WildfireFinalEnv(render_mode="human")
    
    for ep in range(5):
        obs, _ = env.reset()
        done = False
        score = 0
        print(f"Episode {ep+1} Start")
        
        while not done:
            if model:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample() # 随机
            
            obs, reward, terminated, truncated, _ = env.step(action)
            
            # !!! 关键：必须在这里调用 render 才有画面 !!!
            env.render()
            
            score += reward
            if terminated or truncated:
                done = True
                print(f"Episode {ep+1} End | Score: {score:.2f}")
    
    env.close()

if __name__ == "__main__":
    # # 1. 训练阶段 (如果已经训练好，可以注释掉这行)
    # train_final_model()
    
    # 2. 演示阶段 (训练完后会自动运行，或者单独运行)
    visualize_final_model()