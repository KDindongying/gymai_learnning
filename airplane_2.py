import math
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import os

class AirplaneDynamicPolygonEnv(gym.Env):
    """
    修正版：动态扩张多边形
    - 修正了渲染逻辑：镜头固定，不再随物体变大而变焦，确保肉眼可见扩张过程。
    - 增加了参照物：显示初始轮廓以供对比。
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode

        # 动作空间：0=左转, 1=不转, 2=右转
        self.action_space = spaces.Discrete(3)

        # 观测空间
        high = np.array([np.inf] * 5, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # 仿真参数
        self.dt = 0.1
        self.speed = 0.20              # 速度稍快，防止被墙追上
        self.max_turn_rate = np.deg2rad(20) 

        self.max_steps = 1500          
        self.step_count = 0

        # 动态扩张参数
        self.expansion_rate = 0.0      
        self.current_scale = 1.0       
        self.base_vertices = None      
        self.base_path_points = None   
        
        # 运行时数据
        self.polygon_vertices = None   
        self.path_points = None
        self.center = None
        
        # 进度追踪
        self.last_edge_idx = None
        self.edge_progress = 0.0

        # 距离控制
        self.target_dist = 0.25      # 稍微离远一点，安全缓冲
        self.fail_dist_inner = -0.2  # 判定更严格，进墙一点点就死
        self.fail_dist_outer = 1.5   

        # 飞机状态
        self.plane_pos = None
        self.plane_theta = None
        self.last_action = 1

        # 渲染
        self.fig = None
        self.ax = None

    def _generate_random_polygon(self):
        n_vertices = np.random.randint(3, 6) 
        angles = np.sort(np.random.uniform(0, 2 * np.pi, size=n_vertices))
        radii = np.random.uniform(0.8, 1.2, size=n_vertices) # 初始小一点
        xs = radii * np.cos(angles)
        ys = radii * np.sin(angles)
        vertices = np.stack([xs, ys], axis=1)
        return vertices

    def _sample_path_points(self, vertices, points_per_edge=60):
        pts = []
        n = len(vertices)
        for i in range(n):
            p0 = vertices[i]
            p1 = vertices[(i + 1) % n]
            for t in np.linspace(0, 1, points_per_edge, endpoint=False):
                p = (1 - t) * p0 + t * p1
                pts.append(p)
        return np.array(pts, dtype=np.float32)

    def _update_geometry(self):
        # 核心：根据 scale 放大顶点
        self.polygon_vertices = self.base_vertices * self.current_scale
        self.path_points = self.base_path_points * self.current_scale
        self.center = np.mean(self.polygon_vertices, axis=0)

    def _is_inside_polygon(self, point):
        x, y = point
        verts = self.polygon_vertices
        n = len(verts)
        inside = False
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            if (y1 > y) != (y2 > y):
                xinters = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if xinters > x:
                    inside = not inside
        return inside

    def _get_nearest_path_info(self):
        diffs = self.path_points - self.plane_pos
        dists_sq = np.sum(diffs ** 2, axis=1)
        idx = int(np.argmin(dists_sq))
        min_dist = math.sqrt(dists_sq[idx])
        
        is_inside = self._is_inside_polygon(self.plane_pos)
        signed_dist = -min_dist if is_inside else min_dist

        N = len(self.path_points)
        nearest_pt = self.path_points[idx]
        next_pt = self.path_points[(idx + 1) % N]
        tangent = next_pt - nearest_pt
        norm = np.linalg.norm(tangent)
        if norm > 1e-6: tangent /= norm

        return idx, signed_dist, tangent, is_inside

    def _get_obs(self, signed_dist, tangent_vec):
        plane_vec = np.array([math.cos(self.plane_theta), math.sin(self.plane_theta)])
        cross_prod = plane_vec[0]*tangent_vec[1] - plane_vec[1]*tangent_vec[0]
        dot_prod = np.dot(plane_vec, tangent_vec)
        
        obs = np.array([
            signed_dist,        
            dot_prod,           
            cross_prod,         
            self.last_action,   
            1.0 if signed_dist < 0 else 0.0 
        ], dtype=np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.edge_progress = 0.0
        
        # === 调整扩张速度 ===
        # 0.0015 每步，1000步就是增加 1.5倍，非常明显
        self.expansion_rate = self.np_random.uniform(0.001, 0.0025)
        self.current_scale = 1.0

        self.base_vertices = self._generate_random_polygon()
        self.base_path_points = self._sample_path_points(self.base_vertices)
        
        self._update_geometry()

        start_idx = np.random.randint(0, len(self.path_points))
        start_pt = self.path_points[start_idx]
        
        vec_c = start_pt - self.center
        norm_c = np.linalg.norm(vec_c)
        if norm_c > 1e-6: vec_c /= norm_c
        
        self.plane_pos = start_pt + vec_c * self.target_dist
        
        tangent_idx = (start_idx + 1) % len(self.path_points)
        tangent = self.path_points[tangent_idx] - start_pt
        base_angle = math.atan2(tangent[1], tangent[0])
        self.plane_theta = base_angle 

        self.last_edge_idx = start_idx
        self.last_action = 1 

        idx, signed_dist, tangent, inside = self._get_nearest_path_info()
        return self._get_obs(signed_dist, tangent), {}

    def step(self, action):
        self.step_count += 1
        self.last_action = action

        # 1. 扩张
        self.current_scale += self.expansion_rate
        self._update_geometry()

        # 2. 运动
        if action == 0:   dtheta = self.max_turn_rate
        elif action == 2: dtheta = -self.max_turn_rate
        else:             dtheta = 0.0
        
        self.plane_theta += dtheta
        vx = self.speed * math.cos(self.plane_theta)
        vy = self.speed * math.sin(self.plane_theta)
        self.plane_pos += np.array([vx, vy], dtype=np.float32)

        # 3. 观测与奖励
        idx, signed_dist, tangent, inside = self._get_nearest_path_info()

        N = len(self.path_points)
        diff = idx - self.last_edge_idx
        if diff < -N/2: diff += N
        elif diff > N/2: diff -= N
        
        if diff > 0: self.edge_progress += diff
        self.last_edge_idx = idx

        reward = 0.0
        reward += 0.01 
        
        dist_error = abs(signed_dist - self.target_dist)
        reward += 1.0 * np.exp(-8.0 * dist_error) 

        if signed_dist > -0.1 and signed_dist < 0.6: 
             reward += 1.0 * (diff if diff > 0 else 0)

        if inside:
            reward -= (0.5 + 4.0 * abs(signed_dist))
        if action != 1: 
            reward -= 0.05 

        terminated = False
        truncated = False
        success = False

        if self.edge_progress >= N: 
            terminated = True
            success = True
            reward += 50.0 
            
        if signed_dist < self.fail_dist_inner: 
            terminated = True
            reward -= 50.0 
        # 飞离判定随 scale 变化
        elif signed_dist > self.fail_dist_outer * self.current_scale: 
            terminated = True
            reward -= 20.0 
            
        if self.step_count >= self.max_steps:
            truncated = True

        info = {
            "is_success": success, 
            "dist": signed_dist, 
            "scale": self.current_scale
        }
        
        if self.render_mode == "human":
            self.render()

        return self._get_obs(signed_dist, tangent), reward, terminated, truncated, info

    def render(self):
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(6, 6))
        
        self.ax.clear()
        
        # 1. 画初始轮廓 (虚线，灰色) - 作为参照物
        if self.base_vertices is not None:
            base_poly = np.vstack([self.base_vertices, self.base_vertices[0]])
            self.ax.plot(base_poly[:, 0], base_poly[:, 1], color='gray', linestyle='--', alpha=0.5, label="Start Size")

        # 2. 画当前动态多边形 (实线，橙色)
        poly = np.vstack([self.polygon_vertices, self.polygon_vertices[0]])
        self.ax.fill(poly[:, 0], poly[:, 1], color='orange', alpha=0.3)
        self.ax.plot(poly[:, 0], poly[:, 1], color='darkred', linewidth=2, label="Current Size")

        # 3. 画飞机
        color = 'red' if self._is_inside_polygon(self.plane_pos) else 'blue'
        self.ax.scatter(self.plane_pos[0], self.plane_pos[1], s=60, c=color, marker='>', label="Plane")
        
        # 4. 固定镜头范围！不要随 scale 变化！
        # 设定一个足够大的范围，比如 [-8, 8]，这样你看得到它从中间慢慢变大占满屏幕
        fixed_limit = 8.0 
        self.ax.set_xlim(self.center[0] - fixed_limit, self.center[0] + fixed_limit)
        self.ax.set_ylim(self.center[1] - fixed_limit, self.center[1] + fixed_limit)
        
        self.ax.set_title(f"Scale: {self.current_scale:.2f} | Expansion: {self.expansion_rate*1000:.1f}e-3/step")
        self.ax.legend(loc="upper right")
        
        plt.draw()
        plt.pause(0.001)

    def close(self):
        if self.fig:
            plt.close(self.fig)

# ==========================================
# 训练与演示逻辑 (保持不变)
# ==========================================
def train_dynamic_model():
    from stable_baselines3 import PPO
    env = AirplaneDynamicPolygonEnv(render_mode=None)
    model = PPO("MlpPolicy", env, verbose=1, learning_rate=3e-4, n_steps=2048, batch_size=64)
    
    print(">>> 开始训练动态扩张任务... (500,000 steps)")
    model.learn(total_timesteps=500_000)
    model.save("ppo_dynamic_polygon_v2")
    env.close()
    return model

def visualize_dynamic(model=None):
    from stable_baselines3 import PPO
    if model is None:
        if os.path.exists("ppo_dynamic_polygon_v2.zip"):
            model = PPO.load("ppo_dynamic_polygon_v2")
        else:
            print("未找到模型")
            return

    env = AirplaneDynamicPolygonEnv(render_mode="human")
    for episode in range(5):
        obs, info = env.reset()
        print(f"Episode {episode+1} Start")
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                done = True
                print(f"结束 | Final Scale: {info['scale']:.2f} | Success: {info['is_success']}")
    env.close()

if __name__ == "__main__":
    # train_dynamic_model()
    visualize_dynamic()