import pygame
import random
import math
import threading
import queue
import numpy as np
import time
from concurrent.futures import ThreadPoolExecutor
import warnings

try:
    import cupy as cp
    CUDA_AVAILABLE = True
except ImportError:
    CUDA_AVAILABLE = False
    warnings.warn("CUDA acceleration not available. Install cupy for CUDA support.")

# 初始化Pygame
pygame.init()

# 设置窗口大小和标题
WIDTH, HEIGHT = 800, 600
window = pygame.display.set_mode((WIDTH, HEIGHT), pygame.HWSURFACE | pygame.DOUBLEBUF)
pygame.display.set_caption("弹性小球碰撞分裂模拟")

# 定义颜色
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

# 定义固定大圆的参数
CENTER_X, CENTER_Y = WIDTH // 2, HEIGHT // 2
BOUNDARY_RADIUS = 250

# 物理参数
GRAVITY = 0.5
ELASTICITY = 1.2
MIN_BALL_RADIUS = 10
INITIAL_BALL_RADIUS = 30
SPLIT_FORCE = 8.0
SPLIT_ANGLE = 30
MAX_SPEED = 15.0  # 添加最大速度限制

# 多线程参数
MAX_THREADS = 8
SPLIT_QUEUE = queue.Queue()

class Ball:
    def __init__(self, x, y, radius, color, vx=0, vy=0):
        self.x = float(x)
        self.y = float(y)
        self.radius = float(radius)
        self.color = color
        self.vx = float(vx)
        self.vy = float(vy)
        self.should_split = False
        self.active = True
        self.collision_cooldown = 0

    def clamp_speed(self):
        # 限制速度大小
        speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
        if speed > MAX_SPEED:
            scale = MAX_SPEED / speed
            self.vx *= scale
            self.vy *= scale

    def keep_in_bounds(self):
        # 确保球始终在边界内
        distance_to_center = math.sqrt((self.x - CENTER_X)**2 + (self.y - CENTER_Y)**2)
        if distance_to_center + self.radius > BOUNDARY_RADIUS:
            angle = math.atan2(self.y - CENTER_Y, self.x - CENTER_X)
            self.x = CENTER_X + (BOUNDARY_RADIUS - self.radius - 1) * math.cos(angle)
            self.y = CENTER_Y + (BOUNDARY_RADIUS - self.radius - 1) * math.sin(angle)

    def update(self):
        if not self.active:
            return False

        # 更新速度和位置
        self.vy += GRAVITY
        self.clamp_speed()
        
        # 预测下一位置
        next_x = self.x + self.vx
        next_y = self.y + self.vy
        
        # 检查下一位置是否会超出边界
        next_distance = math.sqrt((next_x - CENTER_X)**2 + (next_y - CENTER_Y)**2)
        
        if next_distance + self.radius > BOUNDARY_RADIUS and self.collision_cooldown == 0:
            # 计算碰撞点的角度
            angle = math.atan2(self.y - CENTER_Y, self.x - CENTER_X)
            
            # 计算反弹
            normal_x = math.cos(angle)
            normal_y = math.sin(angle)
            
            # 计算反弹速度
            dot_product = self.vx * normal_x + self.vy * normal_y
            self.vx = self.vx - 2 * dot_product * normal_x
            self.vy = self.vy - 2 * dot_product * normal_y
            
            # 确保反弹速度足够大
            min_speed = math.sqrt(2 * GRAVITY * BOUNDARY_RADIUS) * 1.2
            current_speed = math.sqrt(self.vx * self.vx + self.vy * self.vy)
            if current_speed < min_speed:
                scale = min_speed / current_speed
                self.vx *= scale
                self.vy *= scale
            
            self.should_split = True
            self.collision_cooldown = 5
        
        # 更新位置
        self.x += self.vx
        self.y += self.vy
        
        # 确保位置在边界内
        self.keep_in_bounds()
        
        if self.collision_cooldown > 0:
            self.collision_cooldown -= 1
        
        return True

    def split(self):
        if self.radius > MIN_BALL_RADIUS and self.active:
            new_radius = self.radius / 1.4
            
            # 计算分裂方向
            base_angle = math.atan2(self.vy, self.vx)
            split_angle_rad = math.radians(SPLIT_ANGLE)
            
            # 创建两个新球
            balls = []
            for angle_offset in [split_angle_rad, -split_angle_rad]:
                angle = base_angle + angle_offset
                new_ball = Ball(
                    self.x,
                    self.y,
                    new_radius,
                    random.choice(COLORS),
                    SPLIT_FORCE * math.cos(angle),
                    SPLIT_FORCE * math.sin(angle)
                )
                new_ball.keep_in_bounds()  # 确保新球在边界内
                balls.append(new_ball)
            
            self.active = False
            return balls
        return []

class Renderer:
    def __init__(self):
        self.surface = pygame.Surface((WIDTH, HEIGHT))

    def render(self, window, balls):
        self.surface.fill(BLACK)
        
        # 绘制边界圆
        pygame.draw.circle(self.surface, WHITE, (CENTER_X, CENTER_Y), BOUNDARY_RADIUS, 1)
        
        # 绘制活跃的球
        for ball in balls:
            if ball.active:
                try:
                    pos = (int(ball.x), int(ball.y))
                    radius = int(ball.radius)
                    pygame.draw.circle(self.surface, ball.color, pos, radius)
                except (ValueError, TypeError):
                    continue  # 跳过无效的绘制
        
        window.blit(self.surface, (0, 0))

def main():
    clock = pygame.time.Clock()
    running = True
    fps_update_timer = time.time()
    frames = 0
    
    renderer = Renderer()
    
    # 创建初始小球
    balls = [Ball(
        CENTER_X,
        CENTER_Y - 100,
        INITIAL_BALL_RADIUS,
        random.choice(COLORS),
        vx=random.uniform(-2, 2),
        vy=0
    )]
    
    while running:
        frames += 1
        current_time = time.time()
        
        if current_time - fps_update_timer >= 1.0:
            pygame.display.set_caption(f"弹性小球碰撞分裂模拟 - FPS: {frames} - Balls: {len(balls)}")
            frames = 0
            fps_update_timer = current_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()
                if math.hypot(mx - CENTER_X, my - CENTER_Y) < BOUNDARY_RADIUS:
                    new_ball = Ball(
                        float(mx),
                        float(my),
                        INITIAL_BALL_RADIUS,
                        random.choice(COLORS),
                        vx=random.uniform(-4, 4),
                        vy=random.uniform(-4, 4)
                    )
                    balls.append(new_ball)

        # 更新所有球
        for ball in balls:
            ball.update()
            if ball.should_split:
                new_balls = ball.split()
                balls.extend(new_balls)
                ball.should_split = False

        # 移除非活跃球
        balls = [ball for ball in balls if ball.active]
        
        # 渲染
        renderer.render(window, balls)
        pygame.display.flip()
        
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    main()
