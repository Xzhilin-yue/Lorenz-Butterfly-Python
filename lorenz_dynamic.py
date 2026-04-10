import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings
import matplotlib.animation as animation

# 解决中文乱码+屏蔽警告
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 洛伦兹微分方程
def lorenz(state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

# 欧拉法求解
def solve_lorenz(init_state, dt=0.01, num_steps=10000):
    states = np.zeros((num_steps + 1, 3))
    states[0] = init_state
    for i in range(num_steps):
        states[i+1] = states[i] + lorenz(states[i]) * dt
    return states

# 初始化绘图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
line, = ax.plot([], [], [], color='#f9d71c', lw=0.5)
ax.set_title("洛伦兹吸引子 动画演示", fontsize=16)
ax.set_xlabel("X", fontsize=12)
ax.set_ylabel("Y", fontsize=12)
ax.set_zlabel("Z", fontsize=12)

# 求解轨迹
init_state = [1.0, 1.0, 1.0]
states = solve_lorenz(init_state)

# 设置坐标轴范围
ax.set_xlim(states[:,0].min(), states[:,0].max())
ax.set_ylim(states[:,1].min(), states[:,1].max())
ax.set_zlim(states[:,2].min(), states[:,2].max())

# 动画更新函数
def update(frame):
    line.set_data(states[:frame, 0], states[:frame, 1])
    line.set_3d_properties(states[:frame, 2])
    return line,

# 生成动画
ani = animation.FuncAnimation(fig, update, frames=len(states), interval=0.5, blit=True)
plt.show()