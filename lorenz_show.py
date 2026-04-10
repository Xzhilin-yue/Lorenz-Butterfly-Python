import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

# 解决中文乱码+屏蔽警告
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def lorenz(state, sigma=10, rho=28, beta=8/3):
    x, y, z = state
    dx = sigma * (y - x)
    dy = x * (rho - z) - y
    dz = x * y - beta * z
    return np.array([dx, dy, dz])

def solve_lorenz(init_state, dt=0.01, num_steps=10000):
    states = np.zeros((num_steps + 1, 3))
    states[0] = init_state
    for i in range(num_steps):
        states[i+1] = states[i] + lorenz(states[i]) * dt
    return states

# 两个几乎一样的初始值（只差百万分之一）
init1 = [1.0, 1.0, 1.0]
init2 = [1.000001, 1.0, 1.0]

states1 = solve_lorenz(init1)
states2 = solve_lorenz(init2)

# 绘图
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot(states1[:,0], states1[:,1], states1[:,2], color='#f9d71c', lw=0.5, label='初始值 [1.0, 1.0, 1.0]')
ax.plot(states2[:,0], states2[:,1], states2[:,2], color='#ff4d4f', lw=0.5, label='初始值 [1.000001, 1.0, 1.0]')
ax.set_title("蝴蝶效应：初始值的微小差异", fontsize=16)
ax.set_xlabel("X", fontsize=12)
ax.set_ylabel("Y", fontsize=12)
ax.set_zlabel("Z", fontsize=12)
ax.legend()
plt.show()