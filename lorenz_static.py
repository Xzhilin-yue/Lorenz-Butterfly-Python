import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# 字体设置代码在这里！
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------------------- 1. 定义洛伦兹方程组 ----------------------
# 洛伦兹吸引子的核心微分方程
def lorenz_system(state, *, sigma=10, rho=28, beta=8 / 3):
    """
    计算洛伦兹系统在当前状态下的导数
    state: [x, y, z] 当前的三维坐标
    sigma, rho, beta: 经典参数，固定这组值就能画出标准蝴蝶形状
    """
    x, y, z = state
    dx_dt = sigma * (y - x)
    dy_dt = x * (rho - z) - y
    dz_dt = x * y - beta * z
    return np.array([dx_dt, dy_dt, dz_dt])


# ---------------------- 2. 用欧拉法数值求解方程 ----------------------
def solve_lorenz(initial_state, dt=0.01, num_steps=10000):
    """
    用欧拉法迭代计算洛伦兹吸引子的轨迹
    initial_state: 初始坐标 [x0, y0, z0]
    dt: 时间步长，越小越精准
    num_steps: 迭代次数，越多轨迹越完整
    """
    # 初始化存储所有坐标的数组
    states = np.zeros((num_steps + 1, 3))
    states[0] = initial_state  # 第一个点是初始值

    # 迭代计算每一步的坐标
    for i in range(num_steps):
        states[i + 1] = states[i] + lorenz_system(states[i]) * dt

    return states


# ---------------------- 3. 绘制3D蝴蝶曲线 ----------------------
def plot_lorenz(states):
    # 创建3D绘图窗口
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # 绘制轨迹，用黄色线条（和你视频里的颜色一致！）
    ax.plot(states[:, 0], states[:, 1], states[:, 2], color='#f9d71c', lw=0.5)

    # 设置标题和坐标轴
    ax.set_title("洛伦兹吸引子 (Lorenz Attractor)", fontsize=16)
    ax.set_xlabel("X", fontsize=12)
    ax.set_ylabel("Y", fontsize=12)
    ax.set_zlabel("Z", fontsize=12)

    # 美化视角，和视频里的角度接近
    ax.view_init(elev=20, azim=-60)
    plt.show()


# ---------------------- 4. 运行程序 ----------------------
if __name__ == "__main__":
    # 初始坐标（随便选一个小值就行，混沌系统对初始值超敏感！）
    initial_state = [1.0, 1.0, 1.0]
    # 求解轨迹
    states = solve_lorenz(initial_state, dt=0.01, num_steps=10000)
    # 画图！
    plot_lorenz(states)