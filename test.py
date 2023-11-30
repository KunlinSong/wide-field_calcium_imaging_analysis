import matplotlib.pyplot as plt
import numpy as np

# 创建一个示例图像
image = np.random.rand(10, 10)

# 创建一个figure和一个axes
fig, ax = plt.subplots()
ax.imshow(image, cmap="gray")

# 存储点的信息
points = []


# 点击事件的回调函数
def onclick(event):
    global points
    # 点的显示大小为图像宽度的一小部分
    display_size = min(*fig.get_size_inches()) * fig.dpi / 50
    # 存储点的位置和大小
    points.append((event.xdata, event.ydata, display_size))
    # 绘制点
    ax.plot(event.xdata, event.ydata, "o", color="red", markersize=display_size)
    fig.canvas.draw()


# 窗口大小调整的回调函数
def onresize(event):
    # 清除之前的点
    ax.cla()
    # 重新显示图像
    ax.imshow(image, cmap="gray")
    # 重新计算和绘制所有点
    new_display_size = min(*fig.get_size_inches()) * fig.dpi / 50
    for x, y, _ in points:
        ax.plot(x, y, "o", color="red", markersize=new_display_size)
    fig.canvas.draw()


# 连接事件处理函数
fig.canvas.mpl_connect("button_press_event", onclick)
fig.canvas.mpl_connect("resize_event", onresize)

# 显示图像
plt.show()
