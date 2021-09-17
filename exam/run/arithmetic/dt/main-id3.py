import matplotlib.pyplot as plt
import numpy as np
# 随机种子
np.random.seed(123)
n_dots = 20
x = np.linspace(0, 1, n_dots)
y = np.sqrt(x) + 0.2 * np.random.rand(n_dots) - 0.1
y0 = x ** 2

# 调用 polyfit 方法
polyRet = np.polyfit(x, y0, 2)
print(polyRet)
polyP = np.poly1d(polyRet)
print(polyP)
print(polyP(-1))

# 调用结果
import cda.g as g

g.gPolyNomial(x, y, 3)
# 显示图像
plt.show()
# 三张子图

plt.figure(figsize= (18,4), dpi = 200)
titles = ["Under Fitting", "Fitting", "Over Fitting"]
for index, deg in enumerate([1,3, 10]):
    plt.subplot(1, 3, index + 1)
    g.gPolyNomial(x, y, deg)
    plt.title(titles[index], fontsize = 20)
# 显示图像
plt.show()

for i, j in enumerate("Hi"):
    print(i, j)
    print("Next")