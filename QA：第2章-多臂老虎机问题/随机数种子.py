import numpy as np

# 设置种子,种子设置只影响之后的随机操作
np.random.seed(1)
print(np.random.rand(3))  # 第一次调用，结果随机

np.random.seed(1)  # 设置种子
print(np.random.rand(3))  # 第二次调用，结果固定

# 不同的执行顺序会导致不同结果
np.random.seed(1)
a = np.random.rand(2)
print("a = ",a)
b = np.random.rand(2)  # a和b的值固定
print("b = ",b)

np.random.seed(1)
c = np.random.rand(4)  # c的前两个值与a相同，但整体不同
print("c = ",c)