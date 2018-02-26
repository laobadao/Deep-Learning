import numpy as np 

# 2.11  Vectorization| 向量化 --Andrew Ng 

a = np.array([1,2,3,4,5])

print(a)

# [1 2 3 4 5]
# [Finished in 0.3s]

import time

# 随机创建 百万维度的数组 1000000 个数据
a = np.random.rand(1000000)
b = np.random.rand(1000000)

# 记录当前时间 
tic = time.time()

# 执行计算代码 2 个数组相乘
c = np.dot(a,b)

# 再次记录时间
toc = time.time()

# str(1000*(toc-tic)) 计算运行之间 * 1000 毫秒级
print('Vectorization vresion:',str(1000*(toc-tic)),' ms')
print(c)
# Vectorization vresion: 6.009101867675781  ms
# [Finished in 1.1s]

c = 0
tic = time.time()
for i in range(1000000):
	c += a[i]*b[i]
toc = time.time()
print(c)

print('For loop :',str(1000*(toc-tic)),' ms')
# For loop : 588.9410972595215  ms
# c= 249960.353586
# NOTE: It is obvious that the for loop method  is too slow


