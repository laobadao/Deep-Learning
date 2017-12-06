import time
import numpy as np 

"""
for-loop 实现 点积，外积，逐元乘积

"""

def for_dot(x1,x2):
	"""
	dot product - 点积
	Argument：
		- x1,x2 vectors
	Return:
		- s 标量
	"""
	dot = 0

	tic = time.time()
	for i in range(len(x1)):
		dot += x1[i]*x2[i]
	toc = time.time()

	print("dot=",dot,"run time=",str(1000*(toc-tic)),"ms")
	# dot= 245 run time= 0.0 ms
	return dot

def for_outer(x1,x2):
	"""
	outer product - 外积,
	相当于 将其中一个 向量或矩阵 利用broadcasting，横向或纵向叠加，最后得出 （n,m）

	Argument：
		- x1,x2 vectors
	Return:
		- v 矩阵
	"""
	tic = time.time()
	outer = np.zeros((len(x1),len(x2)))
	for i in range(len(x1)):
		for j in range(len(x2)):
			outer[i,j] = x1[i]+x2[j]
	toc = time.time()

	print("outer=",np.shape(outer),"\n",outer,"\nrun time=",str(1000*(toc-tic)),"ms")
	return outer

def classic_ele(x1,x2):
	"""
	CLASSIC ELEMENTWISE IMPLEMENTATION
	np.multiply  逐元相乘 得到 n 行 1 列的矩阵
	
	x1 = np.array([2,3,4,5,6,7,8,9])
	x2 = np.array([4,3,6,5,8,7,8,2])
	result:[  8.   9.  24.  25.  48.  49.  64.  18.] 
	"""
	tic = time.time()
	mul = np.zeros(len(x1))
	for i in range(len(x1)):
		mul[i] = x1[i]*x2[i]
	toc = time.time()

	print("classic_element=",np.shape(mul),"\n",mul,"\nrun time=",str(1000*(toc-tic)),"ms")
	return 

# def element_mul(x1,x2):

if __name__ == '__main__':

	x1 = np.array([2,3,4,5,6,7,8,9])
	x2 = np.array([4,3,6,5,8,7,8,2])

	for_dot(x1,x2)
	# dot= 245 run time= 0.0 ms
	for_outer(x1,x2)
	classic_ele(x1,x2)
	# classic_element= (8,) 
 	# [  8.   9.  24.  25.  48.  49.  64.  18.] 
	

