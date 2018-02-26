
# 2) Vectorization 向量化

**原作者：Koala_Tree** [吴恩达Coursera深度学习课程 DeepLearning.ai 编程作业（1-2）](http://blog.csdn.net/koala_tree/article/details/78057033 )

 /* 本人在对原作者文章进行学习，文章中的部分中文解释，只是在重难点部分加深印象，稍作解释。*/
 
 In deep learning, you deal with very large datasets. Hence, a non-computationally-optimal function can become a huge bottleneck(瓶颈) in your algorithm and can result in a model that takes ages to run.运行很长时间 To make sure that your code is computationally efficient 高效的, you will use vectorization.使用向量化 For example, try to tell the difference between the following implementations of the **dot/outer/elementwise 点积 外积 逐元乘积**  product.
  

使用 **for-loop** 实现 矩阵相关的运算。


```python
import time
import numpy as np

x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### CLASSIC DOT PRODUCT  点积  向量实现中的 点积OF VECTORS IMPLEMENTATION ###
# 点积（英语：Dot Product）
tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot+= x1[i]*x2[i]
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
# dot = 278 

### CLASSIC OUTER PRODUCT 外积 IMPLEMENTATION ###
tic = time.process_time()
outer = np.zeros((len(x1),len(x2))) # we create a len(x1)*len(x2) matrix with only zeros
for i in range(len(x1)):
    for j in range(len(x2)):
        outer[i,j] = x1[i]*x2[j]
toc = time.process_time()
print("np.shape(outer):",np.shape(outer))
# np.shape(outer): (15, 15)
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC ELEMENTWISE 逐元乘积IMPLEMENTATION ###
tic = time.process_time()
mul = np.zeros(len(x1))
for i in range(len(x1)):
    mul[i] = x1[i]*x2[i]
toc = time.process_time()
print("np.shape(mul):",np.shape(mul))
# np.shape(mul): (15,)
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### CLASSIC GENERAL DOT PRODUCT IMPLEMENTATION ###
# Random 3*len(x1) numpy array  W 先初始化为 一个 3 行 len(x1) 列的矩阵
W = np.random.rand(3,len(x1))
tic = time.process_time()
gdot = np.zeros(W.shape[0])
for i in range(W.shape[0]):
    for j in range(len(x1)):
        gdot[i] += W[i,j]*x1[j]
toc = time.process_time()
print("np.shape(gdot):",np.shape(gdot))
# np.shape(gdot): (3,)
print ("gdot = " + str(gdot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
```

    dot = 278
     ----- Computation time = 0.0ms
    np.shape(outer): (15, 15)
    outer = [[ 81.  18.  18.  81.   0.  81.  18.  45.   0.   0.  81.  18.  45.   0.
        0.]
     [ 18.   4.   4.  18.   0.  18.   4.  10.   0.   0.  18.   4.  10.   0.
        0.]
     [ 45.  10.  10.  45.   0.  45.  10.  25.   0.   0.  45.  10.  25.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [ 63.  14.  14.  63.   0.  63.  14.  35.   0.   0.  63.  14.  35.   0.
        0.]
     [ 45.  10.  10.  45.   0.  45.  10.  25.   0.   0.  45.  10.  25.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [ 81.  18.  18.  81.   0.  81.  18.  45.   0.   0.  81.  18.  45.   0.
        0.]
     [ 18.   4.   4.  18.   0.  18.   4.  10.   0.   0.  18.   4.  10.   0.
        0.]
     [ 45.  10.  10.  45.   0.  45.  10.  25.   0.   0.  45.  10.  25.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]
     [  0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.   0.
        0.]]
     ----- Computation time = 0.0ms
    np.shape(mul): (15,)
    elementwise multiplication = [ 81.   4.  10.   0.   0.  63.  10.   0.   0.   0.  81.   4.  25.   0.   0.]
     ----- Computation time = 0.0ms
    np.shape(gdot): (3,)
    gdot = [ 22.80973986  17.27334286  17.79365774]
     ----- Computation time = 0.0ms
    

使用 **numpy** 库中的函数。可对比运算时间观察速度。此处因数据量小，所以运行时间差异并不明显，当数据量大的时候，for-loop 运算和 向量化后的运算，差异是很大的，向量化运算属于并行运算。


```python
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

### VECTORIZED DOT PRODUCT OF VECTORS ###
tic = time.process_time()
dot = np.dot(x1,x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED OUTER PRODUCT ###
tic = time.process_time()
outer = np.outer(x1,x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED ELEMENTWISE MULTIPLICATION ###
tic = time.process_time()
mul = np.multiply(x1,x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")

### VECTORIZED GENERAL DOT PRODUCT ###
tic = time.process_time()
dot = np.dot(W,x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")
```

    dot = 278
     ----- Computation time = 0.0ms
    outer = [[81 18 18 81  0 81 18 45  0  0 81 18 45  0  0]
     [18  4  4 18  0 18  4 10  0  0 18  4 10  0  0]
     [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [63 14 14 63  0 63 14 35  0  0 63 14 35  0  0]
     [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [81 18 18 81  0 81 18 45  0  0 81 18 45  0  0]
     [18  4  4 18  0 18  4 10  0  0 18  4 10  0  0]
     [45 10 10 45  0 45 10 25  0  0 45 10 25  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]
     [ 0  0  0  0  0  0  0  0  0  0  0  0  0  0  0]]
     ----- Computation time = 0.0ms
    elementwise multiplication = [81  4 10  0  0 63 10  0  0  0 81  4 25  0  0]
     ----- Computation time = 0.0ms
    gdot = [ 22.80973986  17.27334286  17.79365774]
     ----- Computation time = 15.625ms
    

As you may have noticed, the vectorized implementation is much cleaner and more efficient. (对比 for-loop 和 使用 numpy 中的函数，你可能已经发现，向量化的实现更加整洁高效)For bigger vectors/matrices, the differences in running time become even bigger.

**Note:** that  `np.dot()` performs a matrix-matrix or matrix-vector multiplication 乘法.（执行 矩阵-矩阵 或 矩阵-向量乘法） This is different from `np.multiply()` and the `*` operator (which is equivalent to `.*` in Matlab/Octave), which performs an element-wise multiplication. 两个操作 是不同的

### 2.1 Implement the L1 and L2 loss functions 实现 L1 and L2 损失函数

**Exercise:** Implement the numpy vectorized version of the L1 loss. You may find the function `abs(x)` (absolute value of x) useful.

**Reminder**: 
- The loss is used to evaluate the performance of your model. The bigger your loss is, the more different your predictions ($\hat{y}$) are from the true values (y). In deep learning, you use optimization algorithms like Gradient Descent to train your model and to minimize the cost. 
- L1 loss is defined as: 

    $\begin{align*} & L_1(\hat{y}, y) = \sum_{i=0}^m|y^{(i)} - \hat{y}^{(i)}| \end{align*}\tag{6}$


```python
# GRADED FUNCTION: L1

def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """

    ### START CODE HERE ### (≈ 1 line of code)
    # np.sum 相加，np.abs 相减     
    loss = np.sum(np.abs(y - yhat))
    ### END CODE HERE ###

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L1 = " + str(L1(yhat,y)))

```

    L1 = 1.1
    

**Exercise:** Implement the numpy vectorized version of the L2 loss. There are several way of implementing the L2 loss but you may find the function np.dot() useful. As a reminder, if $x = [x_1, x_2, ..., x_n]$ , then `np.dot(x,x)` = $\sum_{j=0}^n x_j^{2}$.

hint: 

numpy.power(x1, x2)
数组的元素分别求n次方。x2可以是数字，也可以是数组，但是x1和x2的列数要相同。


```python
# GRADED FUNCTION: L2

def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L2 loss function defined above
    """

    ### START CODE HERE ### (≈ 1 line of code)
    loss =np.sum(np.power((y - yhat), 2))
    ### END CODE HERE ###

    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(L2(yhat,y)))
```

    L2 = 0.43
    

### What to remember: 
- Vectorization is very important in deep learning. It provides computational efficiency and clarity. 向量化在 deep learning 中是十分重要的，它保证了计算的高效和整洁。
- You have reviewed the L1 and L2 loss. 重新评估 L1 and L2 loss 会有什么问题出现
- You are familiar with many numpy functions such as np.sum, np.dot, np.multiply, np.maximum, etc… numpy 中常用函数需掌握。
