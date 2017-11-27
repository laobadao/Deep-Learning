"""
吴恩达Coursera深度学习课程 DeepLearning.ai 编程作业（1-2）

Part 1：Python Basics with Numpy (optional assignment)

1 - Building basic functions with numpy

Numpy is the main package for scientific computing in Python. 
It is maintained by a large community (www.numpy.org). 
In this exercise you will learn several key numpy functions such as np.exp, 
np.log, and np.reshape. 
You will need to know how to use these functions for future assignments.

1.1 - sigmoid function, np.exp()

Exercise: Build a function that returns the sigmoid of a real number x.
 Use math.exp(x) for the exponential function.

Reminder: 
sigmoid(x)=  1/(1+e ^ -x) is sometimes also known as the logistic function. 
It is a non-linear function used not only in Machine Learning (Logistic Regression),
 but also in Deep Learning.

 To refer to a function belonging to a specific package you could 
 call it using package_name.function(). 
 Run the code below to see an example with math.exp().

"""

# GRADED FUNCTION: basic_sigmoid

import math

def basic_sigmoid(x):
    """
    Compute sigmoid of x.

    Arguments:
    x -- A scalar

    Return:
    s -- sigmoid(x)
    """

    ### START CODE HERE ### (≈ 1 line of code)
    # math.exp(x) -> e ^ x ,e 的 x 次方
    s = 1.0 / (1 + 1/ math.exp(x))
    ### END CODE HERE ###

    return s

 # Actually, we rarely use the “math” library in deep learning 
 # because the inputs of the functions are real numbers. 
 # In deep learning we mostly use matrices and vectors. 
 # This is why numpy is more useful.
 # 不常用 math 这个库，因为因为它的输入参数为实数，而实际上，在 deep learning 中，我们常用到的训练数据
 # 都是 矩阵 或向量的形式,所以 numpy 这个库，非常的有用

if __name__ == '__main__':

	print(basic_sigmoid(3))




