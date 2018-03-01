
# Assignment | 05-week1 -Character level language model - Dinosaurus land

该系列仅在原课程基础上课后作业部分添加个人学习笔记，如有错误，还请批评指教。- ZJ
    
>[Coursera 课程](https://www.coursera.org/specializations/deep-learning) |[deeplearning.ai](https://www.deeplearning.ai/) |[网易云课堂](https://mooc.study.163.com/smartSpec/detail/1001319001.htm)

 [CSDN]()：
   

---

Welcome to Dinosaurus Island! 65 million years ago, dinosaurs existed, and in this assignment they are back. You are in charge of a special task. Leading biology researchers are creating new breeds of dinosaurs and bringing them to life on earth, and your job is to give names to these dinosaurs. If a dinosaur does not like its name, it might go beserk, so choose wisely! 

<table>
<td>
<img src="images/dino.jpg" style="width:250;height:300px;">

</td>

</table>

Luckily you have learned some deep learning and you will use it to save the day. Your assistant has collected a list of all the dinosaur names they could find, and compiled them into this [dataset](dinos.txt). (Feel free to take a look by clicking the previous link.) To create new dinosaur names, you will build a character level language model to generate new names. Your algorithm will learn the different name patterns, and randomly generate new names. Hopefully this algorithm will keep you and your team safe from the dinosaurs' wrath! 

By completing this assignment you will learn:

- How to store text data for processing using an RNN 
- How to synthesize 合成 data, by sampling  采样predictions at each time step and passing it to the next RNN-cell unit
- How to build a character-level 字符级 text generation recurrent neural network
- Why clipping the gradients 梯度裁剪 is important 防止梯度爆炸

We will begin by loading in some functions that we have provided for you in `rnn_utils`. Specifically, you have access to functions such as `rnn_forward` and `rnn_backward` which are equivalent to those you've implemented in the previous assignment. 


```python
import numpy as np
from utils import *
import random
```


```python
'''utils 中的代码'''

import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def smooth(loss, cur_loss):
    return loss * 0.999 + cur_loss * 0.001

def print_sample(sample_ix, ix_to_char):
    txt = ''.join(ix_to_char[ix] for ix in sample_ix)
    txt = txt[0].upper() + txt[1:]  # capitalize first character 
    print ('%s' % (txt, ), end='')

def get_initial_loss(vocab_size, seq_length):
    return -np.log(1.0/vocab_size)*seq_length


def initialize_parameters(n_a, n_x, n_y):
    """
    Initialize parameters with small random values
    
    Returns:
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    """
    np.random.seed(1)
    Wax = np.random.randn(n_a, n_x)*0.01 # input to hidden
    Waa = np.random.randn(n_a, n_a)*0.01 # hidden to hidden
    Wya = np.random.randn(n_y, n_a)*0.01 # hidden to output
    b = np.zeros((n_a, 1)) # hidden bias
    by = np.zeros((n_y, 1)) # output bias
    
    parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b,"by": by}
    
    return parameters

def rnn_step_forward(parameters, a_prev, x):
    
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    a_next = np.tanh(np.dot(Wax, x) + np.dot(Waa, a_prev) + b) # hidden state
    p_t = softmax(np.dot(Wya, a_next) + by) # unnormalized log probabilities for next chars # probabilities for next chars 
    
    return a_next, p_t

def rnn_step_backward(dy, gradients, parameters, x, a, a_prev):
    
    gradients['dWya'] += np.dot(dy, a.T)
    gradients['dby'] += dy
    da = np.dot(parameters['Wya'].T, dy) + gradients['da_next'] # backprop into h
    daraw = (1 - a * a) * da # backprop through tanh nonlinearity
    gradients['db'] += daraw
    gradients['dWax'] += np.dot(daraw, x.T)
    gradients['dWaa'] += np.dot(daraw, a_prev.T)
    gradients['da_next'] = np.dot(parameters['Waa'].T, daraw)
    return gradients

def update_parameters(parameters, gradients, lr):

    parameters['Wax'] += -lr * gradients['dWax']
    parameters['Waa'] += -lr * gradients['dWaa']
    parameters['Wya'] += -lr * gradients['dWya']
    parameters['b']  += -lr * gradients['db']
    parameters['by']  += -lr * gradients['dby']
    return parameters

def rnn_forward(X, Y, a0, parameters, vocab_size = 27):
    
    # Initialize x, a and y_hat as empty dictionaries
    x, a, y_hat = {}, {}, {}
    
    a[-1] = np.copy(a0)
    
    # initialize your loss to 0
    loss = 0
    
    for t in range(len(X)):
        
        # Set x[t] to be the one-hot vector representation of the t'th character in X.
        # if X[t] == None, we just have x[t]=0. This is used to set the input for the first timestep to the zero vector. 
        x[t] = np.zeros((vocab_size,1)) 
        if (X[t] != None):
            x[t][X[t]] = 1
        
        # Run one step forward of the RNN
        a[t], y_hat[t] = rnn_step_forward(parameters, a[t-1], x[t])
        
        # Update the loss by substracting the cross-entropy term of this time-step from it.
        loss -= np.log(y_hat[t][Y[t],0])
        
    cache = (y_hat, a, x)
        
    return loss, cache

def rnn_backward(X, Y, parameters, cache):
    # Initialize gradients as an empty dictionary
    gradients = {}
    
    # Retrieve from cache and parameters
    (y_hat, a, x) = cache
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    
    # each one should be initialized to zeros of the same dimension as its corresponding parameter
    gradients['dWax'], gradients['dWaa'], gradients['dWya'] = np.zeros_like(Wax), np.zeros_like(Waa), np.zeros_like(Wya)
    gradients['db'], gradients['dby'] = np.zeros_like(b), np.zeros_like(by)
    gradients['da_next'] = np.zeros_like(a[0])
    
    ### START CODE HERE ###
    # Backpropagate through time
    for t in reversed(range(len(X))):
        dy = np.copy(y_hat[t])
        dy[Y[t]] -= 1
        gradients = rnn_step_backward(dy, gradients, parameters, x[t], a[t], a[t-1])
    ### END CODE HERE ###
    
    return gradients, a


```

## 1 - Problem Statement

### 1.1 - Dataset and Preprocessing

Run the following cell to read the dataset of dinosaur names, create a list of unique characters (such as a-z), and compute the dataset and vocabulary size. 


```python
data = open('dinos.txt', 'r').read()
data= data.lower() # 小写
chars = list(set(data)) #先转化为集合 去除重复的，再转化为list
data_size, vocab_size = len(data), len(chars)
print('There are %d total characters and %d unique characters in your data.' % (data_size, vocab_size))
```

    There are 19909 total characters and 27 unique characters in your data.
    

The characters are a-z (26 characters) plus the "\n" (or newline character), which in this assignment plays a role similar to the `<EOS>` (or "End of sentence") token we had discussed in lecture, only here it indicates the end of the dinosaur name rather than the end of a sentence. In the cell below, we create a python dictionary (i.e., a hash table) to map each character to an index from 0-26. We also create a second python dictionary that maps each index back to the corresponding character character. This will help you figure out what index corresponds to what character in the probability distribution output of the softmax layer. Below, `char_to_ix` and `ix_to_char` are the python dictionaries. 


```python
char_to_ix = { ch:i for i,ch in enumerate(sorted(chars)) }
ix_to_char = { i:ch for i,ch in enumerate(sorted(chars)) }
print(ix_to_char)
print(char_to_ix)
```

    {0: '\n', 1: 'a', 2: 'b', 3: 'c', 4: 'd', 5: 'e', 6: 'f', 7: 'g', 8: 'h', 9: 'i', 10: 'j', 11: 'k', 12: 'l', 13: 'm', 14: 'n', 15: 'o', 16: 'p', 17: 'q', 18: 'r', 19: 's', 20: 't', 21: 'u', 22: 'v', 23: 'w', 24: 'x', 25: 'y', 26: 'z'}
    {'\n': 0, 'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5, 'f': 6, 'g': 7, 'h': 8, 'i': 9, 'j': 10, 'k': 11, 'l': 12, 'm': 13, 'n': 14, 'o': 15, 'p': 16, 'q': 17, 'r': 18, 's': 19, 't': 20, 'u': 21, 'v': 22, 'w': 23, 'x': 24, 'y': 25, 'z': 26}
    

### 1.2 - Overview of the model

Your model will have the following structure: 

- Initialize parameters 
- Run the optimization loop
    - Forward propagation to compute the loss function
    - Backward propagation to compute the gradients with respect to the loss function
    - Clip the gradients to avoid exploding gradients
    - Using the gradients, update your parameter with the gradient descent update rule.
- Return the learned parameters 
    
<img src="images/rnn.png" style="width:450;height:300px;">
<caption><center> **Figure 1**: Recurrent Neural Network, similar to what you had built in the previous notebook "Building a RNN - Step by Step".  </center></caption>

At each time-step, the RNN tries to predict what is the next character given the previous characters. The dataset $X = (x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, ..., x^{\langle T_x \rangle})$ is a list of characters in the training set, while $Y = (y^{\langle 1 \rangle}, y^{\langle 2 \rangle}, ..., y^{\langle T_x \rangle})$ is such that at every time-step $t$, we have $y^{\langle t \rangle} = x^{\langle t+1 \rangle}$. 

## 2 - Building blocks of the model

In this part, you will build two important blocks of the overall model:
- Gradient clipping: to avoid exploding gradients 梯度裁剪-避免梯度爆炸
- Sampling: a technique used to generate characters 采样- 生成字符的一个技巧

You will then apply these two functions to build the model.

### 2.1 - Clipping the gradients in the optimization loop

In this section you will implement the `clip` function that you will call inside of your optimization loop. Recall that your overall loop structure usually consists of a forward pass, a cost computation, a backward pass, and a parameter update.  Before updating the parameters, you will perform gradient clipping when needed to make sure that your gradients are not "exploding," meaning taking on overly large values. 

包含 前向广播，损失计算，反向传播，参数更新，4 部分。 在参数更新之前，执行梯度修剪 

In the exercise below, you will implement a function `clip` that takes in a dictionary of gradients and returns a clipped version of gradients if needed. There are different ways to clip gradients; we will use a simple element-wise clipping procedure, in which every element of the gradient vector is clipped to lie between some range [-N, N]. More generally, you will provide a `maxValue` (say 10). In this example, if any component of the gradient vector is greater than 10, it would be set to 10; and if any component of the gradient vector is less than -10, it would be set to -10. If it is between -10 and 10, it is left alone. 

 我们使用最简单的逐元乘积执行裁剪过程 保证其在[-N, N] 区间内 ，设置 一个 最大值 10，如果任何一个 值大于10 了，则设置为 10 ，最小值同理。

<img src="images/clip.png" style="width:400;height:150px;">
<caption><center> **Figure 2**: Visualization of gradient descent with and without gradient clipping, in a case where the network is running into slight "exploding gradient" problems. </center></caption>

**Exercise**: Implement the function below to return the clipped gradients of your dictionary `gradients`. Your function takes in a maximum threshold and returns the clipped versions of your gradients. You can check out this [hint](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.clip.html) for examples of how to clip in numpy. You will need to use the argument `out = ...`.


```python
### GRADED FUNCTION: clip

def clip(gradients, maxValue):
    '''
    Clips the gradients' values between minimum and maximum.
    
    Arguments:
    gradients -- a dictionary containing the gradients "dWaa", "dWax", "dWya", "db", "dby"
    maxValue -- everything above this number is set to this number, and everything less than -maxValue is set to -maxValue
    
    Returns: 
    gradients -- a dictionary with the clipped gradients.
    '''
   
    ### START CODE HERE ###
    # clip to mitigate exploding gradients, loop over [dWax, dWaa, dWya, db, dby]. (≈2 lines)
    for name,val in gradients.items():
        gradients[name] = np.clip(val, -maxValue, maxValue, out=gradients[name])
    ### END CODE HERE ###
    
    return gradients
```

所以，这里使用的梯度裁剪，仅仅是设置了，设置了最大值和最小值区间。去除了较大或较小的部分。


```python
np.random.seed(3)
dWax = np.random.randn(5,3)*10
dWaa = np.random.randn(5,5)*10
dWya = np.random.randn(2,5)*10
db = np.random.randn(5,1)*10
dby = np.random.randn(2,1)*10
gradients = {"dWax": dWax, "dWaa": dWaa, "dWya": dWya, "db": db, "dby": dby}
gradients = clip(gradients, 10)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("gradients[\"dWax\"][3][1] =", gradients["dWax"][3][1])
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
```

    gradients["dWaa"][1][2] = 10.0
    gradients["dWax"][3][1] = -10.0
    gradients["dWya"][1][2] = 0.2971381536101662
    gradients["db"][4] = [10.]
    gradients["dby"][1] = [8.45833407]
    

** Expected output:**

<table>
<tr>
    <td> 
    **gradients["dWaa"][1][2] **
    </td>
    <td> 
    10.0
    </td>
</tr>

<tr>
    <td> 
    **gradients["dWax"][3][1]**
    </td>
    <td> 
    -10.0
    </td>
    </td>
</tr>
<tr>
    <td> 
    **gradients["dWya"][1][2]**
    </td>
    <td> 
0.29713815361
    </td>
</tr>
<tr>
    <td> 
    **gradients["db"][4]**
    </td>
    <td> 
[ 10.]
    </td>
</tr>
<tr>
    <td> 
    **gradients["dby"][1]**
    </td>
    <td> 
[ 8.45833407]
    </td>
</tr>

</table>

### 2.2 - Sampling

Now assume that your model is trained. You would like to generate new text (characters). The process of generation is explained in the picture below:

<img src="images/dinos3.png" style="width:500;height:300px;">
<caption><center> **Figure 3**: In this picture, we assume the model is already trained. We pass in $x^{\langle 1\rangle} = \vec{0}$ at the first time step, and have the network then sample one character at a time. </center></caption>

**Exercise**: Implement the `sample` function below to sample characters. You need to carry out 4 steps:

- **Step 1**: Pass the network the first "dummy" input $x^{\langle 1 \rangle} = \vec{0}$ (the vector of zeros). This is the default input before we've generated any characters. We also set $a^{\langle 0 \rangle} = \vec{0}$

- **Step 2**: Run one step of forward propagation to get $a^{\langle 1 \rangle}$ and $\hat{y}^{\langle 1 \rangle}$. Here are the equations:

$$ a^{\langle t+1 \rangle} = \tanh(W_{ax}  x^{\langle t \rangle } + W_{aa} a^{\langle t \rangle } + b)\tag{1}$$

$$ z^{\langle t + 1 \rangle } = W_{ya}  a^{\langle t + 1 \rangle } + b_y \tag{2}$$

$$ \hat{y}^{\langle t+1 \rangle } = softmax(z^{\langle t + 1 \rangle })\tag{3}$$

Note that $\hat{y}^{\langle t+1 \rangle }$ is a (softmax) probability vector (its entries are between 0 and 1 and sum to 1). $\hat{y}^{\langle t+1 \rangle}_i$ represents the probability that the character indexed by "i" is the next character.  We have provided a `softmax()` function that you can use.

- **Step 3**: Carry out sampling: Pick the next character's index according to the probability distribution specified by $\hat{y}^{\langle t+1 \rangle }$. This means that if $\hat{y}^{\langle t+1 \rangle }_i = 0.16$, you will pick the index "i" with 16% probability. To implement it, you can use [`np.random.choice`](https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.random.choice.html).

Here is an example of how to use `np.random.choice()`:
```python
np.random.seed(0)
p = np.array([0.1, 0.0, 0.7, 0.2])
index = np.random.choice([0, 1, 2, 3], p = p.ravel())
```
This means that you will pick the `index` according to the distribution: 
$P(index = 0) = 0.1, P(index = 1) = 0.0, P(index = 2) = 0.7, P(index = 3) = 0.2$.

- **Step 4**: The last step to implement in `sample()` is to overwrite the variable `x`, which currently stores $x^{\langle t \rangle }$, with the value of $x^{\langle t + 1 \rangle }$. You will represent $x^{\langle t + 1 \rangle }$ by creating a one-hot vector corresponding to the character you've chosen as your prediction. You will then forward propagate $x^{\langle t + 1 \rangle }$ in Step 1 and keep repeating the process until you get a "\n" character, indicating you've reached the end of the dinosaur name. 


```python
# GRADED FUNCTION: sample

def sample(parameters, char_to_ix, seed):
    """
    Sample a sequence of characters according to a sequence of probability distributions output of the RNN 

    采样简单理解为 随机选取

    Arguments:
    parameters -- python dictionary containing the parameters Waa, Wax, Wya, by, and b. 
    char_to_ix -- python dictionary mapping each character to an index.
    seed -- used for grading purposes. Do not worry about it.

    Returns:
    indices -- a list of length n containing the indices of the sampled characters.
    """
    
    # Retrieve parameters and relevant shapes from "parameters" dictionary
    Waa, Wax, Wya, by, b = parameters['Waa'], parameters['Wax'], parameters['Wya'], parameters['by'], parameters['b']
    vocab_size = by.shape[0] # by: (27, 1) 根据上述公式 tag 2 中的 计算 z 时，因为是在字符级 的基础上做预测，所以 by 的行坐标 与 词汇表大小相同
    n_a = Waa.shape[1] 
    
#     print("Waa.shape:", Waa.shape)
#     print('by:', by.shape)
#     print('b:', b.shape)
    
    # Waa.shape: (100, 100)
    #   by: (27, 1)
    
    ### START CODE HERE ###
    # Step 1: Create the one-hot vector x for the first character (initializing the sequence generation). (≈1 line)
    # x 是一个 one-hot 向量 维度是（27,1） x 字符级 所以是 27 个字符中任意一个
    x = np.zeros((vocab_size, 1))
    # Step 1': Initialize a_prev as zeros (≈1 line) 记住，这是 字符级别的，都相当于是向量
    a_prev = np.zeros((n_a,1 ))
    
    # Create an empty list of indices, this is the list which will contain the list of indices of the characters to generate (≈1 line)
    indices = []
    
    # Idx is a flag to detect a newline character, we initialize it to -1
    idx = -1 
    
    # Loop over time-steps t. At each time-step, sample a character from a probability distribution and append 
    # its index to "indices". We'll stop if we reach 50 characters (which should be very unlikely with a well 
    # trained model), which helps debugging and prevents entering an infinite loop. 
    counter = 0
    newline_character = char_to_ix['\n'] # 返回的是字符“\n”所在索引位置
    
    while (idx != newline_character and counter != 50):
        
        # Step 2: Forward propagate x using the equations (1), (2) and (3)
        a = np.tanh(np.matmul(Wax, x) + np.matmul(Waa, a_prev) + b)
        z = np.matmul(Wya, a) + by
        y = softmax(z)
        
        # for grading purposes
        np.random.seed(counter+seed) 
        
        # Step 3: Sample the index of a character within the vocabulary from the probability distribution y
        idx = np.random.choice(range(vocab_size), p = y.ravel())

        # Append the index to "indices"
        indices.append(idx)
        
        # Step 4: Overwrite the input character as the one corresponding to the sampled index.
        x = np.zeros((vocab_size, 1))
        x[idx] = 1
        
        # Update "a_prev" to be "a"
        a_prev = a
        
        # for grading purposes
        seed +=1
        counter +=1
        
    ### END CODE HERE ###

    if (counter == 50):
        indices.append(char_to_ix['\n'])
    
    return indices
```

错误记录：




```python
np.random.seed(2)
_, n_a = 20, 100
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}


indices = sample(parameters, char_to_ix, 0)
print("Sampling:")
print("list of sampled indices:", indices)
print("list of sampled characters:", [ix_to_char[i] for i in indices])
```

    Sampling:
    list of sampled indices: [12, 17, 24, 14, 13, 9, 10, 22, 24, 6, 13, 11, 12, 6, 21, 15, 21, 14, 3, 2, 1, 21, 18, 24, 7, 25, 6, 25, 18, 10, 16, 2, 3, 8, 15, 12, 11, 7, 1, 12, 10, 2, 7, 7, 11, 3, 6, 23, 13, 1, 0]
    list of sampled characters: ['l', 'q', 'x', 'n', 'm', 'i', 'j', 'v', 'x', 'f', 'm', 'k', 'l', 'f', 'u', 'o', 'u', 'n', 'c', 'b', 'a', 'u', 'r', 'x', 'g', 'y', 'f', 'y', 'r', 'j', 'p', 'b', 'c', 'h', 'o', 'l', 'k', 'g', 'a', 'l', 'j', 'b', 'g', 'g', 'k', 'c', 'f', 'w', 'm', 'a', '\n']
    

** Expected output:**
<table>
<tr>
    <td> 
    **list of sampled indices:**
    </td>
    <td> 
    [12, 17, 24, 14, 13, 9, 10, 22, 24, 6, 13, 11, 12, 6, 21, 15, 21, 14, 3, 2, 1, 21, 18, 24, <br>
    7, 25, 6, 25, 18, 10, 16, 2, 3, 8, 15, 12, 11, 7, 1, 12, 10, 2, 7, 7, 11, 5, 6, 12, 25, 0, 0]
    </td>
    </tr><tr>
    <td> 
    **list of sampled characters:**
    </td>
    <td> 
    ['l', 'q', 'x', 'n', 'm', 'i', 'j', 'v', 'x', 'f', 'm', 'k', 'l', 'f', 'u', 'o', <br>
    'u', 'n', 'c', 'b', 'a', 'u', 'r', 'x', 'g', 'y', 'f', 'y', 'r', 'j', 'p', 'b', 'c', 'h', 'o', <br>
    'l', 'k', 'g', 'a', 'l', 'j', 'b', 'g', 'g', 'k', 'e', 'f', 'l', 'y', '\n', '\n']
    </td>
    
        
    
</tr>
</table>

## 3 - Building the language model 

It is time to build the character-level language model for text generation. 


### 3.1 - Gradient descent 

In this section you will implement a function performing one step of stochastic gradient descent (with clipped gradients). You will go through the training examples one at a time, so the optimization algorithm will be stochastic gradient descent. As a reminder, here are the steps of a common optimization loop for an RNN:

- Forward propagate through the RNN to compute the loss
- Backward propagate through time to compute the gradients of the loss with respect to the parameters
- Clip the gradients if necessary 
- Update your parameters using gradient descent 

**Exercise**: Implement this optimization process (one step of stochastic gradient descent). 

We provide you with the following functions: 

```python
def rnn_forward(X, Y, a_prev, parameters):
    """ Performs the forward propagation through the RNN and computes the cross-entropy loss.
    It returns the loss' value as well as a "cache" storing values to be used in the backpropagation."""
    ....
    return loss, cache
    
def rnn_backward(X, Y, parameters, cache):
    """ Performs the backward propagation through time to compute the gradients of the loss with respect
    to the parameters. It returns also all the hidden states."""
    ...
    return gradients, a

def update_parameters(parameters, gradients, learning_rate):
    """ Updates parameters using the Gradient Descent Update Rule."""
    ...
    return parameters
```


```python
# GRADED FUNCTION: optimize

def optimize(X, Y, a_prev, parameters, learning_rate = 0.01):
    """
    Execute one step of the optimization to train the model.
    
    Arguments:
    X -- list of integers, where each integer is a number that maps to a character in the vocabulary.
    Y -- list of integers, exactly the same as X but shifted one index to the left.
    a_prev -- previous hidden state.
    parameters -- python dictionary containing:
                        Wax -- Weight matrix multiplying the input, numpy array of shape (n_a, n_x)
                        Waa -- Weight matrix multiplying the hidden state, numpy array of shape (n_a, n_a)
                        Wya -- Weight matrix relating the hidden-state to the output, numpy array of shape (n_y, n_a)
                        b --  Bias, numpy array of shape (n_a, 1)
                        by -- Bias relating the hidden-state to the output, numpy array of shape (n_y, 1)
    learning_rate -- learning rate for the model.
    
    Returns:
    loss -- value of the loss function (cross-entropy)
    gradients -- python dictionary containing:
                        dWax -- Gradients of input-to-hidden weights, of shape (n_a, n_x)
                        dWaa -- Gradients of hidden-to-hidden weights, of shape (n_a, n_a)
                        dWya -- Gradients of hidden-to-output weights, of shape (n_y, n_a)
                        db -- Gradients of bias vector, of shape (n_a, 1)
                        dby -- Gradients of output bias vector, of shape (n_y, 1)
    a[len(X)-1] -- the last hidden state, of shape (n_a, 1)
    """
    
    ### START CODE HERE ###
    
    # Forward propagate through time (≈1 line)
    loss, cache = rnn_forward(X, Y, a_prev, parameters)
    
    # Backpropagate through time (≈1 line)
    gradients, a = rnn_backward(X, Y, parameters, cache)
    
    # Clip your gradients between -5 (min) and 5 (max) (≈1 line)
    gradients = clip(gradients, 5)
    
    # Update parameters (≈1 line)
    parameters = update_parameters(parameters, gradients, learning_rate)
    
    ### END CODE HERE ###
    
    return loss, gradients, a[len(X)-1]
```


```python
np.random.seed(1)
vocab_size, n_a = 27, 100
a_prev = np.random.randn(n_a, 1)
Wax, Waa, Wya = np.random.randn(n_a, vocab_size), np.random.randn(n_a, n_a), np.random.randn(vocab_size, n_a)
b, by = np.random.randn(n_a, 1), np.random.randn(vocab_size, 1)
parameters = {"Wax": Wax, "Waa": Waa, "Wya": Wya, "b": b, "by": by}
X = [12,3,5,11,22,3]
Y = [4,14,11,22,25, 26]

loss, gradients, a_last = optimize(X, Y, a_prev, parameters, learning_rate = 0.01)
print("Loss =", loss)
print("gradients[\"dWaa\"][1][2] =", gradients["dWaa"][1][2])
print("np.argmax(gradients[\"dWax\"]) =", np.argmax(gradients["dWax"]))
print("gradients[\"dWya\"][1][2] =", gradients["dWya"][1][2])
print("gradients[\"db\"][4] =", gradients["db"][4])
print("gradients[\"dby\"][1] =", gradients["dby"][1])
print("a_last[4] =", a_last[4])
```

    Loss = 126.50397572165383
    gradients["dWaa"][1][2] = 0.1947093153471825
    np.argmax(gradients["dWax"]) = 93
    gradients["dWya"][1][2] = -0.007773876032003897
    gradients["db"][4] = [-0.06809825]
    gradients["dby"][1] = [0.01538192]
    a_last[4] = [-1.]
    

** Expected output:**

<table>


<tr>
    <td> 
    **Loss **
    </td>
    <td> 
    126.503975722
    </td>
</tr>
<tr>
    <td> 
    **gradients["dWaa"][1][2]**
    </td>
    <td> 
    0.194709315347
    </td>
<tr>
    <td> 
    **np.argmax(gradients["dWax"])**
    </td>
    <td> 93
    </td>
</tr>
<tr>
    <td> 
    **gradients["dWya"][1][2]**
    </td>
    <td> -0.007773876032
    </td>
</tr>
<tr>
    <td> 
    **gradients["db"][4]**
    </td>
    <td> [-0.06809825]
    </td>
</tr>
<tr>
    <td> 
    **gradients["dby"][1]**
    </td>
    <td>[ 0.01538192]
    </td>
</tr>
<tr>
    <td> 
    **a_last[4]**
    </td>
    <td> [-1.]
    </td>
</tr>

</table>

### 3.2 - Training the model 

Given the dataset of dinosaur names, we use each line of the dataset (one name) as one training example. Every 100 steps of stochastic gradient descent, you will sample 10 randomly chosen names to see how the algorithm is doing. Remember to shuffle the dataset, so that stochastic gradient descent visits the examples in random order.  先将数据随机打乱，这样可以随机选取任意样本

**Exercise**: Follow the instructions and implement `model()`. When `examples[index]` contains one dinosaur name (string), to create an example (X, Y), you can use this:
```python
        index = j % len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]] 
        Y = X[1:] + [char_to_ix["\n"]]
```
Note that we use: `index= j % len(examples)`, where `j = 1....num_iterations`, to make sure that `examples[index]` is always a valid statement (`index` is smaller than `len(examples)`).
The first entry of `X` being `None` will be interpreted by `rnn_forward()` as setting $x^{\langle 0 \rangle} = \vec{0}$. Further, this ensures that `Y` is equal to `X` but shifted one step to the left, and with an additional "\n" appended to signify the end of the dinosaur name. 


```python
# GRADED FUNCTION: model

def model(data, ix_to_char, char_to_ix, num_iterations = 35000, n_a = 50, dino_names = 7, vocab_size = 27):
    """
    Trains the model and generates dinosaur names. 
    
    Arguments:
    data -- text corpus
    ix_to_char -- dictionary that maps the index to a character
    char_to_ix -- dictionary that maps a character to an index
    num_iterations -- number of iterations to train the model for
    n_a -- number of units of the RNN cell
    dino_names -- number of dinosaur names you want to sample at each iteration.  每次迭代 采样 7 个名字
    vocab_size -- number of unique characters found in the text, size of the vocabulary
    
    Returns:
    parameters -- learned parameters
    """
    
    # Retrieve 恢复 n_x and n_y from vocab_size
    n_x, n_y = vocab_size, vocab_size
    
    # Initialize parameters
    parameters = initialize_parameters(n_a, n_x, n_y)
    
    # Initialize loss (this is required because we want to smooth our loss, don't worry about it)
    loss = get_initial_loss(vocab_size, dino_names)
    
    # Build list of all dinosaur names (training examples).
    with open("dinos.txt") as f:
        examples = f.readlines()
    examples = [x.lower().strip() for x in examples]
    
    
    # Shuffle list of all dinosaur names
    np.random.seed(0)
    np.random.shuffle(examples)
    
    # Initialize the hidden state of your LSTM
    a_prev = np.zeros((n_a, 1))
    
    # Optimization loop
    for j in range(num_iterations):
        
        ### START CODE HERE ###
        
        # Use the hint above to define one training example (X,Y) (≈ 2 lines)
        index = j%len(examples)
        X = [None] + [char_to_ix[ch] for ch in examples[index]] 
        Y = X[1:] + [char_to_ix["\n"]]
        
        # Perform one optimization step: Forward-prop -> Backward-prop -> Clip -> Update parameters
        # Choose a learning rate of 0.01
        curr_loss, gradients, a_prev = optimize(X, Y, a_prev, parameters)
        
        ### END CODE HERE ###
        
        # Use a latency trick to keep the loss smooth. It happens here to accelerate the training.
        loss = smooth(loss, curr_loss)

        # Every 2000 Iteration, generate "n" characters thanks to sample() to check if the model is learning properly
        if j % 2000 == 0:
            
            print('Iteration: %d, Loss: %f' % (j, loss) + '\n')
            
            # The number of dinosaur names to print
            seed = 0
            for name in range(dino_names):
                
                # Sample indices and print them
                sampled_indices = sample(parameters, char_to_ix, seed)
                print_sample(sampled_indices, ix_to_char)
                
                seed += 1  # To get the same result for grading purposed, increment the seed by one. 
      
            print('\n')
        
    return parameters
```

Run the following cell, you should observe your model outputting random-looking characters at the first iteration. After a few thousand iterations, your model should learn to generate reasonable-looking names. 


```python
parameters = model(data, ix_to_char, char_to_ix)
```

    Iteration: 0, Loss: 23.087336
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Nkzxwtdmfqoeyhsqwasjkjvu
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Kneb
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Kzxwtdmfqoeyhsqwasjkjvu
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Neb
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Zxwtdmfqoeyhsqwasjkjvu
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Eb
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Xwtdmfqoeyhsqwasjkjvu
    
    
    Iteration: 2000, Loss: 27.884160
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Liusskeomnolxeros
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Hmdaairus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Hytroligoraurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Lecalosapaus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Xusicikoraurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Abalpsamantisaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Tpraneronxeros
    
    
    Iteration: 4000, Loss: 25.901815
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Mivrosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Inee
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Ivtroplisaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Mbaaisaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Wusichisaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Cabaselachus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Toraperlethosdarenitochusthiamamumamaon
    
    
    Iteration: 6000, Loss: 24.608779
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Onwusceomosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Lieeaerosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Lxussaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Oma
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Xusteonosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Eeahosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Toreonosaurus
    
    
    Iteration: 8000, Loss: 24.070350
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Onxusichepriuon
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Kilabersaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Lutrodon
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Omaaerosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Xutrcheps
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Edaksoje
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Trodiktonus
    
    
    Iteration: 10000, Loss: 23.844446
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Onyusaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Klecalosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Lustodon
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Ola
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Xusodonia
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Eeaeosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Troceosaurus
    
    
    Iteration: 12000, Loss: 23.291971
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Onyxosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Kica
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Lustrepiosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Olaagrraiansaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Yuspangosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Eealosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Trognesaurus
    
    
    Iteration: 14000, Loss: 23.382339
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Meutromodromurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Inda
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Iutroinatorsaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Maca
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Yusteratoptititan
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Ca
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Troclosaurus
    
    
    Iteration: 16000, Loss: 23.259291
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Meustomia
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Indaadps
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Justolongchudosatrus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Macabosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Yuspanhosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Caaerosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Trodon
    
    
    Iteration: 18000, Loss: 22.940799
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Phusaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Meicamitheastosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Mussteratops
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Peg
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Ytrong
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Egaltor
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Trolome
    
    
    Iteration: 20000, Loss: 22.894192
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Meutrodon
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Lledansteh
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Lwuspconyxauosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Macalosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Yusocichugus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Eiagosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Trrangosaurus
    
    
    Iteration: 22000, Loss: 22.851820
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Onustolia
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Midcagosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Mwrrodonnonus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Ola
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Yurodon
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Eiaeptia
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Trodoniohus
    
    
    Iteration: 24000, Loss: 22.700408
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Meutosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Jmacagosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Kurrodon
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Macaistel
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Yuroeleton
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Eiaeror
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Trodonosaurus
    
    
    Iteration: 26000, Loss: 22.736918
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Niutosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Liga
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Lustoingosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Necakroia
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Xrprinhtilus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Eiaestehastes
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Trocilosaurus
    
    
    Iteration: 28000, Loss: 22.595568
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Meutosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Kolaaeus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Kystodonisaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Macahtopadrus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Xtrrararkaumurpasaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Eiaeosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Trodmanolus
    
    
    Iteration: 30000, Loss: 22.609381
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Meutosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Kracakosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Lustodon
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Macaisthachwisaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Wusqandosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Eiacosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Trsatisaurus
    
    
    Iteration: 32000, Loss: 22.251308
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Mausinasaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Incaadropeglsaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Itrosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Macamisaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Wuroenatoraerax
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Ehanosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Trnanclodratosaurus
    
    
    Iteration: 34000, Loss: 22.477910
    
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Mawspichaniaekorocimamroberax
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Inda
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Itrus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Macaesis
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Wrosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Elaeosaurus
    Waa.shape: (50, 50)
    by: (27, 1)
    b: (50, 1)
    Stegngosaurus
    
    
    

## Conclusion

You can see that your algorithm has started to generate plausible dinosaur names towards the end of the training. At first, it was generating random characters, but towards the end you could see dinosaur names with cool endings. Feel free to run the algorithm even longer and play with hyperparameters to see if you can get even better results. Our implemetation generated some really cool names like `maconucon`, `marloralus` and `macingsersaurus`. Your model hopefully also learned that dinosaur names tend to end in `saurus`, `don`, `aura`, `tor`, etc.

If your model generates some non-cool names, don't blame the model entirely--not all actual dinosaur names sound cool. (For example, `dromaeosauroides` is an actual dinosaur name and is in the training set.) But this model should give you a set of candidates from which you can pick the coolest! 

This assignment had used a relatively small dataset, so that you could train an RNN quickly on a CPU. Training a model of the english language requires a much bigger dataset, and usually needs much more computation, and could run for many hours on GPUs. We ran our dinosaur name for quite some time, and so far our favoriate name is the great, undefeatable, and fierce: Mangosaurus!

<img src="images/mangosaurus.jpeg" style="width:250;height:300px;">

## 4 - Writing like Shakespeare

The rest of this notebook is optional and is not graded, but we hope you'll do it anyway since it's quite fun and informative. 

A similar (but more complicated) task is to generate Shakespeare poems. Instead of learning from a dataset of Dinosaur names you can use a collection of Shakespearian poems. Using LSTM cells, you can learn longer term dependencies that span many characters in the text--e.g., where a character appearing somewhere a sequence can influence what should be a different character much much later in ths sequence. These long term dependencies were less important with dinosaur names, since the names were quite short. 


<img src="images/shakespeare.jpg" style="width:500;height:400px;">
<caption><center> Let's become poets! </center></caption>

We have implemented a Shakespeare poem generator with Keras. Run the following cell to load the required packages and models. This may take a few minutes. 


```python
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Model, load_model, Sequential
from keras.layers import Dense, Activation, Dropout, Input, Masking
from keras.layers import LSTM
from keras.utils.data_utils import get_file
from keras.preprocessing.sequence import pad_sequences
from shakespeare_utils import *
import sys
import io
```

    d:\program files\python36\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    

    Loading text data...
    Creating training set...
    number of training examples: 31412
    Vectorizing training set...
    Loading model...
    

To save you some time, we have already trained a model for ~1000 epochs on a collection of Shakespearian poems called [*"The Sonnets"*](shakespeare.txt). 

Let's train the model for one more epoch. When it finishes training for an epoch---this will also take a few minutes---you can run `generate_output`, which will prompt asking you for an input (`<`40 characters). The poem will start with your sentence, and our RNN-Shakespeare will complete the rest of the poem for you! For example, try "Forsooth this maketh no sense " (don't enter the quotation marks). Depending on whether you include the space at the end, your results might also differ--try it both ways, and try other inputs as well. 



```python
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

model.fit(x, y, batch_size=128, epochs=1, callbacks=[print_callback])
```

    Epoch 1/1
    31412/31412 [==============================] - 45s 1ms/step - loss: 2.5432
    




    <keras.callbacks.History at 0x1f40ab380f0>




```python
# Run this cell to try with different inputs without having to re-train the model 
generate_output()
```

    Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: where are  you ? my love.
    
    
    Here is your poem: 
    
    where are  you ? my love.
    so to eve to by monter the time the bid,
    and beautyso hearting foot chalke deand:
    the lopperveh that bace my hister live mied,
    my peeter's berllose briat of wrateling true,
    a bud my ispeles thought i ashaying wited,
    a wend the state's bucince i be peter tingside
    is care on mening beronss, bage my theors,
    on time thou thy srabus cide midh storms now butr, he his witth fassude it tand:
    i me and the

说一句额外的话，看到这生成的诗，我个人感觉，AI ML DL 进步的空间 可创新性 还是那么的大，情感，是诗歌的灵魂，怎么赋予机器以情感，我是很好奇的。

The RNN-Shakespeare model is very similar to the one you have built for dinosaur names. The only major differences are:
- LSTMs instead of the basic RNN to capture longer-range dependencies
- The model is a deeper, stacked LSTM model (2 layer)
- Using Keras instead of python to simplify the code 

If you want to learn more, you can also check out the Keras Team's text generation implementation on GitHub: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py.

Congratulations on finishing this notebook! 

**References**:
- This exercise took inspiration from Andrej Karpathy's implementation: https://gist.github.com/karpathy/d4dee566867f8291f086. To learn more about text generation, also check out Karpathy's [blog post](http://karpathy.github.io/2015/05/21/rnn-effectiveness/).
- For the Shakespearian poem generator, our implementation was based on the implementation of an LSTM text generator by the Keras team: https://github.com/keras-team/keras/blob/master/examples/lstm_text_generation.py 

---
 
**PS: 欢迎扫码关注公众号：「SelfImprovementLab」！专注「深度学习」，「机器学习」，「人工智能」。以及 「早起」，「阅读」，「运动」，「英语 」「其他」不定期建群 打卡互助活动。**

<center><img src="http://upload-images.jianshu.io/upload_images/1157146-cab5ba89dfeeec4b.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"></center>

