
# Assignment | 05-week1 -Improvise a Jazz Solo with an LSTM Network

该系列仅在原课程基础上课后作业部分添加个人学习笔记，如有错误，还请批评指教。- ZJ
    
>[Coursera 课程](https://www.coursera.org/specializations/deep-learning) |[deeplearning.ai](https://www.deeplearning.ai/) |[网易云课堂](https://mooc.study.163.com/smartSpec/detail/1001319001.htm)

 [CSDN]()：
   

---

Welcome to your final programming assignment of this week! In this notebook, you will implement a model that uses an LSTM to generate music. You will even be able to listen to your own music at the end of the assignment. 

**You will learn to:**
- Apply an LSTM to music generation.
- Generate your own jazz music with deep learning.

Please run the following cell to load all the packages required in this assignment. This may take a few minutes. 


```python
from __future__ import print_function
import IPython
import sys
from music21 import *
import numpy as np
from grammar import *
from qa import *
from preprocess import * 
from music_utils import *
from data_utils import *
from keras.models import load_model, Model
from keras.layers import Dense, Activation, Dropout, Input, LSTM, Reshape, Lambda, RepeatVector
from keras.initializers import glorot_uniform
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras import backend as K
```

    d:\program files\python36\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    

## 1 - Problem statement

You would like to create a jazz music piece specially for a friend's birthday. However, you don't know any instruments or music composition. Fortunately, you know deep learning and will solve this problem using an LSTM netwok.  

You will train a network to generate novel jazz solos in a style representative of a body of performed work.

<img src="images/jazz.jpg" style="width:450;height:300px;">


### 1.1 - Dataset

You will train your algorithm on a corpus of Jazz music. Run the cell below to listen to a snippet of the audio from the training set:


```python
IPython.display.Audio('./data/30s_seq.mp3')
```





                <audio controls="controls" >
                 
                    Your browser does not support the audio element.
                </audio>
              



We have taken care of the preprocessing of the musical data to render it in terms of musical "values." You can informally think of each "value" as a note, which comprises a pitch and a duration. For example, if you press down a specific piano key for 0.5 seconds, then you have just played a note. In music theory, a "value" is actually more complicated than this--specifically, it also captures the information needed to play multiple notes at the same time. For example, when playing a music piece, you might press down two piano keys at the same time (playng multiple notes at the same time generates what's called a "chord"和弦). But we don't need to worry about the details of music theory for this assignment. For the purpose of this assignment, all you need to know is that we will obtain a dataset of values, and will learn an RNN model to generate sequences of values. 

Our music generation system will use 78 unique values. Run the following code to load the raw music data and preprocess it into values. This might take a few minutes.


```python
X, Y, n_values, indices_values = load_music_utils()
print('shape of X:', X.shape)
print('number of training examples:', X.shape[0])
print('Tx (length of sequence):', X.shape[1])
print('total # of unique values:', n_values)
print('Shape of Y:', Y.shape)
# 共 60 个训练样本，每个训练样本的 序列长度是 30 ，音符和弦相关的汇集表 共 78 
```

    shape of X: (60, 30, 78)
    number of training examples: 60
    Tx (length of sequence): 30
    total # of unique values: 78
    Shape of Y: (30, 60, 78)
    

You have just loaded the following:

- `X`: This is an (m, $T_x$, 78) dimensional array. We have m training examples, each of which is a snippet of $T_x =30$ musical values. At each time step, the input is one of 78 different possible values, represented as a one-hot vector. Thus for example, X[i,t,:] is a one-hot vector representating the value of the i-th example at time t. 

- `Y`: This is essentially the same as `X`, but shifted one step to the left (to the past). Similar to the dinosaurus assignment, we're interested in the network using the previous values to predict the next value, so our sequence model will try to predict $y^{\langle t \rangle}$ given $x^{\langle 1\rangle}, \ldots, x^{\langle t \rangle}$. However, the data in `Y` is reordered to be dimension $(T_y, m, 78)$, where $T_y = T_x$. This format makes it more convenient to feed to the LSTM later. 

- `n_values`: The number of unique values in this dataset. This should be 78. 

- `indices_values`: python dictionary mapping from 0-77 to musical values.

### 1.2 - Overview of our model

Here is the architecture of the model we will use. This is similar to the Dinosaurus model you had used in the previous notebook, except that in you will be implementing it in Keras. The architecture is as follows: 

<img src="images/music_generation.png" style="width:600;height:400px;">

<!--
<img src="images/djmodel.png" style="width:600;height:400px;">
<br>
<caption><center> **Figure 1**: LSTM model. $X = (x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, ..., x^{\langle T_x \rangle})$ is a window of size $T_x$ scanned over the musical corpus. Each $x^{\langle t \rangle}$ is an index corresponding to a value (ex: "A,0.250,< m2,P-4 >") while $\hat{y}$ is the prediction for the next value  </center></caption>
!--> 

We will be training the model on random snippets of 30 values taken from a much longer piece of music. Thus, we won't bother to set the first input $x^{\langle 1 \rangle} = \vec{0}$, which we had done previously to denote the start of a dinosaur name, since now most of these snippets of audio start somewhere in the middle of a piece of music. We are setting each of the snippts to have the same length $T_x = 30$ to make vectorization easier. 


## 2 - Building the model

In this part you will build and train a model that will learn musical patterns. To do so, you will need to build a model that takes in X of shape $(m, T_x, 78)$ and Y of shape $(T_y, m, 78)$. We will use an LSTM with 64 dimensional hidden states. Lets set `n_a = 64`. 



```python
n_a = 64 
```


Here's how you can create a Keras model with multiple inputs and outputs. If you're building an RNN where even at test time entire input sequence $x^{\langle 1 \rangle}, x^{\langle 2 \rangle}, \ldots, x^{\langle T_x \rangle}$ were *given in advance*, for example if the inputs were words and the output was a label, then Keras has simple built-in functions to build the model. However, for sequence generation, at test time we don't know all the values of $x^{\langle t\rangle}$ in advance; instead we generate them one at a time using $x^{\langle t\rangle} = y^{\langle t-1 \rangle}$. So the code will be a bit more complicated, and you'll need to implement your own for-loop to iterate over the different time steps. 

The function `djmodel()` will call the LSTM layer $T_x$ times using a for-loop, and it is important that all $T_x$ copies have the same weights. I.e., it should not re-initiaiize the weights every time---the $T_x$ steps should have shared weights. The key steps for implementing layers with shareable weights in Keras are: 
1. Define the layer objects (we will use global variables for this).
2. Call these objects when propagating the input.

We have defined the layers objects you need as global variables. Please run the next cell to create them. Please check the Keras documentation to make sure you understand what these layers are: [Reshape()](https://keras.io/layers/core/#reshape), [LSTM()](https://keras.io/layers/recurrent/#lstm), [Dense()](https://keras.io/layers/core/#dense).



```python
reshapor = Reshape((1, 78))                        # Used in Step 2.B of djmodel(), below
LSTM_cell = LSTM(n_a, return_state = True)         # Used in Step 2.C
densor = Dense(n_values, activation='softmax')     # Used in Step 2.D
```

Each of `reshapor`, `LSTM_cell` and `densor` are now layer objects, and you can use them to implement `djmodel()`. In order to propagate a Keras tensor object X through one of these layers, use `layer_object(X)` (or `layer_object([X,Y])` if it requires multiple inputs.). For example, `reshapor(X)` will propagate X through the `Reshape((1,78))` layer defined above.

 
**Exercise**: Implement `djmodel()`. You will need to carry out 2 steps:

1. Create an empty list "outputs" to save the outputs of the LSTM Cell at every time step.
2. Loop for $t \in 1, \ldots, T_x$:

    A. Select the "t"th time-step vector from X. The shape of this selection should be (78,). To do so, create a custom [Lambda](https://keras.io/layers/core/#lambda) layer in Keras by using this line of code:
```    
           x = Lambda(lambda x: X[:,t,:])(X)
``` 
Look over the Keras documentation to figure out what this does. It is creating a "temporary" or "unnamed" function (that's what Lambda functions are) that extracts out the appropriate one-hot vector, and making this function a Keras `Layer` object to apply to `X`. 

    B. Reshape x to be (1,78). You may find the `reshapor()` layer (defined below) helpful.

    C. Run x through one step of LSTM_cell. Remember to initialize the LSTM_cell with the previous step's hidden state $a$ and cell state $c$. Use the following formatting:
```python
a, _, c = LSTM_cell(input_x, initial_state=[previous hidden state, previous cell state])
```

    D. Propagate the LSTM's output activation value through a dense+softmax layer using `densor`. 
    
    E. Append the predicted value to the list of "outputs"
 



```python
# GRADED FUNCTION: djmodel

def djmodel(Tx, n_a, n_values):
    """
    Implement the model
    
    Arguments:
    Tx -- length of the sequence in a corpus
    n_a -- the number of activations used in our model
    n_values -- number of unique values in the music data 
    
    Returns:
    model -- a keras model with the 
    """
    
    # Define the input of your model with a shape  X --(m, Tx, n_values)  n_values = 78 
    X = Input(shape=(Tx, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM  隐藏状态 ： n_a = 64  
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    
    
    ### START CODE HERE ### 
    # Step 1: Create empty list to append the outputs while you iterate (≈1 line)
    outputs = []
    
    # Step 2: Loop
    for t in range(Tx):
        
        # Step 2.A: select the "t"th time step vector from X. (m, Tx, n_values)
        x = Lambda(lambda x: X[:, t, :])(X)
        # Step 2.B: Use reshapor to reshape x to be (1, n_values) (≈1 line)
        x = reshapor(x)
        
        # Step 2.C: Perform one step of the LSTM_cell
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        # Step 2.D: Apply densor to the hidden state output of LSTM_Cell
        out = densor(a)
        # Step 2.E: add the output to "outputs"
        outputs.append(out)
        
    # Step 3: Create model instance
    model = Model(input=[X, a0, c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return model
```

Run the following cell to define your model. We will use `Tx=30`, `n_a=64` (the dimension of the LSTM activations  LSTM 激活函数的维度), and `n_values=78`. This cell may take a few seconds to run. 


```python
model = djmodel(Tx = 30 , n_a = 64, n_values = 78)
```

    C:\Users\qhtf\AppData\Roaming\Python\Python36\site-packages\ipykernel_launcher.py:44: UserWarning: Update your `Model` call to the Keras 2 API: `Model(outputs=[<tf.Tenso..., inputs=[<tf.Tenso...)`
    

You now need to compile your model to be trained. We will Adam and a categorical cross-entropy loss.


```python
opt = Adam(lr=0.01, beta_1=0.9, beta_2=0.999, decay=0.01) # 超参数 learning_rate 学习率 ，beta_1, beta_2 的设定 decay:Learning rate decay over each update.

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

```

Finally, lets initialize `a0` and `c0` for the LSTM's initial state to be zero. 


```python
m = 60
a0 = np.zeros((m, n_a))
c0 = np.zeros((m, n_a))
```

Lets now fit the model! We will turn `Y` to a list before doing so, since the cost function expects `Y` to be provided in this format (one list item per time-step). So `list(Y)` is a list with 30 items, where each of the list items is of shape (60,78). Lets train for 100 epochs. This will take a few minutes. 




```python
model.fit([X, a0, c0], list(Y), epochs=100)

```

    Epoch 1/100
    60/60 [==============================] - 19s 316ms/step - loss: 125.9216 - dense_1_loss_1: 4.3546 - dense_1_loss_2: 4.3476 - dense_1_loss_3: 4.3439 - dense_1_loss_4: 4.3483 - dense_1_loss_5: 4.3439 - dense_1_loss_6: 4.3465 - dense_1_loss_7: 4.3428 - dense_1_loss_8: 4.3433 - dense_1_loss_9: 4.3390 - dense_1_loss_10: 4.3417 - dense_1_loss_11: 4.3352 - dense_1_loss_12: 4.3507 - dense_1_loss_13: 4.3469 - dense_1_loss_14: 4.3361 - dense_1_loss_15: 4.3412 - dense_1_loss_16: 4.3411 - dense_1_loss_17: 4.3493 - dense_1_loss_18: 4.3379 - dense_1_loss_19: 4.3318 - dense_1_loss_20: 4.3431 - dense_1_loss_21: 4.3416 - dense_1_loss_22: 4.3405 - dense_1_loss_23: 4.3349 - dense_1_loss_24: 4.3379 - dense_1_loss_25: 4.3429 - dense_1_loss_26: 4.3357 - dense_1_loss_27: 4.3455 - dense_1_loss_28: 4.3347 - dense_1_loss_29: 4.3432 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.0500 - dense_1_acc_2: 0.0333 - dense_1_acc_3: 0.0667 - dense_1_acc_4: 0.0333 - dense_1_acc_5: 0.0000e+00 - dense_1_acc_6: 0.0167 - dense_1_acc_7: 0.0667 - dense_1_acc_8: 0.0500 - dense_1_acc_9: 0.0667 - dense_1_acc_10: 0.0500 - dense_1_acc_11: 0.1000 - dense_1_acc_12: 0.0167 - dense_1_acc_13: 0.0167 - dense_1_acc_14: 0.0667 - dense_1_acc_15: 0.0000e+00 - dense_1_acc_16: 0.0500 - dense_1_acc_17: 0.0667 - dense_1_acc_18: 0.0333 - dense_1_acc_19: 0.1167 - dense_1_acc_20: 0.0000e+00 - dense_1_acc_21: 0.0500 - dense_1_acc_22: 0.0500 - dense_1_acc_23: 0.1167 - dense_1_acc_24: 0.1167 - dense_1_acc_25: 0.0333 - dense_1_acc_26: 0.0500 - dense_1_acc_27: 0.0500 - dense_1_acc_28: 0.0167 - dense_1_acc_29: 0.0667 - dense_1_acc_30: 0.0000e+00                                                                                  
    Epoch 2/100
    60/60 [==============================] - 0s 2ms/step - loss: 122.1951 - dense_1_loss_1: 4.3314 - dense_1_loss_2: 4.3034 - dense_1_loss_3: 4.2770 - dense_1_loss_4: 4.2792 - dense_1_loss_5: 4.2472 - dense_1_loss_6: 4.2584 - dense_1_loss_7: 4.2406 - dense_1_loss_8: 4.2140 - dense_1_loss_9: 4.2278 - dense_1_loss_10: 4.2043 - dense_1_loss_11: 4.1957 - dense_1_loss_12: 4.2406 - dense_1_loss_13: 4.2156 - dense_1_loss_14: 4.1940 - dense_1_loss_15: 4.1784 - dense_1_loss_16: 4.1875 - dense_1_loss_17: 4.1964 - dense_1_loss_18: 4.1895 - dense_1_loss_19: 4.1513 - dense_1_loss_20: 4.2047 - dense_1_loss_21: 4.2089 - dense_1_loss_22: 4.1584 - dense_1_loss_23: 4.1804 - dense_1_loss_24: 4.1975 - dense_1_loss_25: 4.2234 - dense_1_loss_26: 4.1456 - dense_1_loss_27: 4.1687 - dense_1_loss_28: 4.1610 - dense_1_loss_29: 4.2145 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.0667 - dense_1_acc_2: 0.1500 - dense_1_acc_3: 0.2167 - dense_1_acc_4: 0.1833 - dense_1_acc_5: 0.2833 - dense_1_acc_6: 0.1333 - dense_1_acc_7: 0.1833 - dense_1_acc_8: 0.2333 - dense_1_acc_9: 0.1667 - dense_1_acc_10: 0.2500 - dense_1_acc_11: 0.2167 - dense_1_acc_12: 0.1500 - dense_1_acc_13: 0.1833 - dense_1_acc_14: 0.1833 - dense_1_acc_15: 0.2167 - dense_1_acc_16: 0.2000 - dense_1_acc_17: 0.1333 - dense_1_acc_18: 0.1500 - dense_1_acc_19: 0.1667 - dense_1_acc_20: 0.0667 - dense_1_acc_21: 0.1333 - dense_1_acc_22: 0.1167 - dense_1_acc_23: 0.1167 - dense_1_acc_24: 0.1167 - dense_1_acc_25: 0.0833 - dense_1_acc_26: 0.1833 - dense_1_acc_27: 0.1167 - dense_1_acc_28: 0.1667 - dense_1_acc_29: 0.1167 - dense_1_acc_30: 0.0000e+00
    Epoch 3/100
    60/60 [==============================] - 0s 1ms/step - loss: 115.9358 - dense_1_loss_1: 4.3081 - dense_1_loss_2: 4.2526 - dense_1_loss_3: 4.1884 - dense_1_loss_4: 4.1732 - dense_1_loss_5: 4.1031 - dense_1_loss_6: 4.1317 - dense_1_loss_7: 4.0619 - dense_1_loss_8: 3.9717 - dense_1_loss_9: 3.9622 - dense_1_loss_10: 3.8240 - dense_1_loss_11: 3.8217 - dense_1_loss_12: 4.1020 - dense_1_loss_13: 3.9293 - dense_1_loss_14: 3.8554 - dense_1_loss_15: 3.9025 - dense_1_loss_16: 3.9569 - dense_1_loss_17: 4.0114 - dense_1_loss_18: 4.0399 - dense_1_loss_19: 3.7482 - dense_1_loss_20: 4.0156 - dense_1_loss_21: 4.1079 - dense_1_loss_22: 3.9015 - dense_1_loss_23: 3.8389 - dense_1_loss_24: 3.8796 - dense_1_loss_25: 4.1367 - dense_1_loss_26: 3.7337 - dense_1_loss_27: 3.9134 - dense_1_loss_28: 3.8858 - dense_1_loss_29: 4.1782 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.0667 - dense_1_acc_2: 0.1500 - dense_1_acc_3: 0.2167 - dense_1_acc_4: 0.2000 - dense_1_acc_5: 0.3333 - dense_1_acc_6: 0.1167 - dense_1_acc_7: 0.1500 - dense_1_acc_8: 0.2667 - dense_1_acc_9: 0.0833 - dense_1_acc_10: 0.1000 - dense_1_acc_11: 0.1000 - dense_1_acc_12: 0.0833 - dense_1_acc_13: 0.1000 - dense_1_acc_14: 0.0833 - dense_1_acc_15: 0.0833 - dense_1_acc_16: 0.0500 - dense_1_acc_17: 0.0500 - dense_1_acc_18: 0.1167 - dense_1_acc_19: 0.0667 - dense_1_acc_20: 0.0000e+00 - dense_1_acc_21: 0.0500 - dense_1_acc_22: 0.1333 - dense_1_acc_23: 0.1000 - dense_1_acc_24: 0.0500 - dense_1_acc_25: 0.0500 - dense_1_acc_26: 0.0667 - dense_1_acc_27: 0.0833 - dense_1_acc_28: 0.0833 - dense_1_acc_29: 0.0167 - dense_1_acc_30: 0.0000e+00
    Epoch 4/100
    60/60 [==============================] - 0s 1ms/step - loss: 113.1482 - dense_1_loss_1: 4.2857 - dense_1_loss_2: 4.2005 - dense_1_loss_3: 4.0911 - dense_1_loss_4: 4.0643 - dense_1_loss_5: 3.9361 - dense_1_loss_6: 3.9924 - dense_1_loss_7: 3.8936 - dense_1_loss_8: 3.6965 - dense_1_loss_9: 3.8225 - dense_1_loss_10: 3.6392 - dense_1_loss_11: 3.7633 - dense_1_loss_12: 4.1325 - dense_1_loss_13: 3.8519 - dense_1_loss_14: 3.7446 - dense_1_loss_15: 3.8381 - dense_1_loss_16: 3.8466 - dense_1_loss_17: 3.9256 - dense_1_loss_18: 3.9667 - dense_1_loss_19: 3.7844 - dense_1_loss_20: 3.9776 - dense_1_loss_21: 4.0048 - dense_1_loss_22: 3.9360 - dense_1_loss_23: 3.8247 - dense_1_loss_24: 3.7539 - dense_1_loss_25: 3.9945 - dense_1_loss_26: 3.5672 - dense_1_loss_27: 3.6829 - dense_1_loss_28: 3.8601 - dense_1_loss_29: 4.0709 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.0667 - dense_1_acc_2: 0.1667 - dense_1_acc_3: 0.2000 - dense_1_acc_4: 0.1500 - dense_1_acc_5: 0.2833 - dense_1_acc_6: 0.1167 - dense_1_acc_7: 0.1167 - dense_1_acc_8: 0.1833 - dense_1_acc_9: 0.1333 - dense_1_acc_10: 0.1500 - dense_1_acc_11: 0.0833 - dense_1_acc_12: 0.0667 - dense_1_acc_13: 0.0833 - dense_1_acc_14: 0.1000 - dense_1_acc_15: 0.0833 - dense_1_acc_16: 0.0667 - dense_1_acc_17: 0.1833 - dense_1_acc_18: 0.0500 - dense_1_acc_19: 0.0667 - dense_1_acc_20: 0.1167 - dense_1_acc_21: 0.0833 - dense_1_acc_22: 0.0167 - dense_1_acc_23: 0.0333 - dense_1_acc_24: 0.0667 - dense_1_acc_25: 0.0500 - dense_1_acc_26: 0.1333 - dense_1_acc_27: 0.0833 - dense_1_acc_28: 0.0500 - dense_1_acc_29: 0.1000 - dense_1_acc_30: 0.0000e+00
    Epoch 5/100
   .
   .
   .
   .
   .
    60/60 [==============================] - 0s 1ms/step - loss: 6.2689 - dense_1_loss_1: 3.7845 - dense_1_loss_2: 1.2346 - dense_1_loss_3: 0.3671 - dense_1_loss_4: 0.0967 - dense_1_loss_5: 0.0647 - dense_1_loss_6: 0.0501 - dense_1_loss_7: 0.0355 - dense_1_loss_8: 0.0351 - dense_1_loss_9: 0.0329 - dense_1_loss_10: 0.0281 - dense_1_loss_11: 0.0288 - dense_1_loss_12: 0.0283 - dense_1_loss_13: 0.0255 - dense_1_loss_14: 0.0270 - dense_1_loss_15: 0.0291 - dense_1_loss_16: 0.0289 - dense_1_loss_17: 0.0265 - dense_1_loss_18: 0.0271 - dense_1_loss_19: 0.0261 - dense_1_loss_20: 0.0293 - dense_1_loss_21: 0.0304 - dense_1_loss_22: 0.0276 - dense_1_loss_23: 0.0270 - dense_1_loss_24: 0.0256 - dense_1_loss_25: 0.0278 - dense_1_loss_26: 0.0265 - dense_1_loss_27: 0.0287 - dense_1_loss_28: 0.0336 - dense_1_loss_29: 0.0359 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6667 - dense_1_acc_3: 0.9167 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 87/100
    60/60 [==============================] - 0s 1ms/step - loss: 6.2256 - dense_1_loss_1: 3.7809 - dense_1_loss_2: 1.2235 - dense_1_loss_3: 0.3603 - dense_1_loss_4: 0.0947 - dense_1_loss_5: 0.0633 - dense_1_loss_6: 0.0489 - dense_1_loss_7: 0.0347 - dense_1_loss_8: 0.0342 - dense_1_loss_9: 0.0321 - dense_1_loss_10: 0.0274 - dense_1_loss_11: 0.0279 - dense_1_loss_12: 0.0276 - dense_1_loss_13: 0.0249 - dense_1_loss_14: 0.0262 - dense_1_loss_15: 0.0282 - dense_1_loss_16: 0.0282 - dense_1_loss_17: 0.0259 - dense_1_loss_18: 0.0264 - dense_1_loss_19: 0.0254 - dense_1_loss_20: 0.0285 - dense_1_loss_21: 0.0296 - dense_1_loss_22: 0.0269 - dense_1_loss_23: 0.0262 - dense_1_loss_24: 0.0248 - dense_1_loss_25: 0.0270 - dense_1_loss_26: 0.0255 - dense_1_loss_27: 0.0283 - dense_1_loss_28: 0.0324 - dense_1_loss_29: 0.0355 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6667 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 88/100
    60/60 [==============================] - 0s 1ms/step - loss: 6.1823 - dense_1_loss_1: 3.7776 - dense_1_loss_2: 1.2114 - dense_1_loss_3: 0.3527 - dense_1_loss_4: 0.0929 - dense_1_loss_5: 0.0619 - dense_1_loss_6: 0.0477 - dense_1_loss_7: 0.0338 - dense_1_loss_8: 0.0334 - dense_1_loss_9: 0.0314 - dense_1_loss_10: 0.0268 - dense_1_loss_11: 0.0272 - dense_1_loss_12: 0.0271 - dense_1_loss_13: 0.0244 - dense_1_loss_14: 0.0256 - dense_1_loss_15: 0.0275 - dense_1_loss_16: 0.0276 - dense_1_loss_17: 0.0253 - dense_1_loss_18: 0.0257 - dense_1_loss_19: 0.0247 - dense_1_loss_20: 0.0279 - dense_1_loss_21: 0.0289 - dense_1_loss_22: 0.0262 - dense_1_loss_23: 0.0255 - dense_1_loss_24: 0.0244 - dense_1_loss_25: 0.0262 - dense_1_loss_26: 0.0248 - dense_1_loss_27: 0.0280 - dense_1_loss_28: 0.0313 - dense_1_loss_29: 0.0346 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6667 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 89/100
    60/60 [==============================] - 0s 1ms/step - loss: 6.1403 - dense_1_loss_1: 3.7743 - dense_1_loss_2: 1.1999 - dense_1_loss_3: 0.3456 - dense_1_loss_4: 0.0906 - dense_1_loss_5: 0.0605 - dense_1_loss_6: 0.0464 - dense_1_loss_7: 0.0331 - dense_1_loss_8: 0.0326 - dense_1_loss_9: 0.0305 - dense_1_loss_10: 0.0262 - dense_1_loss_11: 0.0265 - dense_1_loss_12: 0.0264 - dense_1_loss_13: 0.0238 - dense_1_loss_14: 0.0250 - dense_1_loss_15: 0.0269 - dense_1_loss_16: 0.0269 - dense_1_loss_17: 0.0248 - dense_1_loss_18: 0.0250 - dense_1_loss_19: 0.0241 - dense_1_loss_20: 0.0272 - dense_1_loss_21: 0.0282 - dense_1_loss_22: 0.0257 - dense_1_loss_23: 0.0249 - dense_1_loss_24: 0.0238 - dense_1_loss_25: 0.0257 - dense_1_loss_26: 0.0242 - dense_1_loss_27: 0.0272 - dense_1_loss_28: 0.0308 - dense_1_loss_29: 0.0335 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6667 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 90/100
    60/60 [==============================] - 0s 1ms/step - loss: 6.1011 - dense_1_loss_1: 3.7711 - dense_1_loss_2: 1.1893 - dense_1_loss_3: 0.3391 - dense_1_loss_4: 0.0885 - dense_1_loss_5: 0.0592 - dense_1_loss_6: 0.0454 - dense_1_loss_7: 0.0324 - dense_1_loss_8: 0.0319 - dense_1_loss_9: 0.0298 - dense_1_loss_10: 0.0256 - dense_1_loss_11: 0.0260 - dense_1_loss_12: 0.0258 - dense_1_loss_13: 0.0231 - dense_1_loss_14: 0.0244 - dense_1_loss_15: 0.0265 - dense_1_loss_16: 0.0262 - dense_1_loss_17: 0.0240 - dense_1_loss_18: 0.0245 - dense_1_loss_19: 0.0236 - dense_1_loss_20: 0.0265 - dense_1_loss_21: 0.0275 - dense_1_loss_22: 0.0251 - dense_1_loss_23: 0.0244 - dense_1_loss_24: 0.0232 - dense_1_loss_25: 0.0252 - dense_1_loss_26: 0.0239 - dense_1_loss_27: 0.0260 - dense_1_loss_28: 0.0304 - dense_1_loss_29: 0.0323 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6667 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 91/100
    60/60 [==============================] - 0s 1ms/step - loss: 6.0635 - dense_1_loss_1: 3.7681 - dense_1_loss_2: 1.1783 - dense_1_loss_3: 0.3334 - dense_1_loss_4: 0.0866 - dense_1_loss_5: 0.0579 - dense_1_loss_6: 0.0443 - dense_1_loss_7: 0.0318 - dense_1_loss_8: 0.0312 - dense_1_loss_9: 0.0291 - dense_1_loss_10: 0.0250 - dense_1_loss_11: 0.0255 - dense_1_loss_12: 0.0252 - dense_1_loss_13: 0.0226 - dense_1_loss_14: 0.0239 - dense_1_loss_15: 0.0260 - dense_1_loss_16: 0.0256 - dense_1_loss_17: 0.0235 - dense_1_loss_18: 0.0240 - dense_1_loss_19: 0.0231 - dense_1_loss_20: 0.0259 - dense_1_loss_21: 0.0269 - dense_1_loss_22: 0.0245 - dense_1_loss_23: 0.0238 - dense_1_loss_24: 0.0227 - dense_1_loss_25: 0.0247 - dense_1_loss_26: 0.0234 - dense_1_loss_27: 0.0253 - dense_1_loss_28: 0.0298 - dense_1_loss_29: 0.0315 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6667 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 92/100
    60/60 [==============================] - 0s 1ms/step - loss: 6.0243 - dense_1_loss_1: 3.7647 - dense_1_loss_2: 1.1673 - dense_1_loss_3: 0.3269 - dense_1_loss_4: 0.0846 - dense_1_loss_5: 0.0567 - dense_1_loss_6: 0.0433 - dense_1_loss_7: 0.0310 - dense_1_loss_8: 0.0305 - dense_1_loss_9: 0.0285 - dense_1_loss_10: 0.0245 - dense_1_loss_11: 0.0249 - dense_1_loss_12: 0.0246 - dense_1_loss_13: 0.0221 - dense_1_loss_14: 0.0233 - dense_1_loss_15: 0.0253 - dense_1_loss_16: 0.0250 - dense_1_loss_17: 0.0229 - dense_1_loss_18: 0.0235 - dense_1_loss_19: 0.0226 - dense_1_loss_20: 0.0253 - dense_1_loss_21: 0.0263 - dense_1_loss_22: 0.0240 - dense_1_loss_23: 0.0233 - dense_1_loss_24: 0.0223 - dense_1_loss_25: 0.0240 - dense_1_loss_26: 0.0227 - dense_1_loss_27: 0.0250 - dense_1_loss_28: 0.0288 - dense_1_loss_29: 0.0305 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6667 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 93/100
    60/60 [==============================] - 0s 1ms/step - loss: 5.9890 - dense_1_loss_1: 3.7615 - dense_1_loss_2: 1.1573 - dense_1_loss_3: 0.3211 - dense_1_loss_4: 0.0832 - dense_1_loss_5: 0.0555 - dense_1_loss_6: 0.0424 - dense_1_loss_7: 0.0302 - dense_1_loss_8: 0.0298 - dense_1_loss_9: 0.0279 - dense_1_loss_10: 0.0239 - dense_1_loss_11: 0.0243 - dense_1_loss_12: 0.0241 - dense_1_loss_13: 0.0216 - dense_1_loss_14: 0.0227 - dense_1_loss_15: 0.0246 - dense_1_loss_16: 0.0245 - dense_1_loss_17: 0.0225 - dense_1_loss_18: 0.0229 - dense_1_loss_19: 0.0220 - dense_1_loss_20: 0.0248 - dense_1_loss_21: 0.0257 - dense_1_loss_22: 0.0234 - dense_1_loss_23: 0.0228 - dense_1_loss_24: 0.0219 - dense_1_loss_25: 0.0234 - dense_1_loss_26: 0.0220 - dense_1_loss_27: 0.0249 - dense_1_loss_28: 0.0279 - dense_1_loss_29: 0.0301 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6833 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 94/100
    60/60 [==============================] - 0s 1ms/step - loss: 5.9543 - dense_1_loss_1: 3.7587 - dense_1_loss_2: 1.1472 - dense_1_loss_3: 0.3150 - dense_1_loss_4: 0.0817 - dense_1_loss_5: 0.0544 - dense_1_loss_6: 0.0415 - dense_1_loss_7: 0.0296 - dense_1_loss_8: 0.0291 - dense_1_loss_9: 0.0273 - dense_1_loss_10: 0.0235 - dense_1_loss_11: 0.0238 - dense_1_loss_12: 0.0236 - dense_1_loss_13: 0.0212 - dense_1_loss_14: 0.0222 - dense_1_loss_15: 0.0241 - dense_1_loss_16: 0.0240 - dense_1_loss_17: 0.0220 - dense_1_loss_18: 0.0224 - dense_1_loss_19: 0.0215 - dense_1_loss_20: 0.0243 - dense_1_loss_21: 0.0251 - dense_1_loss_22: 0.0229 - dense_1_loss_23: 0.0223 - dense_1_loss_24: 0.0213 - dense_1_loss_25: 0.0228 - dense_1_loss_26: 0.0216 - dense_1_loss_27: 0.0242 - dense_1_loss_28: 0.0274 - dense_1_loss_29: 0.0295 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6833 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 95/100
    60/60 [==============================] - 0s 1ms/step - loss: 5.9199 - dense_1_loss_1: 3.7554 - dense_1_loss_2: 1.1372 - dense_1_loss_3: 0.3089 - dense_1_loss_4: 0.0802 - dense_1_loss_5: 0.0533 - dense_1_loss_6: 0.0405 - dense_1_loss_7: 0.0291 - dense_1_loss_8: 0.0286 - dense_1_loss_9: 0.0267 - dense_1_loss_10: 0.0230 - dense_1_loss_11: 0.0232 - dense_1_loss_12: 0.0231 - dense_1_loss_13: 0.0208 - dense_1_loss_14: 0.0218 - dense_1_loss_15: 0.0236 - dense_1_loss_16: 0.0235 - dense_1_loss_17: 0.0215 - dense_1_loss_18: 0.0220 - dense_1_loss_19: 0.0210 - dense_1_loss_20: 0.0237 - dense_1_loss_21: 0.0246 - dense_1_loss_22: 0.0224 - dense_1_loss_23: 0.0218 - dense_1_loss_24: 0.0209 - dense_1_loss_25: 0.0224 - dense_1_loss_26: 0.0213 - dense_1_loss_27: 0.0235 - dense_1_loss_28: 0.0269 - dense_1_loss_29: 0.0289 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6833 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 96/100
    60/60 [==============================] - 0s 1ms/step - loss: 5.8865 - dense_1_loss_1: 3.7523 - dense_1_loss_2: 1.1271 - dense_1_loss_3: 0.3035 - dense_1_loss_4: 0.0786 - dense_1_loss_5: 0.0522 - dense_1_loss_6: 0.0398 - dense_1_loss_7: 0.0285 - dense_1_loss_8: 0.0280 - dense_1_loss_9: 0.0262 - dense_1_loss_10: 0.0225 - dense_1_loss_11: 0.0228 - dense_1_loss_12: 0.0226 - dense_1_loss_13: 0.0203 - dense_1_loss_14: 0.0214 - dense_1_loss_15: 0.0232 - dense_1_loss_16: 0.0229 - dense_1_loss_17: 0.0211 - dense_1_loss_18: 0.0215 - dense_1_loss_19: 0.0206 - dense_1_loss_20: 0.0232 - dense_1_loss_21: 0.0241 - dense_1_loss_22: 0.0219 - dense_1_loss_23: 0.0214 - dense_1_loss_24: 0.0205 - dense_1_loss_25: 0.0220 - dense_1_loss_26: 0.0208 - dense_1_loss_27: 0.0228 - dense_1_loss_28: 0.0263 - dense_1_loss_29: 0.0281 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6833 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 97/100
    60/60 [==============================] - 0s 1ms/step - loss: 5.8536 - dense_1_loss_1: 3.7492 - dense_1_loss_2: 1.1172 - dense_1_loss_3: 0.2979 - dense_1_loss_4: 0.0772 - dense_1_loss_5: 0.0511 - dense_1_loss_6: 0.0390 - dense_1_loss_7: 0.0279 - dense_1_loss_8: 0.0274 - dense_1_loss_9: 0.0257 - dense_1_loss_10: 0.0221 - dense_1_loss_11: 0.0223 - dense_1_loss_12: 0.0222 - dense_1_loss_13: 0.0198 - dense_1_loss_14: 0.0209 - dense_1_loss_15: 0.0228 - dense_1_loss_16: 0.0224 - dense_1_loss_17: 0.0206 - dense_1_loss_18: 0.0211 - dense_1_loss_19: 0.0202 - dense_1_loss_20: 0.0227 - dense_1_loss_21: 0.0236 - dense_1_loss_22: 0.0215 - dense_1_loss_23: 0.0209 - dense_1_loss_24: 0.0201 - dense_1_loss_25: 0.0216 - dense_1_loss_26: 0.0205 - dense_1_loss_27: 0.0223 - dense_1_loss_28: 0.0258 - dense_1_loss_29: 0.0274 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6833 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 98/100
    60/60 [==============================] - 0s 1ms/step - loss: 5.8240 - dense_1_loss_1: 3.7461 - dense_1_loss_2: 1.1087 - dense_1_loss_3: 0.2933 - dense_1_loss_4: 0.0757 - dense_1_loss_5: 0.0501 - dense_1_loss_6: 0.0383 - dense_1_loss_7: 0.0274 - dense_1_loss_8: 0.0268 - dense_1_loss_9: 0.0252 - dense_1_loss_10: 0.0216 - dense_1_loss_11: 0.0219 - dense_1_loss_12: 0.0217 - dense_1_loss_13: 0.0194 - dense_1_loss_14: 0.0205 - dense_1_loss_15: 0.0223 - dense_1_loss_16: 0.0220 - dense_1_loss_17: 0.0202 - dense_1_loss_18: 0.0207 - dense_1_loss_19: 0.0198 - dense_1_loss_20: 0.0223 - dense_1_loss_21: 0.0232 - dense_1_loss_22: 0.0211 - dense_1_loss_23: 0.0205 - dense_1_loss_24: 0.0197 - dense_1_loss_25: 0.0211 - dense_1_loss_26: 0.0201 - dense_1_loss_27: 0.0218 - dense_1_loss_28: 0.0251 - dense_1_loss_29: 0.0270 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6833 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 99/100
    60/60 [==============================] - 0s 1ms/step - loss: 5.7939 - dense_1_loss_1: 3.7431 - dense_1_loss_2: 1.0993 - dense_1_loss_3: 0.2887 - dense_1_loss_4: 0.0745 - dense_1_loss_5: 0.0492 - dense_1_loss_6: 0.0376 - dense_1_loss_7: 0.0269 - dense_1_loss_8: 0.0264 - dense_1_loss_9: 0.0247 - dense_1_loss_10: 0.0213 - dense_1_loss_11: 0.0214 - dense_1_loss_12: 0.0213 - dense_1_loss_13: 0.0191 - dense_1_loss_14: 0.0201 - dense_1_loss_15: 0.0218 - dense_1_loss_16: 0.0216 - dense_1_loss_17: 0.0198 - dense_1_loss_18: 0.0203 - dense_1_loss_19: 0.0194 - dense_1_loss_20: 0.0219 - dense_1_loss_21: 0.0228 - dense_1_loss_22: 0.0207 - dense_1_loss_23: 0.0201 - dense_1_loss_24: 0.0193 - dense_1_loss_25: 0.0207 - dense_1_loss_26: 0.0197 - dense_1_loss_27: 0.0214 - dense_1_loss_28: 0.0245 - dense_1_loss_29: 0.0264 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6833 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    Epoch 100/100
    60/60 [==============================] - 0s 1ms/step - loss: 5.7645 - dense_1_loss_1: 3.7402 - dense_1_loss_2: 1.0904 - dense_1_loss_3: 0.2835 - dense_1_loss_4: 0.0733 - dense_1_loss_5: 0.0484 - dense_1_loss_6: 0.0368 - dense_1_loss_7: 0.0264 - dense_1_loss_8: 0.0258 - dense_1_loss_9: 0.0243 - dense_1_loss_10: 0.0209 - dense_1_loss_11: 0.0209 - dense_1_loss_12: 0.0209 - dense_1_loss_13: 0.0187 - dense_1_loss_14: 0.0197 - dense_1_loss_15: 0.0214 - dense_1_loss_16: 0.0212 - dense_1_loss_17: 0.0195 - dense_1_loss_18: 0.0199 - dense_1_loss_19: 0.0190 - dense_1_loss_20: 0.0215 - dense_1_loss_21: 0.0223 - dense_1_loss_22: 0.0203 - dense_1_loss_23: 0.0197 - dense_1_loss_24: 0.0190 - dense_1_loss_25: 0.0203 - dense_1_loss_26: 0.0192 - dense_1_loss_27: 0.0212 - dense_1_loss_28: 0.0239 - dense_1_loss_29: 0.0259 - dense_1_loss_30: 0.0000e+00 - dense_1_acc_1: 0.1000 - dense_1_acc_2: 0.6833 - dense_1_acc_3: 0.9333 - dense_1_acc_4: 1.0000 - dense_1_acc_5: 1.0000 - dense_1_acc_6: 1.0000 - dense_1_acc_7: 1.0000 - dense_1_acc_8: 1.0000 - dense_1_acc_9: 1.0000 - dense_1_acc_10: 1.0000 - dense_1_acc_11: 1.0000 - dense_1_acc_12: 1.0000 - dense_1_acc_13: 1.0000 - dense_1_acc_14: 1.0000 - dense_1_acc_15: 1.0000 - dense_1_acc_16: 1.0000 - dense_1_acc_17: 1.0000 - dense_1_acc_18: 1.0000 - dense_1_acc_19: 1.0000 - dense_1_acc_20: 1.0000 - dense_1_acc_21: 1.0000 - dense_1_acc_22: 1.0000 - dense_1_acc_23: 1.0000 - dense_1_acc_24: 1.0000 - dense_1_acc_25: 1.0000 - dense_1_acc_26: 1.0000 - dense_1_acc_27: 1.0000 - dense_1_acc_28: 1.0000 - dense_1_acc_29: 1.0000 - dense_1_acc_30: 0.0333
    




    <keras.callbacks.History at 0x235dc340a20>



You should see the model loss going down. Now that you have trained a model, lets go on the the final section to implement an inference algorithm, and generate some music! 

## 3 - Generating music 音乐生成

You now have a trained model which has learned the patterns of the jazz soloist. Lets now use this model to synthesize new music. 

#### 3.1 - Predicting & Sampling

<img src="images/music_gen.png" style="width:600;height:400px;">

At each step of sampling, you will take as input the activation `a` and cell state `c` from the previous state of the LSTM, forward propagate by one step, and get a new output activation as well as cell state. The new activation `a` can then be used to generate the output, using `densor` as before. 

To start off the model, we will initialize `x0` as well as the LSTM activation and and cell value `a0` and `c0` to be zeros. 


<!-- 
You are about to build a function that will do this inference for you. Your function takes in your previous model and the number of time steps `Ty` that you want to sample. It will return a keras model that would be able to generate sequences for you. Furthermore, the function takes in a dense layer of `78` units and the number of activations. 
!--> 


**Exercise:** Implement the function below to sample a sequence of musical values. Here are some of the key steps you'll need to implement inside the for-loop that generates the $T_y$ output characters: 

Step 2.A: Use `LSTM_Cell`, which inputs the previous step's `c` and `a` to generate the current step's `c` and `a`. 

Step 2.B: Use `densor` (defined previously) to compute a softmax on `a` to get the output for the current step. 

Step 2.C: Save the output you have just generated by appending it to `outputs`.

Step 2.D: Sample x to the be "out"'s one-hot version (the prediction) so that you can pass it to the next LSTM's step.  We have already provided this line of code, which uses a [Lambda](https://keras.io/layers/core/#lambda) function. 
```python
x = Lambda(one_hot)(out) 
```
[Minor technical note: Rather than sampling a value at random according to the probabilities in `out`, this line of code actually chooses the single most likely note at each step using an argmax.]



```python
# GRADED FUNCTION: music_inference_model

def music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_values -- integer, umber of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- integer, number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_values))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs". out.shape = (None, 78) (≈1 line)
        outputs.append(out)
        
        # Step 2.D: Select the next value according to "out", and set "x" to be the one-hot representation of the
        #           selected value, which will be passed as the input to LSTM_cell on the next step. We have provided 
        #           the line of code you need to do this. 
        x = Lambda(one_hot)(out)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    
    ### END CODE HERE ###
    
    return inference_model
```

Run the cell below to define your inference model. This model is hard coded to generate 50 values. 硬编码生成 50 个值


```python
inference_model = music_inference_model(LSTM_cell, densor, n_values = 78, n_a = 64, Ty = 50)
```

Finally, this creates the zero-valued vectors you will use to initialize `x` and the LSTM state variables `a` and `c`. 


```python
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))
```

**Exercise**: Implement `predict_and_sample()`. This function takes many arguments including the inputs [x_initializer, a_initializer, c_initializer]. In order to predict the output corresponding to this input, you will need to carry-out 3 steps:
1. Use your inference model to predict an output given your set of inputs. The output `pred` should be a list of length 20 where each element is a numpy-array of shape ($T_y$, n_values)
2. Convert `pred` into a numpy array of $T_y$ indices. Each index corresponds is computed by taking the `argmax` of an element of the `pred` list. [Hint](https://docs.scipy.org/doc/numpy/reference/generated/numpy.argmax.html).
3. Convert the indices into their one-hot vector representations. [Hint](https://keras.io/utils/#to_categorical).

实现 预测采样 函数 1. 使用模型，预测出结果，并作为输出，2. 输出结果 `pred`  转化为 数组，3.将 indicies 转化为 one -hot 向量


```python
# GRADED FUNCTION: predict_and_sample

def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    ### START CODE HERE ###
    # Step 1: Use your inference model to predict an output sequence given x_initializer, a_initializer and c_initializer.
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities ---argmax返回的是最大数的索引
    indices = np.argmax(np.array(pred), axis=-1)
    # Step 3: Convert indices to one-hot vectors, the shape of the results should be (1, )
    results = to_categorical(indices, num_classes=x_initializer.shape[-1])
    ### END CODE HERE ###
    
    return results, indices
```


```python
results, indices = predict_and_sample(inference_model, x_initializer, a_initializer, c_initializer)
print("np.argmax(results[12]) =", np.argmax(results[12]))
print("np.argmax(results[17]) =", np.argmax(results[17]))
print("list(indices[12:18]) =", list(indices[12:18]))
```

    np.argmax(results[12]) = 29
    np.argmax(results[17]) = 31
    list(indices[12:18]) = [array([29], dtype=int64), array([31], dtype=int64), array([55], dtype=int64), array([8], dtype=int64), array([29], dtype=int64), array([31], dtype=int64)]
    

**Expected Output**: Your results may differ because Keras' results are not completely predictable. However, if you have trained your LSTM_cell with model.fit() for exactly 100 epochs as described above, you should very likely observe a sequence of indices that are not all identical. Moreover, you should observe that: np.argmax(results[12]) is the first element of list(indices[12:18]) and np.argmax(results[17]) is the last element of list(indices[12:18]). 

<table>
    <tr>
        <td>
            **np.argmax(results[12])** =
        </td>
        <td>
        1
        </td>
    </tr>
    <tr>
        <td>
            **np.argmax(results[12])** =
        </td>
        <td>
        42
        </td>
    </tr>
    <tr>
        <td>
            **list(indices[12:18])** =
        </td>
        <td>
            [array([1]), array([42]), array([54]), array([17]), array([1]), array([42])]
        </td>
    </tr>
</table>

#### 3.3 - Generate music 

Finally, you are ready to generate music. Your RNN generates a sequence of values. The following code generates music by first calling your `predict_and_sample()` function. These values are then post-processed into musical chords (meaning that multiple values or notes can be played at the same time). 

Most computational music algorithms use some post-processing because it is difficult to generate music that sounds good without such post-processing. The post-processing does things such as clean up the generated audio by making sure the same sound is not repeated too many times, that two successive notes are not too far from each other in pitch, and so on. One could argue that a lot of these post-processing steps are hacks; also, a lot the music generation literature has also focused on hand-crafting post-processors, and a lot of the output quality depends on the quality of the post-processing and not just the quality of the RNN. But this post-processing does make a huge difference, so lets use it in our implementation as well. 

Lets make some music! 

Run the following cell to generate music and record it into your `out_stream`. This can take a couple of minutes.


```python
out_stream = generate_music(inference_model)
```

    Predicting new values for different set of chords.
    Generated 51 sounds using the predicted values for the set of chords ("1") and after pruning
    Generated 51 sounds using the predicted values for the set of chords ("2") and after pruning
    Generated 51 sounds using the predicted values for the set of chords ("3") and after pruning
    Generated 50 sounds using the predicted values for the set of chords ("4") and after pruning
    Generated 50 sounds using the predicted values for the set of chords ("5") and after pruning
    Your generated music is saved in output/my_music.midi
    

To listen to your music, click File->Open... Then go to "output/" and download "my_music.midi". Either play it on your computer with an application that can read midi files if you have one, or use one of the free online "MIDI to mp3" conversion tools to convert this to mp3.  

As reference, here also is a 30sec audio clip we generated using this algorithm. 


```python
IPython.display.Audio('./data/30s_trained_model.mp3')
```





                <audio controls="controls" >

                    Your browser does not support the audio element.
                </audio>
              



### Congratulations!

You have come to the end of the notebook. 

<font color="blue">
Here's what you should remember:
- A sequence model can be used to generate musical values, which are then post-processed into midi music.   序列模型可用于生成音乐值，然后将其后处理为MIDI音乐。
- Fairly similar models can be used to generate dinosaur names or to generate music, with the major difference being the input fed to the model.   相当相似的模型可用于生成恐龙名称或生成音乐，主要区别在于输入给模型的输入
- In Keras, sequence generation involves defining layers with shared weights, which are then repeated for the different time steps $1, \ldots, T_x$. 在Keras中，序列生成涉及定义具有共享权重的图层，然后针对不同的时间步骤重复这些图层

Congratulations on completing this assignment and generating a jazz solo! 

作业中用到的 utils 中的代码。


```python
'''
data_utils.py
'''

from music_utils import * 
from preprocess import * 
from keras.utils import to_categorical

chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
N_tones = len(set(corpus))
n_a = 64
x_initializer = np.zeros((1, 1, 78))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))

def load_music_utils():
    chords, abstract_grammars = get_musical_data('data/original_metheny.mid')
    corpus, tones, tones_indices, indices_tones = get_corpus_data(abstract_grammars)
    N_tones = len(set(corpus))
    X, Y, N_tones = data_processing(corpus, tones_indices, 60, 30)   
    return (X, Y, N_tones, indices_tones)


def generate_music(inference_model, corpus = corpus, abstract_grammars = abstract_grammars, tones = tones, tones_indices = tones_indices, indices_tones = indices_tones, T_y = 10, max_tries = 1000, diversity = 0.5):
    """
    Generates music using a model trained to learn musical patterns of a jazz soloist. Creates an audio stream
    to save the music and play it.
    
    Arguments:
    model -- Keras model Instance, output of djmodel()
    corpus -- musical corpus, list of 193 tones as strings (ex: 'C,0.333,<P1,d-5>')
    abstract_grammars -- list of grammars, on element can be: 'S,0.250,<m2,P-4> C,0.250,<P4,m-2> A,0.250,<P4,m-2>'
    tones -- set of unique tones, ex: 'A,0.250,<M2,d-4>' is one element of the set.
    tones_indices -- a python dictionary mapping unique tone (ex: A,0.250,< m2,P-4 >) into their corresponding indices (0-77)
    indices_tones -- a python dictionary mapping indices (0-77) into their corresponding unique tone (ex: A,0.250,< m2,P-4 >)
    Tx -- integer, number of time-steps used at training time
    temperature -- scalar value, defines how conservative/creative the model is when generating music
    
    Returns:
    predicted_tones -- python list containing predicted tones
    """
    
    # set up audio stream
    out_stream = stream.Stream()
    
    # Initialize chord variables
    curr_offset = 0.0                                     # variable used to write sounds to the Stream.
    num_chords = int(len(chords) / 3)                     # number of different set of chords
    
    print("Predicting new values for different set of chords.")
    # Loop over all 18 set of chords. At each iteration generate a sequence of tones
    # and use the current chords to convert it into actual sounds 
    for i in range(1, num_chords):
        
        # Retrieve current chord from stream
        curr_chords = stream.Voice()
        
        # Loop over the chords of the current set of chords
        for j in chords[i]:
            # Add chord to the current chords with the adequate offset, no need to understand this
            curr_chords.insert((j.offset % 4), j)
        
        # Generate a sequence of tones using the model
        _, indices = predict_and_sample(inference_model)
        indices = list(indices.squeeze())
        pred = [indices_tones[p] for p in indices]
        
        predicted_tones = 'C,0.25 '
        for k in range(len(pred) - 1):
            predicted_tones += pred[k] + ' ' 
        
        predicted_tones +=  pred[-1]
                
        #### POST PROCESSING OF THE PREDICTED TONES ####
        # We will consider "A" and "X" as "C" tones. It is a common choice.
        predicted_tones = predicted_tones.replace(' A',' C').replace(' X',' C')

        # Pruning #1: smoothing measure
        predicted_tones = prune_grammar(predicted_tones)
        
        # Use predicted tones and current chords to generate sounds
        sounds = unparse_grammar(predicted_tones, curr_chords)

        # Pruning #2: removing repeated and too close together sounds
        sounds = prune_notes(sounds)

        # Quality assurance: clean up sounds
        sounds = clean_up_notes(sounds)

        # Print number of tones/notes in sounds
        print('Generated %s sounds using the predicted values for the set of chords ("%s") and after pruning' % (len([k for k in sounds if isinstance(k, note.Note)]), i))
        
        # Insert sounds into the output stream
        for m in sounds:
            out_stream.insert(curr_offset + m.offset, m)
        for mc in curr_chords:
            out_stream.insert(curr_offset + mc.offset, mc)

        curr_offset += 4.0
        
    # Initialize tempo of the output stream with 130 bit per minute
    out_stream.insert(0.0, tempo.MetronomeMark(number=130))

    # Save audio stream to fine
    mf = midi.translate.streamToMidiFile(out_stream)
    mf.open("output/my_music.midi", 'wb')
    mf.write()
    print("Your generated music is saved in output/my_music.midi")
    mf.close()
    
    # Play the final stream through output (see 'play' lambda function above)
    # play = lambda x: midi.realtime.StreamPlayer(x).play()
    # play(out_stream)
    
    return out_stream


def predict_and_sample(inference_model, x_initializer = x_initializer, a_initializer = a_initializer, 
                       c_initializer = c_initializer):
    """
    Predicts the next value of values using the inference model.
    
    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 78), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel
    Ty -- length of the sequence you'd like to generate.
    
    Returns:
    results -- numpy-array of shape (Ty, 78), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """
    
    ### START CODE HERE ###
    pred = inference_model.predict([x_initializer, a_initializer, c_initializer])
    indices = np.argmax(pred, axis = -1)
    results = to_categorical(indices, num_classes=78)
    ### END CODE HERE ###
    
    return results, indices
```


```python
'''
grammar.py

'''

'''
Author:     Ji-Sung Kim, Evan Chow
Project:    jazzml / (used in) deepjazz
Purpose:    Extract, manipulate, process musical grammar

Directly taken then cleaned up from Evan Chow's jazzml, 
https://github.com/evancchow/jazzml,with permission.
'''

from collections import OrderedDict, defaultdict
from itertools import groupby
from music21 import *
import copy, random, pdb

#from preprocess import *

''' Helper function to determine if a note is a scale tone. '''
def __is_scale_tone(chord, note):
    # Method: generate all scales that have the chord notes th check if note is
    # in names

    # Derive major or minor scales (minor if 'other') based on the quality
    # of the chord.
    scaleType = scale.DorianScale() # i.e. minor pentatonic
    if chord.quality == 'major':
        scaleType = scale.MajorScale()
    # Can change later to deriveAll() for flexibility. If so then use list
    # comprehension of form [x for a in b for x in a].
    scales = scaleType.derive(chord) # use deriveAll() later for flexibility
    allPitches = list(set([pitch for pitch in scales.getPitches()]))
    allNoteNames = [i.name for i in allPitches] # octaves don't matter

    # Get note name. Return true if in the list of note names.
    noteName = note.name
    return (noteName in allNoteNames)

''' Helper function to determine if a note is an approach tone. '''
def __is_approach_tone(chord, note):
    # Method: see if note is +/- 1 a chord tone.

    for chordPitch in chord.pitches:
        stepUp = chordPitch.transpose(1)
        stepDown = chordPitch.transpose(-1)
        if (note.name == stepDown.name or 
            note.name == stepDown.getEnharmonic().name or
            note.name == stepUp.name or
            note.name == stepUp.getEnharmonic().name):
                return True
    return False

''' Helper function to determine if a note is a chord tone. '''
def __is_chord_tone(lastChord, note):
    return (note.name in (p.name for p in lastChord.pitches))

''' Helper function to generate a chord tone. '''
def __generate_chord_tone(lastChord):
    lastChordNoteNames = [p.nameWithOctave for p in lastChord.pitches]
    return note.Note(random.choice(lastChordNoteNames))

''' Helper function to generate a scale tone. '''
def __generate_scale_tone(lastChord):
    # Derive major or minor scales (minor if 'other') based on the quality
    # of the lastChord.
    scaleType = scale.WeightedHexatonicBlues() # minor pentatonic
    if lastChord.quality == 'major':
        scaleType = scale.MajorScale()
    # Can change later to deriveAll() for flexibility. If so then use list
    # comprehension of form [x for a in b for x in a].
    scales = scaleType.derive(lastChord) # use deriveAll() later for flexibility
    allPitches = list(set([pitch for pitch in scales.getPitches()]))
    allNoteNames = [i.name for i in allPitches] # octaves don't matter

    # Return a note (no octave here) in a scale that matches the lastChord.
    sNoteName = random.choice(allNoteNames)
    lastChordSort = lastChord.sortAscending()
    sNoteOctave = random.choice([i.octave for i in lastChordSort.pitches])
    sNote = note.Note(("%s%s" % (sNoteName, sNoteOctave)))
    return sNote

''' Helper function to generate an approach tone. '''
def __generate_approach_tone(lastChord):
    sNote = __generate_scale_tone(lastChord)
    aNote = sNote.transpose(random.choice([1, -1]))
    return aNote

''' Helper function to generate a random tone. '''
def __generate_arbitrary_tone(lastChord):
    return __generate_scale_tone(lastChord) # fix later, make random note.


''' Given the notes in a measure ('measure') and the chords in that measure
    ('chords'), generate a list of abstract grammatical symbols to represent 
    that measure as described in GTK's "Learning Jazz Grammars" (2009). 

    Inputs: 
    1) "measure" : a stream.Voice object where each element is a
        note.Note or note.Rest object.

        >>> m1
        <music21.stream.Voice 328482572>
        >>> m1[0]
        <music21.note.Rest rest>
        >>> m1[1]
        <music21.note.Note C>

        Can have instruments and other elements, removes them here.

    2) "chords" : a stream.Voice object where each element is a chord.Chord.

        >>> c1
        <music21.stream.Voice 328497548>
        >>> c1[0]
        <music21.chord.Chord E-4 G4 C4 B-3 G#2>
        >>> c1[1]
        <music21.chord.Chord B-3 F4 D4 A3>

        Can have instruments and other elements, removes them here. 

    Outputs:
    1) "fullGrammar" : a string that holds the abstract grammar for measure.
        Format: 
        (Remember, these are DURATIONS not offsets!)
        "R,0.125" : a rest element of  (1/32) length, or 1/8 quarter note. 
        "C,0.125<M-2,m-6>" : chord note of (1/32) length, generated
                             anywhere from minor 6th down to major 2nd down.
                             (interval <a,b> is not ordered). '''

def parse_melody(fullMeasureNotes, fullMeasureChords):
    # Remove extraneous elements.x
    measure = copy.deepcopy(fullMeasureNotes)
    chords = copy.deepcopy(fullMeasureChords)
    measure.removeByNotOfClass([note.Note, note.Rest])
    chords.removeByNotOfClass([chord.Chord])

    # Information for the start of the measure.
    # 1) measureStartTime: the offset for measure's start, e.g. 476.0.
    # 2) measureStartOffset: how long from the measure start to the first element.
    measureStartTime = measure[0].offset - (measure[0].offset % 4)
    measureStartOffset  = measure[0].offset - measureStartTime

    # Iterate over the notes and rests in measure, finding the grammar for each
    # note in the measure and adding an abstract grammatical string for it. 

    fullGrammar = ""
    prevNote = None # Store previous note. Need for interval.
    numNonRests = 0 # Number of non-rest elements. Need for updating prevNote.
    for ix, nr in enumerate(measure):
        # Get the last chord. If no last chord, then (assuming chords is of length
        # >0) shift first chord in chords to the beginning of the measure.
        try: 
            lastChord = [n for n in chords if n.offset <= nr.offset][-1]
        except IndexError:
            chords[0].offset = measureStartTime
            lastChord = [n for n in chords if n.offset <= nr.offset][-1]

        # FIRST, get type of note, e.g. R for Rest, C for Chord, etc.
        # Dealing with solo notes here. If unexpected chord: still call 'C'.
        elementType = ' '
        # R: First, check if it's a rest. Clearly a rest --> only one possibility.
        if isinstance(nr, note.Rest):
            elementType = 'R'
        # C: Next, check to see if note pitch is in the last chord.
        elif nr.name in lastChord.pitchNames or isinstance(nr, chord.Chord):
            elementType = 'C'
        # L: (Complement tone) Skip this for now.
        # S: Check if it's a scale tone.
        elif __is_scale_tone(lastChord, nr):
            elementType = 'S'
        # A: Check if it's an approach tone, i.e. +-1 halfstep chord tone.
        elif __is_approach_tone(lastChord, nr):
            elementType = 'A'
        # X: Otherwise, it's an arbitrary tone. Generate random note.
        else:
            elementType = 'X'

        # SECOND, get the length for each element. e.g. 8th note = R8, but
        # to simplify things you'll use the direct num, e.g. R,0.125
        if (ix == (len(measure)-1)):
            # formula for a in "a - b": start of measure (e.g. 476) + 4
            diff = measureStartTime + 4.0 - nr.offset
        else:
            diff = measure[ix + 1].offset - nr.offset

        # Combine into the note info.
        noteInfo = "%s,%.3f" % (elementType, nr.quarterLength) # back to diff

        # THIRD, get the deltas (max range up, max range down) based on where
        # the previous note was, +- minor 3. Skip rests (don't affect deltas).
        intervalInfo = ""
        if isinstance(nr, note.Note):
            numNonRests += 1
            if numNonRests == 1:
                prevNote = nr
            else:
                noteDist = interval.Interval(noteStart=prevNote, noteEnd=nr)
                noteDistUpper = interval.add([noteDist, "m3"])
                noteDistLower = interval.subtract([noteDist, "m3"])
                intervalInfo = ",<%s,%s>" % (noteDistUpper.directedName, 
                    noteDistLower.directedName)
                # print "Upper, lower: %s, %s" % (noteDistUpper,
                #     noteDistLower)
                # print "Upper, lower dnames: %s, %s" % (
                #     noteDistUpper.directedName,
                #     noteDistLower.directedName)
                # print "The interval: %s" % (intervalInfo)
                prevNote = nr

        # Return. Do lazy evaluation for real-time performance.
        grammarTerm = noteInfo + intervalInfo 
        fullGrammar += (grammarTerm + " ")

    return fullGrammar.rstrip()

''' Given a grammar string and chords for a measure, returns measure notes. '''
def unparse_grammar(m1_grammar, m1_chords):
    m1_elements = stream.Voice()
    currOffset = 0.0 # for recalculate last chord.
    prevElement = None
    for ix, grammarElement in enumerate(m1_grammar.split(' ')):
        terms = grammarElement.split(',')
        currOffset += float(terms[1]) # works just fine

        # Case 1: it's a rest. Just append
        if terms[0] == 'R':
            rNote = note.Rest(quarterLength = float(terms[1]))
            m1_elements.insert(currOffset, rNote)
            continue

        # Get the last chord first so you can find chord note, scale note, etc.
        try: 
            lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]
        except IndexError:
            m1_chords[0].offset = 0.0
            lastChord = [n for n in m1_chords if n.offset <= currOffset][-1]

        # Case: no < > (should just be the first note) so generate from range
        # of lowest chord note to highest chord note (if not a chord note, else
        # just generate one of the actual chord notes). 

        # Case #1: if no < > to indicate next note range. Usually this lack of < >
        # is for the first note (no precedent), or for rests.
        if (len(terms) == 2): # Case 1: if no < >.
            insertNote = note.Note() # default is C

            # Case C: chord note.
            if terms[0] == 'C':
                insertNote = __generate_chord_tone(lastChord)

            # Case S: scale note.
            elif terms[0] == 'S':
                insertNote = __generate_scale_tone(lastChord)

            # Case A: approach note.
            # Handle both A and X notes here for now.
            else:
                insertNote = __generate_approach_tone(lastChord)

            # Update the stream of generated notes
            insertNote.quarterLength = float(terms[1])
            if insertNote.octave < 4:
                insertNote.octave = 4
            m1_elements.insert(currOffset, insertNote)
            prevElement = insertNote

        # Case #2: if < > for the increment. Usually for notes after the first one.
        else:
            # Get lower, upper intervals and notes.
            interval1 = interval.Interval(terms[2].replace("<",''))
            interval2 = interval.Interval(terms[3].replace(">",''))
            if interval1.cents > interval2.cents:
                upperInterval, lowerInterval = interval1, interval2
            else:
                upperInterval, lowerInterval = interval2, interval1
            lowPitch = interval.transposePitch(prevElement.pitch, lowerInterval)
            highPitch = interval.transposePitch(prevElement.pitch, upperInterval)
            numNotes = int(highPitch.ps - lowPitch.ps + 1) # for range(s, e)

            # Case C: chord note, must be within increment (terms[2]).
            # First, transpose note with lowerInterval to get note that is
            # the lower bound. Then iterate over, and find valid notes. Then
            # choose randomly from those.
            
            if terms[0] == 'C':
                relevantChordTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_chord_tone(lastChord, currNote):
                        relevantChordTones.append(currNote)
                if len(relevantChordTones) > 1:
                    insertNote = random.choice([i for i in relevantChordTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantChordTones) == 1:
                    insertNote = relevantChordTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            # Case S: scale note, must be within increment.
            elif terms[0] == 'S':
                relevantScaleTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_scale_tone(lastChord, currNote):
                        relevantScaleTones.append(currNote)
                if len(relevantScaleTones) > 1:
                    insertNote = random.choice([i for i in relevantScaleTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantScaleTones) == 1:
                    insertNote = relevantScaleTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            # Case A: approach tone, must be within increment.
            # For now: handle both A and X cases.
            else:
                relevantApproachTones = []
                for i in range(0, numNotes):
                    currNote = note.Note(lowPitch.transpose(i).simplifyEnharmonic())
                    if __is_approach_tone(lastChord, currNote):
                        relevantApproachTones.append(currNote)
                if len(relevantApproachTones) > 1:
                    insertNote = random.choice([i for i in relevantApproachTones
                        if i.nameWithOctave != prevElement.nameWithOctave])
                elif len(relevantApproachTones) == 1:
                    insertNote = relevantApproachTones[0]
                else: # if no choices, set to prev element +-1 whole step
                    insertNote = prevElement.transpose(random.choice([-2,2]))
                if insertNote.octave < 3:
                    insertNote.octave = 3
                insertNote.quarterLength = float(terms[1])
                m1_elements.insert(currOffset, insertNote)

            # update the previous element.
            prevElement = insertNote

    return m1_elements    
```


```python
'''
inference_code.py
'''

def inference_model(LSTM_cell, densor, n_x = 78, n_a = 64, Ty = 100):
    """
    Uses the trained "LSTM_cell" and "densor" from model() to generate a sequence of values.
    
    Arguments:
    LSTM_cell -- the trained "LSTM_cell" from model(), Keras layer object
    densor -- the trained "densor" from model(), Keras layer object
    n_x -- number of unique values
    n_a -- number of units in the LSTM_cell
    Ty -- number of time steps to generate
    
    Returns:
    inference_model -- Keras model instance
    """
    
    # Define the input of your model with a shape 
    x0 = Input(shape=(1, n_x))
    
    # Define s0, initial hidden state for the decoder LSTM
    a0 = Input(shape=(n_a,), name='a0')
    c0 = Input(shape=(n_a,), name='c0')
    a = a0
    c = c0
    x = x0

    ### START CODE HERE ###
    # Step 1: Create an empty list of "outputs" to later store your predicted values (≈1 line)
    outputs = []
    
    # Step 2: Loop over Ty and generate a value at every time step
    for t in range(Ty):
        
        # Step 2.A: Perform one step of LSTM_cell (≈1 line)
        a, _, c = LSTM_cell(x, initial_state=[a, c])
        
        # Step 2.B: Apply Dense layer to the hidden state output of the LSTM_cell (≈1 line)
        out = densor(a)

        # Step 2.C: Append the prediction "out" to "outputs" (≈1 line)
        outputs.append(out)
        
        # Step 2.D: Set the prediction "out" to be the next input "x". You will need to use RepeatVector(1). (≈1 line)
        x = RepeatVector(1)(out)
        
    # Step 3: Create model instance with the correct "inputs" and "outputs" (≈1 line)
    inference_model = Model(inputs=[x0, a0, c0], outputs=outputs)
    ### END CODE HERE ###
    
    return inference_model


inference_model = inference_model(LSTM_cell, densor)


x1 = np.zeros((1, 1, 78))
x1[:,:,35] = 1
a1 = np.zeros((1, n_a))
c1 = np.zeros((1, n_a))
predicting = inference_model.predict([x1, a1, c1])


indices = np.argmax(predicting, axis = -1)
results = to_categorical(indices, num_classes=78)
```


```python
'''

midi.py

'''

"""
File: midi.py
Author: Addy771
Description: 
A script which converts MIDI files to WAV and optionally to MP3 using ffmpeg. 
Works by playing each file and using the stereo mix device to record at the same time
"""


import pyaudio  # audio recording
import wave     # file saving
import pygame   # midi playback
import fnmatch  # name matching
import os       # file listing


#### CONFIGURATION ####

do_ffmpeg_convert = True    # Uses FFmpeg to convert WAV files to MP3. Requires ffmpeg.exe in the script folder or PATH
do_wav_cleanup = True       # Deletes WAV files after conversion to MP3
sample_rate = 44100         # Sample rate used for WAV/MP3
channels = 2                # Audio channels (1 = mono, 2 = stereo)
buffer = 1024               # Audio buffer size
mp3_bitrate = 128           # Bitrate to save MP3 with in kbps (CBR)
input_device = 1            # Which recording device to use. On my system Stereo Mix = 1



# Begins playback of a MIDI file
def play_music(music_file):

    try:
        pygame.mixer.music.load(music_file)
        
    except pygame.error:
        print ("Couldn't play %s! (%s)" % (music_file, pygame.get_error()))
        return
        
    pygame.mixer.music.play()



# Init pygame playback
bitsize = -16   # unsigned 16 bit
pygame.mixer.init(sample_rate, bitsize, channels, buffer)

# optional volume 0 to 1.0
pygame.mixer.music.set_volume(1.0)

# Init pyAudio
format = pyaudio.paInt16
audio = pyaudio.PyAudio()



try:

    # Make a list of .mid files in the current directory and all subdirectories
    matches = []
    for root, dirnames, filenames in os.walk("./"):
        for filename in fnmatch.filter(filenames, '*.mid'):
            matches.append(os.path.join(root, filename))
            
    # Play each song in the list
    for song in matches:

        # Create a filename with a .wav extension
        file_name = os.path.splitext(os.path.basename(song))[0]
        new_file = file_name + '.wav'

        # Open the stream and start recording
        stream = audio.open(format=format, channels=channels, rate=sample_rate, input=True, input_device_index=input_device, frames_per_buffer=buffer)
        
        # Playback the song
        print("Playing " + file_name + ".mid\n")
        play_music(song)
        
        frames = []
        
        # Record frames while the song is playing
        while pygame.mixer.music.get_busy():
            frames.append(stream.read(buffer))
            
        # Stop recording
        stream.stop_stream()
        stream.close()

        
        # Configure wave file settings
        wave_file = wave.open(new_file, 'wb')
        wave_file.setnchannels(channels)
        wave_file.setsampwidth(audio.get_sample_size(format))
        wave_file.setframerate(sample_rate)
        
        print("Saving " + new_file)   
        
        # Write the frames to the wave file
        wave_file.writeframes(b''.join(frames))
        wave_file.close()
        
        # Call FFmpeg to handle the MP3 conversion if desired
        if do_ffmpeg_convert:
            os.system('ffmpeg -i ' + new_file + ' -y -f mp3 -ab ' + str(mp3_bitrate) + 'k -ac ' + str(channels) + ' -ar ' + str(sample_rate) + ' -vn ' + file_name + '.mp3')
            
            # Delete the WAV file if desired
            if do_wav_cleanup:        
                os.remove(new_file)
        
    # End PyAudio    
    audio.terminate()    
 
except KeyboardInterrupt:
    # if user hits Ctrl/C then exit
    # (works only in console mode)
    pygame.mixer.music.fadeout(1000)
    pygame.mixer.music.stop()
    raise SystemExit
```


```python
'''
music_utils.py
'''

from __future__ import print_function
import tensorflow as tf
import keras.backend as K
from keras.layers import RepeatVector
import sys
from music21 import *
import numpy as np
from grammar import *
from preprocess import *
from qa import *


def data_processing(corpus, values_indices, m = 60, Tx = 30):
    # cut the corpus into semi-redundant sequences of Tx values
    Tx = Tx 
    N_values = len(set(corpus))
    np.random.seed(0)
    X = np.zeros((m, Tx, N_values), dtype=np.bool)
    Y = np.zeros((m, Tx, N_values), dtype=np.bool)
    for i in range(m):
#         for t in range(1, Tx):
        random_idx = np.random.choice(len(corpus) - Tx)
        corp_data = corpus[random_idx:(random_idx + Tx)]
        for j in range(Tx):
            idx = values_indices[corp_data[j]]
            if j != 0:
                X[i, j, idx] = 1
                Y[i, j-1, idx] = 1
    
    Y = np.swapaxes(Y,0,1)
    Y = Y.tolist()
    return np.asarray(X), np.asarray(Y), N_values 

def next_value_processing(model, next_value, x, predict_and_sample, indices_values, abstract_grammars, duration, max_tries = 1000, temperature = 0.5):
    """
    Helper function to fix the first value.
    
    Arguments:
    next_value -- predicted and sampled value, index between 0 and 77
    x -- numpy-array, one-hot encoding of next_value
    predict_and_sample -- predict function
    indices_values -- a python dictionary mapping indices (0-77) into their corresponding unique value (ex: A,0.250,< m2,P-4 >)
    abstract_grammars -- list of grammars, on element can be: 'S,0.250,<m2,P-4> C,0.250,<P4,m-2> A,0.250,<P4,m-2>'
    duration -- scalar, index of the loop in the parent function
    max_tries -- Maximum numbers of time trying to fix the value
    
    Returns:
    next_value -- process predicted value
    """

    # fix first note: must not have < > and not be a rest
    if (duration < 0.00001):
        tries = 0
        while (next_value.split(',')[0] == 'R' or 
            len(next_value.split(',')) != 2):
            # give up after 1000 tries; random from input's first notes
            if tries >= max_tries:
                #print('Gave up on first note generation after', max_tries, 'tries')
                # np.random is exclusive to high
                rand = np.random.randint(0, len(abstract_grammars))
                next_value = abstract_grammars[rand].split(' ')[0]
            else:
                next_value = predict_and_sample(model, x, indices_values, temperature)

            tries += 1
            
    return next_value


def sequence_to_matrix(sequence, values_indices):
    """
    Convert a sequence (slice of the corpus) into a matrix (numpy) of one-hot vectors corresponding 
    to indices in values_indices
    
    Arguments:
    sequence -- python list
    
    Returns:
    x -- numpy-array of one-hot vectors 
    """
    sequence_len = len(sequence)
    x = np.zeros((1, sequence_len, len(values_indices)))
    for t, value in enumerate(sequence):
        if (not value in values_indices): print(value)
        x[0, t, values_indices[value]] = 1.
    return x

def one_hot(x):
    x = K.argmax(x)
    x = tf.one_hot(x, 78) 
    x = RepeatVector(1)(x)
    return x
```


```python
'''
preprocess.py
'''

'''
Author:     Ji-Sung Kim
Project:    deepjazz
Purpose:    Parse, cleanup and process data.

Code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml with
express permission.
'''

from __future__ import print_function

from music21 import *
from collections import defaultdict, OrderedDict
from itertools import groupby, zip_longest

from grammar import *

from grammar import parse_melody
from music_utils import *

#----------------------------HELPER FUNCTIONS----------------------------------#

''' Helper function to parse a MIDI file into its measures and chords '''
def __parse_midi(data_fn):
    # Parse the MIDI data for separate melody and accompaniment parts.
    midi_data = converter.parse(data_fn)
    # Get melody part, compress into single voice.
    melody_stream = midi_data[5]     # For Metheny piece, Melody is Part #5.
    melody1, melody2 = melody_stream.getElementsByClass(stream.Voice)
    for j in melody2:
        melody1.insert(j.offset, j)
    melody_voice = melody1

    for i in melody_voice:
        if i.quarterLength == 0.0:
            i.quarterLength = 0.25

    # Change key signature to adhere to comp_stream (1 sharp, mode = major).
    # Also add Electric Guitar. 
    melody_voice.insert(0, instrument.ElectricGuitar())
    melody_voice.insert(0, key.KeySignature(sharps=1))

    # The accompaniment parts. Take only the best subset of parts from
    # the original data. Maybe add more parts, hand-add valid instruments.
    # Should add least add a string part (for sparse solos).
    # Verified are good parts: 0, 1, 6, 7 '''
    partIndices = [0, 1, 6, 7]
    comp_stream = stream.Voice()
    comp_stream.append([j.flat for i, j in enumerate(midi_data) 
        if i in partIndices])

    # Full stream containing both the melody and the accompaniment. 
    # All parts are flattened. 
    full_stream = stream.Voice()
    for i in range(len(comp_stream)):
        full_stream.append(comp_stream[i])
    full_stream.append(melody_voice)

    # Extract solo stream, assuming you know the positions ..ByOffset(i, j).
    # Note that for different instruments (with stream.flat), you NEED to use
    # stream.Part(), not stream.Voice().
    # Accompanied solo is in range [478, 548)
    solo_stream = stream.Voice()
    for part in full_stream:
        curr_part = stream.Part()
        curr_part.append(part.getElementsByClass(instrument.Instrument))
        curr_part.append(part.getElementsByClass(tempo.MetronomeMark))
        curr_part.append(part.getElementsByClass(key.KeySignature))
        curr_part.append(part.getElementsByClass(meter.TimeSignature))
        curr_part.append(part.getElementsByOffset(476, 548, 
                                                  includeEndBoundary=True))
        cp = curr_part.flat
        solo_stream.insert(cp)

    # Group by measure so you can classify. 
    # Note that measure 0 is for the time signature, metronome, etc. which have
    # an offset of 0.0.
    melody_stream = solo_stream[-1]
    measures = OrderedDict()
    offsetTuples = [(int(n.offset / 4), n) for n in melody_stream]
    measureNum = 0 # for now, don't use real m. nums (119, 120)
    for key_x, group in groupby(offsetTuples, lambda x: x[0]):
        measures[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Get the stream of chords.
    # offsetTuples_chords: group chords by measure number.
    chordStream = solo_stream[0]
    chordStream.removeByClass(note.Rest)
    chordStream.removeByClass(note.Note)
    offsetTuples_chords = [(int(n.offset / 4), n) for n in chordStream]

    # Generate the chord structure. Use just track 1 (piano) since it is
    # the only instrument that has chords. 
    # Group into 4s, just like before. 
    chords = OrderedDict()
    measureNum = 0
    for key_x, group in groupby(offsetTuples_chords, lambda x: x[0]):
        chords[measureNum] = [n[1] for n in group]
        measureNum += 1

    # Fix for the below problem.
    #   1) Find out why len(measures) != len(chords).
    #   ANSWER: resolves at end but melody ends 1/16 before last measure so doesn't
    #           actually show up, while the accompaniment's beat 1 right after does.
    #           Actually on second thought: melody/comp start on Ab, and resolve to
    #           the same key (Ab) so could actually just cut out last measure to loop.
    #           Decided: just cut out the last measure. 
    del chords[len(chords) - 1]
    assert len(chords) == len(measures)

    return measures, chords

''' Helper function to get the grammatical data from given musical data. '''
def __get_abstract_grammars(measures, chords):
    # extract grammars
    abstract_grammars = []
    for ix in range(1, len(measures)):
        m = stream.Voice()
        for i in measures[ix]:
            m.insert(i.offset, i)
        c = stream.Voice()
        for j in chords[ix]:
            c.insert(j.offset, j)
        parsed = parse_melody(m, c)
        abstract_grammars.append(parsed)

    return abstract_grammars

#----------------------------PUBLIC FUNCTIONS----------------------------------#

''' Get musical data from a MIDI file '''
def get_musical_data(data_fn):
    
    measures, chords = __parse_midi(data_fn)
    abstract_grammars = __get_abstract_grammars(measures, chords)

    return chords, abstract_grammars

''' Get corpus data from grammatical data '''
def get_corpus_data(abstract_grammars):
    corpus = [x for sublist in abstract_grammars for x in sublist.split(' ')]
    values = set(corpus)
    val_indices = dict((v, i) for i, v in enumerate(values))
    indices_val = dict((i, v) for i, v in enumerate(values))

    return corpus, values, val_indices, indices_val

'''
def load_music_utils():
    chord_data, raw_music_data = get_musical_data('data/original_metheny.mid')
    music_data, values, values_indices, indices_values = get_corpus_data(raw_music_data)

    X, Y = data_processing(music_data, values_indices, Tx = 20, step = 3)
    return (X, Y)
'''

```


```python
'''
qa.py
'''

'''
Author:     Ji-Sung Kim, Evan Chow
Project:    deepjazz
Purpose:    Provide pruning and cleanup functions.

Code adapted from Evan Chow's jazzml, https://github.com/evancchow/jazzml 
with express permission.
'''
from itertools import zip_longest
import random

from music21 import *

#----------------------------HELPER FUNCTIONS----------------------------------#

''' Helper function to down num to the nearest multiple of mult. '''
def __roundDown(num, mult):
    return (float(num) - (float(num) % mult))

''' Helper function to round up num to nearest multiple of mult. '''
def __roundUp(num, mult):
    return __roundDown(num, mult) + mult

''' Helper function that, based on if upDown < 0 or upDown >= 0, rounds number 
    down or up respectively to nearest multiple of mult. '''
def __roundUpDown(num, mult, upDown):
    if upDown < 0:
        return __roundDown(num, mult)
    else:
        return __roundUp(num, mult)

''' Helper function, from recipes, to iterate over list in chunks of n 
    length. '''
def __grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)

#----------------------------PUBLIC FUNCTIONS----------------------------------#

''' Smooth the measure, ensuring that everything is in standard note lengths 
    (e.g., 0.125, 0.250, 0.333 ... ). '''
def prune_grammar(curr_grammar):
    pruned_grammar = curr_grammar.split(' ')

    for ix, gram in enumerate(pruned_grammar):
        terms = gram.split(',')
        terms[1] = str(__roundUpDown(float(terms[1]), 0.250, 
            random.choice([-1, 1])))
        pruned_grammar[ix] = ','.join(terms)
    pruned_grammar = ' '.join(pruned_grammar)

    return pruned_grammar

''' Remove repeated notes, and notes that are too close together. '''
def prune_notes(curr_notes):
    for n1, n2 in __grouper(curr_notes, n=2):
        if n2 == None: # corner case: odd-length list
            continue
        if isinstance(n1, note.Note) and isinstance(n2, note.Note):
            if n1.nameWithOctave == n2.nameWithOctave:
                curr_notes.remove(n2)

    return curr_notes

''' Perform quality assurance on notes '''
def clean_up_notes(curr_notes):
    removeIxs = []
    for ix, m in enumerate(curr_notes):
        # QA1: ensure nothing is of 0 quarter note len, if so changes its len
        if (m.quarterLength == 0.0):
            m.quarterLength = 0.250
        # QA2: ensure no two melody notes have same offset, i.e. form a chord.
        # Sorted, so same offset would be consecutive notes.
        if (ix < (len(curr_notes) - 1)):
            if (m.offset == curr_notes[ix + 1].offset and
                isinstance(curr_notes[ix + 1], note.Note)):
                removeIxs.append((ix + 1))
    curr_notes = [i for ix, i in enumerate(curr_notes) if ix not in removeIxs]

    return curr_notes
```

**References**

The ideas presented in this notebook came primarily from three computational music papers cited below. The implementation here also took significant inspiration and used many components from Ji-Sung Kim's github repository.

- Ji-Sung Kim, 2016, [deepjazz](https://github.com/jisungk/deepjazz)
- Jon Gillick, Kevin Tang and Robert Keller, 2009. [Learning Jazz Grammars](http://ai.stanford.edu/~kdtang/papers/smc09-jazzgrammar.pdf)
- Robert Keller and David Morrison, 2007, [A Grammatical Approach to Automatic Improvisation](http://smc07.uoa.gr/SMC07%20Proceedings/SMC07%20Paper%2055.pdf)
- François Pachet, 1999, [Surprising Harmonies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.5.7473&rep=rep1&type=pdf)

We're also grateful to François Germain for valuable feedback.

---
 
**PS: 欢迎扫码关注公众号：「SelfImprovementLab」！专注「深度学习」，「机器学习」，「人工智能」。以及 「早起」，「阅读」，「运动」，「英语 」「其他」不定期建群 打卡互助活动。**

<center><img src="http://upload-images.jianshu.io/upload_images/1157146-cab5ba89dfeeec4b.jpg?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240"></center>

