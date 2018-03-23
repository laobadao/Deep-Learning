
# Assignment | 05-week3 -Part_1-Neural Machine Translation

该系列仅在原课程基础上课后作业部分添加个人学习笔记，如有错误，还请批评指教。- ZJ
    
>[Coursera 课程](https://www.coursera.org/specializations/deep-learning) |[deeplearning.ai](https://www.deeplearning.ai/) |[网易云课堂](https://mooc.study.163.com/smartSpec/detail/1001319001.htm)

 [CSDN]()：
   

---

Welcome to your first programming assignment for this week! 

You will build a Neural Machine Translation (NMT) model to translate human readable dates ("25th of June, 2009") into machine readable dates ("2009-06-25"). You will do this using an attention model, one of the most sophisticated sequence to sequence models. 

您将建立一个神经机器翻译（NMT）模型，将人类可读日期（“2009年6月25日”）转换为机器可读日期（“2009-06-25”）。您将使用注意模型来完成此操作，这是模型序列中最复杂的序列之一。

This notebook was produced together with NVIDIA's Deep Learning Institute. 

Let's load all the packages you will need for this assignment.


```python
from keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from keras.layers import RepeatVector, Dense, Activation, Lambda
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.models import load_model, Model
import keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
%matplotlib inline


```

    d:\program files\python36\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.
    

## 1 - Translating human readable dates into machine readable dates

The model you will build here could be used to translate from one language to another, such as translating from English to Hindi. However, language translation requires massive datasets and usually takes days of training on GPUs. To give you a place to experiment with these models even without using massive datasets, we will instead use a simpler "date translation" task. 

您将在此创建的模型可用于从一种语言翻译为另一种语言，如从英语翻译为印地语。 但是，语言翻译需要大量的数据集，并且通常需要几天的GPU训练。 为了让您有机会尝试这些模型，即使不使用海量数据集，我们也会使用更简单的“日期转换”任务。

The network will input a date written in a variety of possible formats (*e.g. "the 29th of August 1958", "03/30/1968", "24 JUNE 1987"*) and translate them into standardized, machine readable dates (*e.g. "1958-08-29", "1968-03-30", "1987-06-24"*). We will have the network learn to output dates in the common machine-readable format YYYY-MM-DD. 

tqdm（读音：taqadum, تقدّم）在阿拉伯语中的意思是进展。tqdm可以在长循环中添加一个进度提示信息，用户只需要封装任意的迭代器 tqdm (iterator)，是一个快速、扩展性强的进度条工具库

<!-- 
Take a look at [nmt_utils.py](./nmt_utils.py) to see all the formatting. Count and figure out how the formats work, you will need this knowledge later. !--> 


```python
'''
nmt_utils.py

要把这里面代码 都看懂，写注释 学一遍

'''


import numpy as np
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from keras.utils import to_categorical
import keras.backend as K
import matplotlib.pyplot as plt

fake = Faker()
fake.seed(12345)
random.seed(12345)

# Define format of the data we would like to generate
FORMATS = ['short',
           'medium',
           'long',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'full',
           'd MMM YYY', 
           'd MMMM YYY',
           'dd MMM YYY',
           'd MMM, YYY',
           'd MMMM, YYY',
           'dd, MMM YYY',
           'd MM YY',
           'd MMMM YYY',
           'MMMM d YYY',
           'MMMM d, YYY',
           'dd.MM.YY']

# change this if you want it to work with another language
LOCALES = ['en_US']

def load_date():
    """
        Loads some fake dates 
        :returns: tuple containing human readable string, machine readable string, and date object
    """
    dt = fake.date_object()

    try:
        human_readable = format_date(dt, format=random.choice(FORMATS),  locale='en_US') # locale=random.choice(LOCALES))
        human_readable = human_readable.lower()
        human_readable = human_readable.replace(',','')
        machine_readable = dt.isoformat()
        
    except AttributeError as e:
        return None, None, None

    return human_readable, machine_readable, dt

def load_dataset(m):
    """
        Loads a dataset with m examples and vocabularies
        :m: the number of examples to generate
    """
    
    human_vocab = set()
    machine_vocab = set()
    dataset = []
    Tx = 30
    

    for i in tqdm(range(m)):
        h, m, _ = load_date()
        if h is not None:
            dataset.append((h, m))
            human_vocab.update(tuple(h))
            machine_vocab.update(tuple(m))
    
    human = dict(zip(sorted(human_vocab) + ['<unk>', '<pad>'], 
                     list(range(len(human_vocab) + 2))))
    inv_machine = dict(enumerate(sorted(machine_vocab)))
    machine = {v:k for k,v in inv_machine.items()}
 
    return dataset, human, machine, inv_machine

def preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty):
    
    X, Y = zip(*dataset)
    
    X = np.array([string_to_int(i, Tx, human_vocab) for i in X])
    Y = [string_to_int(t, Ty, machine_vocab) for t in Y]
    
    Xoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), X)))
    Yoh = np.array(list(map(lambda x: to_categorical(x, num_classes=len(machine_vocab)), Y)))

    return X, np.array(Y), Xoh, Yoh

def string_to_int(string, length, vocab):
    """
    Converts all strings in the vocabulary into a list of integers representing the positions of the
    input string's characters in the "vocab"
    
    Arguments:
    string -- input string, e.g. 'Wed 10 Jul 2007'
    length -- the number of time steps you'd like, determines if the output will be padded or cut
    vocab -- vocabulary, dictionary used to index every character of your "string"
    
    Returns:
    rep -- list of integers (or '<unk>') (size = length) representing the position of the string's character in the vocabulary
    """
    
    #make lower to standardize
    string = string.lower()
    string = string.replace(',','')
    
    if len(string) > length:
        string = string[:length]
        
    rep = list(map(lambda x: vocab.get(x, '<unk>'), string))
    
    if len(string) < length:
        rep += [vocab['<pad>']] * (length - len(string))
    
    #print (rep)
    return rep


def int_to_string(ints, inv_vocab):
    """
    Output a machine readable list of characters based on a list of indexes in the machine's vocabulary
    
    Arguments:
    ints -- list of integers representing indexes in the machine's vocabulary
    inv_vocab -- dictionary mapping machine readable indexes to machine readable characters 
    
    Returns:
    l -- list of characters corresponding to the indexes of ints thanks to the inv_vocab mapping
    """
    
    l = [inv_vocab[i] for i in ints]
    return l


EXAMPLES = ['3 May 1979', '5 Apr 09', '20th February 2016', 'Wed 10 Jul 2007']

def run_example(model, input_vocabulary, inv_output_vocabulary, text):
    encoded = string_to_int(text, TIME_STEPS, input_vocabulary)
    prediction = model.predict(np.array([encoded]))
    prediction = np.argmax(prediction[0], axis=-1)
    return int_to_string(prediction, inv_output_vocabulary)

def run_examples(model, input_vocabulary, inv_output_vocabulary, examples=EXAMPLES):
    predicted = []
    for example in examples:
        predicted.append(''.join(run_example(model, input_vocabulary, inv_output_vocabulary, example)))
        print('input:', example)
        print('output:', predicted[-1])
    return predicted


def softmax(x, axis=1):
    """Softmax activation function.
    # Arguments
        x : Tensor.
        axis: Integer, axis along which the softmax normalization is applied.
    # Returns
        Tensor, output of softmax transformation.
    # Raises
        ValueError: In case `dim(x) == 1`.
    """
    ndim = K.ndim(x)
    if ndim == 2:
        return K.softmax(x)
    elif ndim > 2:
        e = K.exp(x - K.max(x, axis=axis, keepdims=True))
        s = K.sum(e, axis=axis, keepdims=True)
        return e / s
    else:
        raise ValueError('Cannot apply softmax to a tensor that is 1D')
        

def plot_attention_map(model, input_vocabulary, inv_output_vocabulary, text, n_s = 128, num = 6, Tx = 30, Ty = 10):
    """
    Plot the attention map.
  
    """
    attention_map = np.zeros((10, 30))
    Ty, Tx = attention_map.shape
    
    s0 = np.zeros((1, n_s))
    c0 = np.zeros((1, n_s))
    layer = model.layers[num]

    encoded = np.array(string_to_int(text, Tx, input_vocabulary)).reshape((1, 30))
    encoded = np.array(list(map(lambda x: to_categorical(x, num_classes=len(input_vocabulary)), encoded)))

    f = K.function(model.inputs, [layer.get_output_at(t) for t in range(Ty)])
    r = f([encoded, s0, c0])
    
    for t in range(Ty):
        for t_prime in range(Tx):
            attention_map[t][t_prime] = r[t][0,t_prime,0]

    # Normalize attention map
#     row_max = attention_map.max(axis=1)
#     attention_map = attention_map / row_max[:, None]

    prediction = model.predict([encoded, s0, c0])
    
    predicted_text = []
    for i in range(len(prediction)):
        predicted_text.append(int(np.argmax(prediction[i], axis=1)))
        
    predicted_text = list(predicted_text)
    predicted_text = int_to_string(predicted_text, inv_output_vocabulary)
    text_ = list(text)
    
    # get the lengths of the string
    input_length = len(text)
    output_length = Ty
    
    # Plot the attention_map
    plt.clf()
    f = plt.figure(figsize=(8, 8.5))
    ax = f.add_subplot(1, 1, 1)

    # add image
    i = ax.imshow(attention_map, interpolation='nearest', cmap='Blues')

    # add colorbar
    cbaxes = f.add_axes([0.2, 0, 0.6, 0.03])
    cbar = f.colorbar(i, cax=cbaxes, orientation='horizontal')
    cbar.ax.set_xlabel('Alpha value (Probability output of the "softmax")', labelpad=2)

    # add labels
    ax.set_yticks(range(output_length))
    ax.set_yticklabels(predicted_text[:output_length])

    ax.set_xticks(range(input_length))
    ax.set_xticklabels(text_[:input_length], rotation=45)

    ax.set_xlabel('Input Sequence')
    ax.set_ylabel('Output Sequence')

    # add grid and legend
    ax.grid()

    #f.show()
    
    return attention_map
```

### 1.1 - Dataset

We will train the model on a dataset of 10000 human readable dates and their equivalent, standardized, machine readable dates. Let's run the following cells to load the dataset and print some examples. 


```python
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
```

    100%|█████████████████████████████████████████████████████| 10000/10000 [00:00<00:00, 14157.20it/s]
    


```python
print(dataset[:10]) # 0 到 9 不包含 10 
print(dataset[0])
print(type(dataset))
```

    [('9 may 1998', '1998-05-09'), ('10.09.70', '1970-09-10'), ('4/28/90', '1990-04-28'), ('thursday january 26 1995', '1995-01-26'), ('monday march 7 1983', '1983-03-07'), ('sunday may 22 1988', '1988-05-22'), ('tuesday july 8 2008', '2008-07-08'), ('08 sep 1999', '1999-09-08'), ('1 jan 1981', '1981-01-01'), ('monday may 22 1995', '1995-05-22')]
    ('9 may 1998', '1998-05-09')
    <class 'list'>
    

You've loaded:
- `dataset`: a list of tuples of (human readable date, machine readable date) 元组的 list
- `human_vocab`: a python dictionary mapping all characters used in the human readable dates to an integer-valued index  一个Python字典将人类可读日期中使用的所有字符映射为整数值索引
- `machine_vocab`: a python dictionary mapping all characters used in machine readable dates to an integer-valued index. These indices are not necessarily consistent with `human_vocab`. 
- `inv_machine_vocab`: the inverse dictionary of `machine_vocab`, mapping from indices back to characters. 

Let's preprocess the data and map the raw text data into the index values. We will also use Tx=30 (which we assume is the maximum length of the human readable date; if we get a longer input, we would have to truncate it) and Ty=10 (since "YYYY-MM-DD" is 10 characters long). 

我们预处理数据并将原始文本数据映射到索引值。 我们还将使用Tx = 30（我们假设它是人类可读日期的最大长度;如果我们得到更长的输入，我们将不得不截断它）并且Ty = 10（因为“YYYY-MM-DD”是10 长字符）。


```python
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)
```

    X.shape: (10000, 30)
    Y.shape: (10000, 10)
    Xoh.shape: (10000, 30, 37)
    Yoh.shape: (10000, 10, 11)
    

You now have:
- `X`: a processed version of the human readable dates in the training set, where each character is replaced by an index mapped to the character via `human_vocab`. Each date is further padded to $T_x$ values with a special character (< pad >). `X.shape = (m, Tx)` 
- `Y`: a processed version of the machine readable dates in the training set, where each character is replaced by the index it is mapped to in `machine_vocab`. You should have `Y.shape = (m, Ty)`. 
- `Xoh`: one-hot version of `X`, the "1" entry's index is mapped to the character thanks to `human_vocab`. `Xoh.shape = (m, Tx, len(human_vocab))`
- `Yoh`: one-hot version of `Y`, the "1" entry's index is mapped to the character thanks to `machine_vocab`. `Yoh.shape = (m, Tx, len(machine_vocab))`. Here, `len(machine_vocab) = 11` since there are 11 characters ('-' as well as 0-9). 


- `X`: 训练集中人类可读日期的处理版本，其中每个字符由通过 `human_vocab` 映射到字符的索引替换。 每个日期用特殊字符（<pad>）进一步填充到 $T_x$ 值。 `X.shape = (m, Tx)`
    
- `Y`: 训练集中机器可读日期的处理版本，其中每个字符都被其映射到`machine_vocab`中的索引替换。 你应该有`Y.shape =（m，Ty）`。 
- `Xoh`:一个 one-hot 向量版本的`X`，由于`human_vocab`，“1”条目的索引被映射到字符。 `Xoh.shape =（m，Tx，len（human_vocab））`
- `Yoh`: “Y”的one-hot 向量 版本，由于“machine_vocab”，“1”条目的索引被映射到字符。 `Yoh.shape =（m，Tx，len（machine_vocab））`。 这里`len（machine_vocab）= 11`因为有11个字符（' - '和0-9）。

Lets also look at some examples of preprocessed training examples. Feel free to play with `index` in the cell below to navigate the dataset and see how source/target dates are preprocessed. 

我们也看一些预处理训练例子的例子。 随意使用下面的单元格中的`index`来浏览数据集，并查看如何预处理源/目标日期。


```python
index = 5
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])
```

    Source date: sunday may 22 1988
    Target date: 1988-05-22
    
    Source after preprocessing (indices): [29 31 25 16 13 34  0 24 13 34  0  5  5  0  4 12 11 11 36 36 36 36 36 36
     36 36 36 36 36 36]
    Target after preprocessing (indices): [ 2 10  9  9  0  1  6  0  3  3]
    
    Source after preprocessing (one-hot): [[0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     [0. 0. 0. ... 0. 0. 0.]
     ...
     [0. 0. 0. ... 0. 0. 1.]
     [0. 0. 0. ... 0. 0. 1.]
     [0. 0. 0. ... 0. 0. 1.]]
    Target after preprocessing (one-hot): [[0. 0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]]
    

## 2 - Neural machine translation with attention

If you had to translate a book's paragraph from French to English, you would not read the whole paragraph, then close the book and translate. Even during the translation process, you would read/re-read and focus on the parts of the French paragraph corresponding to the parts of the English you are writing down. 

The attention mechanism tells a Neural Machine Translation model where it should pay attention to at any step. 


如果你必须把一个书的段落从法文翻译成英文，你不会阅读整段，然后合上书本并翻译。 即使在翻译过程中，您也会阅读/重读，并专注于与您正在写下的英语部分相对应的法语段落的部分。

注意力机制告诉神经机器翻译模型，在任何一个步骤中应该注意它。

### 2.1 - Attention mechanism

In this part, you will implement the attention mechanism presented in the lecture videos. Here is a figure to remind you how the model works. The diagram on the left shows the attention model. The diagram on the right shows what one "Attention" step does to calculate the attention variables $\alpha^{\langle t, t' \rangle}$, which are used to compute the context variable $context^{\langle t \rangle}$ for each timestep in the output ($t=1, \ldots, T_y$).（concatenate 串联）

在这一部分，您将实现视频中提出的注意力机制。 这是一个提醒你模型如何工作运行的图。 左侧的图表显示了注意力模型。 右图显示了“注意力”步骤计算注意力变量的方法$\alpha^{\langle t, t' \rangle}$, 用于计算上下文变量 $context^{\langle t \rangle}$ 用于输出中的每个时间步($t=1, \ldots, T_y$).

<table>
<td> 
<img src="images/attn_model.png" style="width:500;height:500px;"> <br>
</td> 
<td> 
<img src="images/attn_mechanism.png" style="width:500;height:500px;"> <br>
</td> 
</table>
<caption><center> **Figure 1**: Neural machine translation with attention</center></caption>



Here are some properties of the model that you may notice: 

- There are two separate LSTMs in this model (see diagram on the left). Because the one at the bottom of the picture is a Bi-directional LSTM and comes *before* the attention mechanism, we will call it *pre-attention* Bi-LSTM. The LSTM at the top of the diagram comes *after* the attention mechanism, so we will call it the *post-attention* LSTM. The pre-attention Bi-LSTM goes through $T_x$ time steps; the post-attention LSTM goes through $T_y$ time steps. 

（在这个模型中有两个单独的 LSTM（见左图）。 因为图片底部的一个是双向 LSTM，并且在关注机制之前，我们将其称为注意力前的 Bi-LSTM。 图表顶部的 LSTM出现在关注机制之后，因此我们将其称为注意力后的 LSTM。 预注意 Bi-LST 经过 $T_x$ 时间步长; 后注意力 LSTM 经历$T_y$ 的时间步骤。）

- The post-attention LSTM passes $s^{\langle t \rangle}, c^{\langle t \rangle}$ from one time step to the next. In the lecture videos, we were using only a basic RNN for the post-activation sequence model, so the state captured by the RNN output activations $s^{\langle t\rangle}$. But since we are using an LSTM here, the LSTM has both the output activation $s^{\langle t\rangle}$ and the hidden cell state $c^{\langle t\rangle}$. However, unlike previous text generation examples (such as Dinosaurus in week 1), in this model the post-activation LSTM at time $t$ does will not take the specific generated $y^{\langle t-1 \rangle}$ as input; it only takes $s^{\langle t\rangle}$ and $c^{\langle t\rangle}$ as input. We have designed the model this way, because (unlike language generation where adjacent characters are highly correlated) there isn't as strong a dependency between the previous character and the next character in a YYYY-MM-DD date. 

后注意力 LSTM 从一个时间步到下一个时间通过 $s^{\langle t \rangle}, c^{\langle t \rangle}$。 在讲座视频中，我们仅使用了基本的 RNN 作为激活后序列模型，因此 RNN 输出激活捕获的状态为  $s^{\langle t\rangle}$. 但是由于我们在这里使用 LSTM，因此 LSTM 同时具有输出激活$s^{\langle t\rangle}$ 和隐藏单元状态 $c^{\langle t\rangle}$. 然而，与以前的文本生成示例（如第1周的 Dinosaurus）不同，在此模型中，$t$ 后的激活后 LSTM 不会将具体产生的 $y^{\langle t-1 \rangle}$作为输入; 它只需要 $s^{\langle t\rangle}$ and $c^{\langle t\rangle}$ a为输入。 我们以这种方式设计了模型，因为（与邻近字符高度相关的语言生成不同），在 YYYY-MM-DD 日期中，前一个字符与下一个字符之间的依赖性不强。

- We use $a^{\langle t \rangle} = [\overrightarrow{a}^{\langle t \rangle}; \overleftarrow{a}^{\langle t \rangle}]$ to represent the concatenation of the activations of both the forward-direction and backward-directions of the pre-attention Bi-LSTM. 我们用  $a^{\langle t \rangle} = [\overrightarrow{a}^{\langle t \rangle}; \overleftarrow{a}^{\langle t \rangle}]$以表示预注意Bi-LSTM的前向和后向激活的连接。

- The diagram on the right uses a `RepeatVector` node to copy $s^{\langle t-1 \rangle}$'s value $T_x$ times, and then `Concatenation` to concatenate $s^{\langle t-1 \rangle}$ and $a^{\langle t \rangle}$ to compute $e^{\langle t, t'}$, which is then passed through a softmax to compute $\alpha^{\langle t, t' \rangle}$. We'll explain how to use `RepeatVector` and `Concatenation` in Keras below. 

Lets implement this model. You will start by implementing two functions: `one_step_attention()` and `model()`.

**1) `one_step_attention()`**: At step $t$, given all the hidden states of the Bi-LSTM ($[a^{<1>},a^{<2>}, ..., a^{<T_x>}]$) and the previous hidden state of the second LSTM ($s^{<t-1>}$), `one_step_attention()` will compute the attention weights ($[\alpha^{<t,1>},\alpha^{<t,2>}, ..., \alpha^{<t,T_x>}]$) and output the context vector (see Figure  1 (right) for details):
$$context^{<t>} = \sum_{t' = 0}^{T_x} \alpha^{<t,t'>}a^{<t'>}\tag{1}$$ 

Note that we are denoting the attention in this notebook $context^{\langle t \rangle}$. In the lecture videos, the context was denoted $c^{\langle t \rangle}$, but here we are calling it $context^{\langle t \rangle}$ to avoid confusion with the (post-attention) LSTM's internal memory cell variable, which is sometimes also denoted $c^{\langle t \rangle}$. 
  
**2) `model()`**: Implements the entire model. It first runs the input through a Bi-LSTM to get back $[a^{<1>},a^{<2>}, ..., a^{<T_x>}]$. Then, it calls `one_step_attention()` $T_y$ times (`for` loop). At each iteration of this loop, it gives the computed context vector $c^{<t>}$ to the second LSTM, and runs the output of the LSTM through a dense layer with softmax activation to generate a prediction $\hat{y}^{<t>}$. 

[CSDN: DenseNet 简介](https://blog.csdn.net/bryan__/article/details/77337109):https://blog.csdn.net/bryan__/article/details/77337109


**Exercise**: Implement `one_step_attention()`. The function `model()` will call the layers in `one_step_attention()` $T_y$ using a for-loop, and it is important that all $T_y$ copies have the same weights. I.e., it should not re-initiaiize the weights every time. In other words, all $T_y$ steps should have shared weights. Here's how you can implement layers with shareable weights in Keras (它不应该每次重新初始化权重。 换句话说，所有 $T_y$  步骤应该具有共享权重。 以下是如何在Keras中实现可共享权重的图层：):
1. Define the layer objects (as global variables for examples).
2. Call these objects when propagating the input.

We have defined the layers you need as global variables. Please run the following cells to create them. Please check the Keras documentation to make sure you understand what these layers are: [RepeatVector()](https://keras.io/layers/core/#repeatvector), [Concatenate()](https://keras.io/layers/merge/#concatenate), [Dense()](https://keras.io/layers/core/#dense), [Activation()](https://keras.io/layers/core/#activation), [Dot()](https://keras.io/layers/merge/#dot).


```python
# Defined shared layers as global variables
repeator = RepeatVector(Tx)
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
dotor = Dot(axes = 1)
```

Now you can use these layers to implement `one_step_attention()`. In order to propagate a Keras tensor object X through one of these layers, use `layer(X)` (or `layer([X,Y])` if it requires multiple inputs.), e.g. `densor(X)` will propagate X through the `Dense(1)` layer defined above.


```python
# GRADED FUNCTION: one_step_attention

def one_step_attention(a, s_prev):
    """
    Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
    "alphas" and the hidden states "a" of the Bi-LSTM.
    
    Arguments:
    a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
    s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
    
    Returns:
    context -- context vector, input of the next (post-attetion) LSTM cell
    """
    
    ### START CODE HERE ###
    # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
    s_prev = repeator(s_prev)
    # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
    concat = concatenator([a, s_prev])
    # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
    e = densor1(concat)
    # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
    energies = densor2(e)
    # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
    alphas = activator(energies)
    # Use dotor together with "alphas" and "a" to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
    context = dotor([alphas, a])
    ### END CODE HERE ###
    
    return context
```

You will be able to check the expected output of `one_step_attention()` after you've coded the `model()` function.

**Exercise**: Implement `model()` as explained in figure 2 and the text above. Again, we have defined global layers that will share weights to be used in `model()`.


```python
n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state = True)
output_layer = Dense(len(machine_vocab), activation=softmax)
```

Now you can use these layers $T_y$ times in a `for` loop to generate the outputs, and their parameters will not be reinitialized. You will have to carry out the following steps: 

1. Propagate the input into a [Bidirectional](https://keras.io/layers/wrappers/#bidirectional) [LSTM](https://keras.io/layers/recurrent/#lstm)
2. Iterate for $t = 0, \dots, T_y-1$: 
    1. Call `one_step_attention()` on $[\alpha^{<t,1>},\alpha^{<t,2>}, ..., \alpha^{<t,T_x>}]$ and $s^{<t-1>}$ to get the context vector $context^{<t>}$.
    2. Give $context^{<t>}$ to the post-attention LSTM cell. Remember pass in the previous hidden-state $s^{\langle t-1\rangle}$ and cell-states $c^{\langle t-1\rangle}$ of this LSTM using `initial_state= [previous hidden state, previous cell state]`. Get back the new hidden state $s^{<t>}$ and the new cell state $c^{<t>}$.
    3. Apply a softmax layer to $s^{<t>}$, get the output. 
    4. Save the output by adding it to the list of outputs.

3. Create your Keras model instance, it should have three inputs ("inputs", $s^{<0>}$ and $c^{<0>}$) and output the list of "outputs".


```python
# GRADED FUNCTION: model

def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    Arguments:
    Tx -- length of the input sequence
    Ty -- length of the output sequence
    n_a -- hidden state size of the Bi-LSTM
    n_s -- hidden state size of the post-attention LSTM
    human_vocab_size -- size of the python dictionary "human_vocab"
    machine_vocab_size -- size of the python dictionary "machine_vocab"

    Returns:
    model -- Keras model instance
    """
    
    # Define the inputs of your model with a shape (Tx,)
    # Define s0 and c0, initial hidden state for the decoder LSTM of shape (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0
    
    # Initialize empty list of outputs
    outputs = []
    
    ### START CODE HERE ###
    
    # Step 1: Define your pre-attention Bi-LSTM. Remember to use return_sequences=True. (≈ 1 line)
    a = Bidirectional(LSTM(n_a, return_sequences=True), input_shape=(m, Tx, n_a*2))(X)
    print(a.shape)
    print(Ty)
    # Step 2: Iterate for Ty steps
    for t in range(Ty):
    
        # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
        context = one_step_attention(a, s)
        
        # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
        # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(context, initial_state = [s, c])
        
        # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
        out = output_layer(s)
        
        # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
        outputs.append(out)
    
    # Step 3: Create model instance taking three inputs and returning the list of outputs. (≈ 1 line)
    model = Model(inputs=[X,s0,c0],outputs=outputs)
    
    ### END CODE HERE ###
    
    return model
```

Run the following cell to create your model.


```python
model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
```

    (?, ?, 64)
    10
    

Let's get a summary of the model to check if it matches the expected output.


```python
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            (None, 30, 37)       0                                            
    __________________________________________________________________________________________________
    s0 (InputLayer)                 (None, 64)           0                                            
    __________________________________________________________________________________________________
    bidirectional_2 (Bidirectional) (None, 30, 64)       17920       input_2[0][0]                    
    __________________________________________________________________________________________________
    repeat_vector_1 (RepeatVector)  (None, 30, 64)       0           s0[0][0]                         
                                                                     lstm_1[0][0]                     
                                                                     lstm_1[1][0]                     
                                                                     lstm_1[2][0]                     
                                                                     lstm_1[3][0]                     
                                                                     lstm_1[4][0]                     
                                                                     lstm_1[5][0]                     
                                                                     lstm_1[6][0]                     
                                                                     lstm_1[7][0]                     
                                                                     lstm_1[8][0]                     
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 30, 128)      0           bidirectional_2[0][0]            
                                                                     repeat_vector_1[1][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[2][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[3][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[4][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[5][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[6][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[7][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[8][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[9][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[10][0]           
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 30, 10)       1290        concatenate_1[0][0]              
                                                                     concatenate_1[1][0]              
                                                                     concatenate_1[2][0]              
                                                                     concatenate_1[3][0]              
                                                                     concatenate_1[4][0]              
                                                                     concatenate_1[5][0]              
                                                                     concatenate_1[6][0]              
                                                                     concatenate_1[7][0]              
                                                                     concatenate_1[8][0]              
                                                                     concatenate_1[9][0]              
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 30, 1)        11          dense_1[0][0]                    
                                                                     dense_1[1][0]                    
                                                                     dense_1[2][0]                    
                                                                     dense_1[3][0]                    
                                                                     dense_1[4][0]                    
                                                                     dense_1[5][0]                    
                                                                     dense_1[6][0]                    
                                                                     dense_1[7][0]                    
                                                                     dense_1[8][0]                    
                                                                     dense_1[9][0]                    
    __________________________________________________________________________________________________
    attention_weights (Activation)  (None, 30, 1)        0           dense_2[0][0]                    
                                                                     dense_2[1][0]                    
                                                                     dense_2[2][0]                    
                                                                     dense_2[3][0]                    
                                                                     dense_2[4][0]                    
                                                                     dense_2[5][0]                    
                                                                     dense_2[6][0]                    
                                                                     dense_2[7][0]                    
                                                                     dense_2[8][0]                    
                                                                     dense_2[9][0]                    
    __________________________________________________________________________________________________
    dot_1 (Dot)                     (None, 1, 64)        0           attention_weights[0][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[1][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[2][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[3][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[4][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[5][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[6][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[7][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[8][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[9][0]          
                                                                     bidirectional_2[0][0]            
    __________________________________________________________________________________________________
    c0 (InputLayer)                 (None, 64)           0                                            
    __________________________________________________________________________________________________
    lstm_1 (LSTM)                   [(None, 64), (None,  33024       dot_1[0][0]                      
                                                                     s0[0][0]                         
                                                                     c0[0][0]                         
                                                                     dot_1[1][0]                      
                                                                     lstm_1[0][0]                     
                                                                     lstm_1[0][2]                     
                                                                     dot_1[2][0]                      
                                                                     lstm_1[1][0]                     
                                                                     lstm_1[1][2]                     
                                                                     dot_1[3][0]                      
                                                                     lstm_1[2][0]                     
                                                                     lstm_1[2][2]                     
                                                                     dot_1[4][0]                      
                                                                     lstm_1[3][0]                     
                                                                     lstm_1[3][2]                     
                                                                     dot_1[5][0]                      
                                                                     lstm_1[4][0]                     
                                                                     lstm_1[4][2]                     
                                                                     dot_1[6][0]                      
                                                                     lstm_1[5][0]                     
                                                                     lstm_1[5][2]                     
                                                                     dot_1[7][0]                      
                                                                     lstm_1[6][0]                     
                                                                     lstm_1[6][2]                     
                                                                     dot_1[8][0]                      
                                                                     lstm_1[7][0]                     
                                                                     lstm_1[7][2]                     
                                                                     dot_1[9][0]                      
                                                                     lstm_1[8][0]                     
                                                                     lstm_1[8][2]                     
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 11)           715         lstm_1[0][0]                     
                                                                     lstm_1[1][0]                     
                                                                     lstm_1[2][0]                     
                                                                     lstm_1[3][0]                     
                                                                     lstm_1[4][0]                     
                                                                     lstm_1[5][0]                     
                                                                     lstm_1[6][0]                     
                                                                     lstm_1[7][0]                     
                                                                     lstm_1[8][0]                     
                                                                     lstm_1[9][0]                     
    ==================================================================================================
    Total params: 52,960
    Trainable params: 52,960
    Non-trainable params: 0
    __________________________________________________________________________________________________
    

**Expected Output**:

Here is the summary you should see
<table>
    <tr>
        <td>
            **Total params:**
        </td>
        <td>
         185,484
        </td>
    </tr>
        <tr>
        <td>
            **Trainable params:**
        </td>
        <td>
         185,484
        </td>
    </tr>
            <tr>
        <td>
            **Non-trainable params:**
        </td>
        <td>
         0
        </td>
    </tr>
                    <tr>
        <td>
            **bidirectional_1's output shape **
        </td>
        <td>
         (None, 30, 128)  
        </td>
    </tr>
    <tr>
        <td>
            **repeat_vector_1's output shape **
        </td>
        <td>
         (None, 30, 128)  
        </td>
    </tr>
                <tr>
        <td>
            **concatenate_1's output shape **
        </td>
        <td>
         (None, 30, 256) 
        </td>
    </tr>
            <tr>
        <td>
            **attention_weights's output shape **
        </td>
        <td>
         (None, 30, 1)  
        </td>
    </tr>
        <tr>
        <td>
            **dot_1's output shape **
        </td>
        <td>
         (None, 1, 128) 
        </td>
    </tr>
           <tr>
        <td>
            **dense_2's output shape **
        </td>
        <td>
         (None, 11) 
        </td>
    </tr>
</table>


最后得出的相关参数，与预期的参数不同，要看下为啥。

As usual, after creating your model in Keras, you need to compile it and define what loss, optimizer and metrics your are want to use.像往常一样，在Keras中创建模型后，您需要编译它并定义要使用的损失，优化程序和指标。 Compile your model using `categorical_crossentropy` loss, a custom [Adam](https://keras.io/optimizers/#adam) [optimizer](https://keras.io/optimizers/#usage-of-optimizers) (`learning rate = 0.005`, $\beta_1 = 0.9$, $\beta_2 = 0.999$, `decay = 0.01`)  and `['accuracy']` metrics:


```python
### START CODE HERE ### (≈2 lines)
opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
### END CODE HERE ###
```

The last step is to define all your inputs and outputs to fit the model:
- You already have X of shape $(m = 10000, T_x = 30)$ containing the training examples.
- You need to create `s0` and `c0` to initialize your `post_activation_LSTM_cell` with 0s.
- Given the `model()` you coded, you need the "outputs" to be a list of 11 elements of shape (m, T_y). So that: `outputs[i][0], ..., outputs[i][Ty]` represent the true labels (characters) corresponding to the $i^{th}$ training example (`X[i]`). More generally, `outputs[i][j]` is the true label of the $j^{th}$ character in the $i^{th}$ training example.


```python
s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0,1))
```

Let's now fit the model and run it for one epoch.


```python
model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)
```

    Epoch 1/1
     1600/10000 [===>..........................] - ETA: 18:21 - loss: 23.9635 - dense_3_loss_1: 2.4029 - dense_3_loss_2: 2.3787 - dense_3_loss_3: 2.3947 - dense_3_loss_4: 2.3965 - dense_3_loss_5: 2.4078 - dense_3_loss_6: 2.3760 - dense_3_loss_7: 2.4009 - dense_3_loss_8: 2.4096 - dense_3_loss_9: 2.3987 - dense_3_loss_10: 2.3977 - dense_3_acc_1: 0.0000e+00 - dense_3_acc_2: 0.0900 - dense_3_acc_3: 0.1000 - dense_3_acc_4: 0.0800 - dense_3_acc_5: 0.0000e+00 - dense_3_acc_6: 0.0100 - dense_3_acc_7: 0.0500 - dense_3_acc_8: 0.0000e+00 - dense_3_acc_9: 0.0000e+00 - dense_3_acc_10: 0.130 - ETA: 9:10 - loss: 23.7224 - dense_3_loss_1: 2.3929 - dense_3_loss_2: 2.3608 - dense_3_loss_3: 2.3881 - dense_3_loss_4: 2.4066 - dense_3_loss_5: 2.3475 - dense_3_loss_6: 2.3221 - dense_3_loss_7: 2.4053 - dense_3_loss_8: 2.3386 - dense_3_loss_9: 2.3616 - dense_3_loss_10: 2.3989 - dense_3_acc_1: 0.0000e+00 - dense_3_acc_2: 0.1700 - dense_3_acc_3: 0.1400 - dense_3_acc_4: 0.0700 - dense_3_acc_5: 0.1350 - dense_3_acc_6: 0.2750 - dense_3_acc_7: 0.0700 - dense_3_acc_8: 0.1050 - dense_3_acc_9: 0.1450 - dense_3_acc_10: 0.1000            - ETA: 6:06 - loss: 23.4463 - dense_3_loss_1: 2.3844 - dense_3_loss_2: 2.3363 - dense_3_loss_3: 2.3719 - dense_3_loss_4: 2.4187 - dense_3_loss_5: 2.2777 - dense_3_loss_6: 2.2507 - dense_3_loss_7: 2.4078 - dense_3_loss_8: 2.2546 - dense_3_loss_9: 2.3272 - dense_3_loss_10: 2.4170 - dense_3_acc_1: 0.0000e+00 - dense_3_acc_2: 0.2167 - dense_3_acc_3: 0.1467 - dense_3_acc_4: 0.0733 - dense_3_acc_5: 0.1700 - dense_3_acc_6: 0.3700 - dense_3_acc_7: 0.0767 - dense_3_acc_8: 0.1267 - dense_3_acc_9: 0.1833 - dense_3_acc_10: 0.10 - ETA: 4:34 - loss: 23.1588 - dense_3_loss_1: 2.3738 - dense_3_loss_2: 2.3104 - dense_3_loss_3: 2.3632 - dense_3_loss_4: 2.4468 - dense_3_loss_5: 2.1941 - dense_3_loss_6: 2.1623 - dense_3_loss_7: 2.4163 - dense_3_loss_8: 2.1514 - dense_3_loss_9: 2.2885 - dense_3_loss_10: 2.4520 - dense_3_acc_1: 0.0000e+00 - dense_3_acc_2: 0.2225 - dense_3_acc_3: 0.1325 - dense_3_acc_4: 0.0650 - dense_3_acc_5: 0.2050 - dense_3_acc_6: 0.4175 - dense_3_acc_7: 0.0725 - dense_3_acc_8: 0.1375 - dense_3_acc_9: 0.2100 - dense_3_acc_10: 0.10 - ETA: 3:39 - loss: 22.9440 - dense_3_loss_1: 2.3621 - dense_3_loss_2: 2.2798 - dense_3_loss_3: 2.3571 - dense_3_loss_4: 2.4936 - dense_3_loss_5: 2.0867 - dense_3_loss_6: 2.0559 - dense_3_loss_7: 2.4907 - dense_3_loss_8: 2.0182 - dense_3_loss_9: 2.2603 - dense_3_loss_10: 2.5396 - dense_3_acc_1: 0.0000e+00 - dense_3_acc_2: 0.2200 - dense_3_acc_3: 0.1280 - dense_3_acc_4: 0.0580 - dense_3_acc_5: 0.2860 - dense_3_acc_6: 0.3960 - dense_3_acc_7: 0.0620 - dense_3_acc_8: 0.2360 - dense_3_acc_9: 0.1940 - dense_3_acc_10: 0.08 - ETA: 3:02 - loss: 22.7903 - dense_3_loss_1: 2.3420 - dense_3_loss_2: 2.2622 - dense_3_loss_3: 2.3793 - dense_3_loss_4: 2.5425 - dense_3_loss_5: 1.9612 - dense_3_loss_6: 1.9670 - dense_3_loss_7: 2.5650 - dense_3_loss_8: 1.8697 - dense_3_loss_9: 2.2452 - dense_3_loss_10: 2.6562 - dense_3_acc_1: 0.0000e+00 - dense_3_acc_2: 0.1833 - dense_3_acc_3: 0.1067 - dense_3_acc_4: 0.0483 - dense_3_acc_5: 0.4050 - dense_3_acc_6: 0.3300 - dense_3_acc_7: 0.0517 - dense_3_acc_8: 0.3633 - dense_3_acc_9: 0.1617 - dense_3_acc_10: 0.07 - ETA: 2:35 - loss: 22.6810 - dense_3_loss_1: 2.3252 - dense_3_loss_2: 2.2463 - dense_3_loss_3: 2.3815 - dense_3_loss_4: 2.5958 - dense_3_loss_5: 1.8617 - dense_3_loss_6: 1.9110 - dense_3_loss_7: 2.6395 - dense_3_loss_8: 1.7550 - dense_3_loss_9: 2.2203 - dense_3_loss_10: 2.7448 - dense_3_acc_1: 0.0000e+00 - dense_3_acc_2: 0.1571 - dense_3_acc_3: 0.0914 - dense_3_acc_4: 0.0414 - dense_3_acc_5: 0.4900 - dense_3_acc_6: 0.2829 - dense_3_acc_7: 0.0443 - dense_3_acc_8: 0.4543 - dense_3_acc_9: 0.1386 - dense_3_acc_10: 0.06 - ETA: 2:15 - loss: 22.5816 - dense_3_loss_1: 2.3082 - dense_3_loss_2: 2.2404 - dense_3_loss_3: 2.3993 - dense_3_loss_4: 2.6183 - dense_3_loss_5: 1.7960 - dense_3_loss_6: 1.8820 - dense_3_loss_7: 2.6622 - dense_3_loss_8: 1.6804 - dense_3_loss_9: 2.2054 - dense_3_loss_10: 2.7894 - dense_3_acc_1: 0.0000e+00 - dense_3_acc_2: 0.1375 - dense_3_acc_3: 0.0800 - dense_3_acc_4: 0.0362 - dense_3_acc_5: 0.5538 - dense_3_acc_6: 0.2475 - dense_3_acc_7: 0.0388 - dense_3_acc_8: 0.5225 - dense_3_acc_9: 0.1212 - dense_3_acc_10: 0.05 - ETA: 2:00 - loss: 22.4720 - dense_3_loss_1: 2.2964 - dense_3_loss_2: 2.2337 - dense_3_loss_3: 2.4017 - dense_3_loss_4: 2.6215 - dense_3_loss_5: 1.7603 - dense_3_loss_6: 1.8735 - dense_3_loss_7: 2.6573 - dense_3_loss_8: 1.6418 - dense_3_loss_9: 2.1875 - dense_3_loss_10: 2.7983 - dense_3_acc_1: 0.0000e+00 - dense_3_acc_2: 0.1222 - dense_3_acc_3: 0.0711 - dense_3_acc_4: 0.0322 - dense_3_acc_5: 0.6033 - dense_3_acc_6: 0.2200 - dense_3_acc_7: 0.0344 - dense_3_acc_8: 0.5756 - dense_3_acc_9: 0.1078 - dense_3_acc_10: 0.04 - ETA: 1:48 - loss: 22.3729 - dense_3_loss_1: 2.2867 - dense_3_loss_2: 2.2260 - dense_3_loss_3: 2.3951 - dense_3_loss_4: 2.6205 - dense_3_loss_5: 1.7405 - dense_3_loss_6: 1.8725 - dense_3_loss_7: 2.6436 - dense_3_loss_8: 1.6236 - dense_3_loss_9: 2.1638 - dense_3_loss_10: 2.8005 - dense_3_acc_1: 1.0000e-03 - dense_3_acc_2: 0.1100 - dense_3_acc_3: 0.0640 - dense_3_acc_4: 0.0290 - dense_3_acc_5: 0.6430 - dense_3_acc_6: 0.1980 - dense_3_acc_7: 0.0310 - dense_3_acc_8: 0.6180 - dense_3_acc_9: 0.0970 - dense_3_acc_10: 0.04 - ETA: 1:38 - loss: 22.3013 - dense_3_loss_1: 2.2754 - dense_3_loss_2: 2.2213 - dense_3_loss_3: 2.3939 - dense_3_loss_4: 2.6179 - dense_3_loss_5: 1.7298 - dense_3_loss_6: 1.8724 - dense_3_loss_7: 2.6352 - dense_3_loss_8: 1.6166 - dense_3_loss_9: 2.1458 - dense_3_loss_10: 2.7930 - dense_3_acc_1: 0.0536 - dense_3_acc_2: 0.1000 - dense_3_acc_3: 0.0582 - dense_3_acc_4: 0.0264 - dense_3_acc_5: 0.6755 - dense_3_acc_6: 0.1800 - dense_3_acc_7: 0.0282 - dense_3_acc_8: 0.6527 - dense_3_acc_9: 0.0882 - dense_3_acc_10: 0.0391   - ETA: 1:29 - loss: 22.2300 - dense_3_loss_1: 2.2661 - dense_3_loss_2: 2.2121 - dense_3_loss_3: 2.3872 - dense_3_loss_4: 2.6116 - dense_3_loss_5: 1.7213 - dense_3_loss_6: 1.8700 - dense_3_loss_7: 2.6358 - dense_3_loss_8: 1.6122 - dense_3_loss_9: 2.1320 - dense_3_loss_10: 2.7816 - dense_3_acc_1: 0.0942 - dense_3_acc_2: 0.0917 - dense_3_acc_3: 0.0533 - dense_3_acc_4: 0.0242 - dense_3_acc_5: 0.7025 - dense_3_acc_6: 0.1650 - dense_3_acc_7: 0.0258 - dense_3_acc_8: 0.6817 - dense_3_acc_9: 0.0808 - dense_3_acc_10: 0.03 - ETA: 1:22 - loss: 22.1636 - dense_3_loss_1: 2.2566 - dense_3_loss_2: 2.2016 - dense_3_loss_3: 2.3809 - dense_3_loss_4: 2.6153 - dense_3_loss_5: 1.7127 - dense_3_loss_6: 1.8637 - dense_3_loss_7: 2.6287 - dense_3_loss_8: 1.6072 - dense_3_loss_9: 2.1149 - dense_3_loss_10: 2.7820 - dense_3_acc_1: 0.1262 - dense_3_acc_2: 0.1069 - dense_3_acc_3: 0.0492 - dense_3_acc_4: 0.0223 - dense_3_acc_5: 0.7254 - dense_3_acc_6: 0.1523 - dense_3_acc_7: 0.0238 - dense_3_acc_8: 0.7062 - dense_3_acc_9: 0.0746 - dense_3_acc_10: 0.03 - ETA: 1:16 - loss: 22.1060 - dense_3_loss_1: 2.2487 - dense_3_loss_2: 2.1877 - dense_3_loss_3: 2.3713 - dense_3_loss_4: 2.6224 - dense_3_loss_5: 1.7024 - dense_3_loss_6: 1.8524 - dense_3_loss_7: 2.6383 - dense_3_loss_8: 1.5996 - dense_3_loss_9: 2.1032 - dense_3_loss_10: 2.7801 - dense_3_acc_1: 0.1336 - dense_3_acc_2: 0.1293 - dense_3_acc_3: 0.0593 - dense_3_acc_4: 0.0207 - dense_3_acc_5: 0.7450 - dense_3_acc_6: 0.1414 - dense_3_acc_7: 0.0221 - dense_3_acc_8: 0.7271 - dense_3_acc_9: 0.0693 - dense_3_acc_10: 0.03 - ETA: 1:10 - loss: 22.0454 - dense_3_loss_1: 2.2402 - dense_3_loss_2: 2.1726 - dense_3_loss_3: 2.3656 - dense_3_loss_4: 2.6359 - dense_3_loss_5: 1.6898 - dense_3_loss_6: 1.8380 - dense_3_loss_7: 2.6434 - dense_3_loss_8: 1.5886 - dense_3_loss_9: 2.0920 - dense_3_loss_10: 2.7794 - dense_3_acc_1: 0.1313 - dense_3_acc_2: 0.1447 - dense_3_acc_3: 0.0673 - dense_3_acc_4: 0.0273 - dense_3_acc_5: 0.7613 - dense_3_acc_6: 0.1320 - dense_3_acc_7: 0.0207 - dense_3_acc_8: 0.7453 - dense_3_acc_9: 0.0647 - dense_3_acc_10: 0.02 - ETA: 1:06 - loss: 21.9791 - dense_3_loss_1: 2.2314 - dense_3_loss_2: 2.1554 - dense_3_loss_3: 2.3618 - dense_3_loss_4: 2.6453 - dense_3_loss_5: 1.6744 - dense_3_loss_6: 1.8184 - dense_3_loss_7: 2.6523 - dense_3_loss_8: 1.5741 - dense_3_loss_9: 2.0772 - dense_3_loss_10: 2.7887 - dense_3_acc_1: 0.1238 - dense_3_acc_2: 0.1569 - dense_3_acc_3: 0.0731 - dense_3_acc_4: 0.0350 - dense_3_acc_5: 0.7244 - dense_3_acc_6: 0.1269 - dense_3_acc_7: 0.0194 - dense_3_acc_8: 0.7612 - dense_3_acc_9: 0.0606 - dense_3_acc_10: 0.0269 3200/10000 [========>.....................] - ETA: 1:02 - loss: 21.9262 - dense_3_loss_1: 2.2247 - dense_3_loss_2: 2.1354 - dense_3_loss_3: 2.3549 - dense_3_loss_4: 2.6645 - dense_3_loss_5: 1.6578 - dense_3_loss_6: 1.8005 - dense_3_loss_7: 2.6595 - dense_3_loss_8: 1.5577 - dense_3_loss_9: 2.0756 - dense_3_loss_10: 2.7956 - dense_3_acc_1: 0.1165 - dense_3_acc_2: 0.1712 - dense_3_acc_3: 0.0794 - dense_3_acc_4: 0.0371 - dense_3_acc_5: 0.6824 - dense_3_acc_6: 0.1306 - dense_3_acc_7: 0.0182 - dense_3_acc_8: 0.7753 - dense_3_acc_9: 0.0571 - dense_3_acc_10: 0.02 - ETA: 58s - loss: 21.8677 - dense_3_loss_1: 2.2190 - dense_3_loss_2: 2.1134 - dense_3_loss_3: 2.3466 - dense_3_loss_4: 2.6801 - dense_3_loss_5: 1.6398 - dense_3_loss_6: 1.7869 - dense_3_loss_7: 2.6662 - dense_3_loss_8: 1.5396 - dense_3_loss_9: 2.0746 - dense_3_loss_10: 2.8014 - dense_3_acc_1: 0.1128 - dense_3_acc_2: 0.1861 - dense_3_acc_3: 0.0861 - dense_3_acc_4: 0.0406 - dense_3_acc_5: 0.6578 - dense_3_acc_6: 0.1233 - dense_3_acc_7: 0.0172 - dense_3_acc_8: 0.7878 - dense_3_acc_9: 0.0539 - dense_3_acc_10: 0.0239 - ETA: 54s - loss: 21.8055 - dense_3_loss_1: 2.2080 - dense_3_loss_2: 2.0939 - dense_3_loss_3: 2.3493 - dense_3_loss_4: 2.6888 - dense_3_loss_5: 1.6215 - dense_3_loss_6: 1.7738 - dense_3_loss_7: 2.6749 - dense_3_loss_8: 1.5212 - dense_3_loss_9: 2.0738 - dense_3_loss_10: 2.8004 - dense_3_acc_1: 0.1137 - dense_3_acc_2: 0.1916 - dense_3_acc_3: 0.0905 - dense_3_acc_4: 0.0453 - dense_3_acc_5: 0.6758 - dense_3_acc_6: 0.1168 - dense_3_acc_7: 0.0163 - dense_3_acc_8: 0.7989 - dense_3_acc_9: 0.0511 - dense_3_acc_10: 0.022 - ETA: 51s - loss: 21.7523 - dense_3_loss_1: 2.1984 - dense_3_loss_2: 2.0732 - dense_3_loss_3: 2.3461 - dense_3_loss_4: 2.6946 - dense_3_loss_5: 1.6038 - dense_3_loss_6: 1.7637 - dense_3_loss_7: 2.6877 - dense_3_loss_8: 1.5040 - dense_3_loss_9: 2.0733 - dense_3_loss_10: 2.8077 - dense_3_acc_1: 0.1080 - dense_3_acc_2: 0.1995 - dense_3_acc_3: 0.0975 - dense_3_acc_4: 0.0430 - dense_3_acc_5: 0.6920 - dense_3_acc_6: 0.1110 - dense_3_acc_7: 0.0155 - dense_3_acc_8: 0.8090 - dense_3_acc_9: 0.0485 - dense_3_acc_10: 0.021 - ETA: 49s - loss: 21.7055 - dense_3_loss_1: 2.1916 - dense_3_loss_2: 2.0520 - dense_3_loss_3: 2.3372 - dense_3_loss_4: 2.7121 - dense_3_loss_5: 1.5876 - dense_3_loss_6: 1.7581 - dense_3_loss_7: 2.6958 - dense_3_loss_8: 1.4892 - dense_3_loss_9: 2.0672 - dense_3_loss_10: 2.8146 - dense_3_acc_1: 0.1029 - dense_3_acc_2: 0.2052 - dense_3_acc_3: 0.1033 - dense_3_acc_4: 0.0410 - dense_3_acc_5: 0.7067 - dense_3_acc_6: 0.1057 - dense_3_acc_7: 0.0148 - dense_3_acc_8: 0.8181 - dense_3_acc_9: 0.0462 - dense_3_acc_10: 0.020 - ETA: 46s - loss: 21.6644 - dense_3_loss_1: 2.1800 - dense_3_loss_2: 2.0302 - dense_3_loss_3: 2.3368 - dense_3_loss_4: 2.7226 - dense_3_loss_5: 1.5732 - dense_3_loss_6: 1.7559 - dense_3_loss_7: 2.7061 - dense_3_loss_8: 1.4776 - dense_3_loss_9: 2.0637 - dense_3_loss_10: 2.8185 - dense_3_acc_1: 0.0982 - dense_3_acc_2: 0.2277 - dense_3_acc_3: 0.1068 - dense_3_acc_4: 0.0391 - dense_3_acc_5: 0.7200 - dense_3_acc_6: 0.1009 - dense_3_acc_7: 0.0141 - dense_3_acc_8: 0.8264 - dense_3_acc_9: 0.0441 - dense_3_acc_10: 0.019 - ETA: 44s - loss: 21.6099 - dense_3_loss_1: 2.1730 - dense_3_loss_2: 2.0083 - dense_3_loss_3: 2.3291 - dense_3_loss_4: 2.7332 - dense_3_loss_5: 1.5605 - dense_3_loss_6: 1.7561 - dense_3_loss_7: 2.7041 - dense_3_loss_8: 1.4691 - dense_3_loss_9: 2.0594 - dense_3_loss_10: 2.8172 - dense_3_acc_1: 0.0939 - dense_3_acc_2: 0.2439 - dense_3_acc_3: 0.1139 - dense_3_acc_4: 0.0374 - dense_3_acc_5: 0.7322 - dense_3_acc_6: 0.0965 - dense_3_acc_7: 0.0135 - dense_3_acc_8: 0.8339 - dense_3_acc_9: 0.0422 - dense_3_acc_10: 0.018 - ETA: 42s - loss: 21.5564 - dense_3_loss_1: 2.1672 - dense_3_loss_2: 1.9859 - dense_3_loss_3: 2.3206 - dense_3_loss_4: 2.7406 - dense_3_loss_5: 1.5491 - dense_3_loss_6: 1.7584 - dense_3_loss_7: 2.7008 - dense_3_loss_8: 1.4635 - dense_3_loss_9: 2.0588 - dense_3_loss_10: 2.8114 - dense_3_acc_1: 0.0900 - dense_3_acc_2: 0.2579 - dense_3_acc_3: 0.1179 - dense_3_acc_4: 0.0358 - dense_3_acc_5: 0.7433 - dense_3_acc_6: 0.0925 - dense_3_acc_7: 0.0129 - dense_3_acc_8: 0.8408 - dense_3_acc_9: 0.0404 - dense_3_acc_10: 0.017 - ETA: 40s - loss: 21.5014 - dense_3_loss_1: 2.1610 - dense_3_loss_2: 1.9620 - dense_3_loss_3: 2.3098 - dense_3_loss_4: 2.7496 - dense_3_loss_5: 1.5394 - dense_3_loss_6: 1.7598 - dense_3_loss_7: 2.6993 - dense_3_loss_8: 1.4601 - dense_3_loss_9: 2.0586 - dense_3_loss_10: 2.8017 - dense_3_acc_1: 0.0864 - dense_3_acc_2: 0.2716 - dense_3_acc_3: 0.1272 - dense_3_acc_4: 0.0380 - dense_3_acc_5: 0.7536 - dense_3_acc_6: 0.0888 - dense_3_acc_7: 0.0124 - dense_3_acc_8: 0.8472 - dense_3_acc_9: 0.0388 - dense_3_acc_10: 0.017 - ETA: 38s - loss: 21.4546 - dense_3_loss_1: 2.1529 - dense_3_loss_2: 1.9355 - dense_3_loss_3: 2.3006 - dense_3_loss_4: 2.7661 - dense_3_loss_5: 1.5307 - dense_3_loss_6: 1.7592 - dense_3_loss_7: 2.6986 - dense_3_loss_8: 1.4581 - dense_3_loss_9: 2.0569 - dense_3_loss_10: 2.7961 - dense_3_acc_1: 0.0831 - dense_3_acc_2: 0.2862 - dense_3_acc_3: 0.1327 - dense_3_acc_4: 0.0396 - dense_3_acc_5: 0.7631 - dense_3_acc_6: 0.0854 - dense_3_acc_7: 0.0119 - dense_3_acc_8: 0.8531 - dense_3_acc_9: 0.0373 - dense_3_acc_10: 0.016 - ETA: 36s - loss: 21.3970 - dense_3_loss_1: 2.1449 - dense_3_loss_2: 1.9083 - dense_3_loss_3: 2.2955 - dense_3_loss_4: 2.7748 - dense_3_loss_5: 1.5214 - dense_3_loss_6: 1.7583 - dense_3_loss_7: 2.6959 - dense_3_loss_8: 1.4565 - dense_3_loss_9: 2.0545 - dense_3_loss_10: 2.7868 - dense_3_acc_1: 0.0800 - dense_3_acc_2: 0.3000 - dense_3_acc_3: 0.1348 - dense_3_acc_4: 0.0422 - dense_3_acc_5: 0.7719 - dense_3_acc_6: 0.0822 - dense_3_acc_7: 0.0115 - dense_3_acc_8: 0.8585 - dense_3_acc_9: 0.0359 - dense_3_acc_10: 0.015 - ETA: 35s - loss: 21.3418 - dense_3_loss_1: 2.1344 - dense_3_loss_2: 1.8798 - dense_3_loss_3: 2.2927 - dense_3_loss_4: 2.7826 - dense_3_loss_5: 1.5105 - dense_3_loss_6: 1.7565 - dense_3_loss_7: 2.6963 - dense_3_loss_8: 1.4552 - dense_3_loss_9: 2.0504 - dense_3_loss_10: 2.7834 - dense_3_acc_1: 0.0771 - dense_3_acc_2: 0.3146 - dense_3_acc_3: 0.1375 - dense_3_acc_4: 0.0446 - dense_3_acc_5: 0.7800 - dense_3_acc_6: 0.0793 - dense_3_acc_7: 0.0111 - dense_3_acc_8: 0.8636 - dense_3_acc_9: 0.0346 - dense_3_acc_10: 0.015 - ETA: 33s - loss: 21.3049 - dense_3_loss_1: 2.1313 - dense_3_loss_2: 1.8578 - dense_3_loss_3: 2.2853 - dense_3_loss_4: 2.7981 - dense_3_loss_5: 1.4978 - dense_3_loss_6: 1.7549 - dense_3_loss_7: 2.6954 - dense_3_loss_8: 1.4535 - dense_3_loss_9: 2.0443 - dense_3_loss_10: 2.7866 - dense_3_acc_1: 0.0745 - dense_3_acc_2: 0.3228 - dense_3_acc_3: 0.1421 - dense_3_acc_4: 0.0452 - dense_3_acc_5: 0.7876 - dense_3_acc_6: 0.0766 - dense_3_acc_7: 0.0107 - dense_3_acc_8: 0.8683 - dense_3_acc_9: 0.0334 - dense_3_acc_10: 0.014 - ETA: 32s - loss: 21.2452 - dense_3_loss_1: 2.1204 - dense_3_loss_2: 1.8326 - dense_3_loss_3: 2.2781 - dense_3_loss_4: 2.8053 - dense_3_loss_5: 1.4847 - dense_3_loss_6: 1.7534 - dense_3_loss_7: 2.6939 - dense_3_loss_8: 1.4519 - dense_3_loss_9: 2.0381 - dense_3_loss_10: 2.7866 - dense_3_acc_1: 0.0720 - dense_3_acc_2: 0.3350 - dense_3_acc_3: 0.1457 - dense_3_acc_4: 0.0460 - dense_3_acc_5: 0.7947 - dense_3_acc_6: 0.0740 - dense_3_acc_7: 0.0103 - dense_3_acc_8: 0.8727 - dense_3_acc_9: 0.0323 - dense_3_acc_10: 0.014 - ETA: 31s - loss: 21.1863 - dense_3_loss_1: 2.1080 - dense_3_loss_2: 1.8088 - dense_3_loss_3: 2.2693 - dense_3_loss_4: 2.8143 - dense_3_loss_5: 1.4717 - dense_3_loss_6: 1.7502 - dense_3_loss_7: 2.6961 - dense_3_loss_8: 1.4503 - dense_3_loss_9: 2.0325 - dense_3_loss_10: 2.7852 - dense_3_acc_1: 0.0697 - dense_3_acc_2: 0.3471 - dense_3_acc_3: 0.1506 - dense_3_acc_4: 0.0455 - dense_3_acc_5: 0.8013 - dense_3_acc_6: 0.0716 - dense_3_acc_7: 0.0100 - dense_3_acc_8: 0.8768 - dense_3_acc_9: 0.0345 - dense_3_acc_10: 0.016 - ETA: 30s - loss: 21.1332 - dense_3_loss_1: 2.0980 - dense_3_loss_2: 1.7886 - dense_3_loss_3: 2.2587 - dense_3_loss_4: 2.8211 - dense_3_loss_5: 1.4590 - dense_3_loss_6: 1.7466 - dense_3_loss_7: 2.7002 - dense_3_loss_8: 1.4475 - dense_3_loss_9: 2.0268 - dense_3_loss_10: 2.7868 - dense_3_acc_1: 0.0675 - dense_3_acc_2: 0.3569 - dense_3_acc_3: 0.1541 - dense_3_acc_4: 0.0475 - dense_3_acc_5: 0.8075 - dense_3_acc_6: 0.0694 - dense_3_acc_7: 0.0097 - dense_3_acc_8: 0.8806 - dense_3_acc_9: 0.0419 - dense_3_acc_10: 0.0187 4800/10000 [=============>................] - ETA: 28s - loss: 21.0850 - dense_3_loss_1: 2.0901 - dense_3_loss_2: 1.7716 - dense_3_loss_3: 2.2461 - dense_3_loss_4: 2.8295 - dense_3_loss_5: 1.4463 - dense_3_loss_6: 1.7428 - dense_3_loss_7: 2.7047 - dense_3_loss_8: 1.4429 - dense_3_loss_9: 2.0196 - dense_3_loss_10: 2.7914 - dense_3_acc_1: 0.0655 - dense_3_acc_2: 0.3645 - dense_3_acc_3: 0.1588 - dense_3_acc_4: 0.0500 - dense_3_acc_5: 0.8133 - dense_3_acc_6: 0.0673 - dense_3_acc_7: 0.0094 - dense_3_acc_8: 0.8842 - dense_3_acc_9: 0.0427 - dense_3_acc_10: 0.020 - ETA: 27s - loss: 21.0281 - dense_3_loss_1: 2.0786 - dense_3_loss_2: 1.7538 - dense_3_loss_3: 2.2327 - dense_3_loss_4: 2.8345 - dense_3_loss_5: 1.4333 - dense_3_loss_6: 1.7398 - dense_3_loss_7: 2.7100 - dense_3_loss_8: 1.4382 - dense_3_loss_9: 2.0150 - dense_3_loss_10: 2.7921 - dense_3_acc_1: 0.0635 - dense_3_acc_2: 0.3729 - dense_3_acc_3: 0.1659 - dense_3_acc_4: 0.0512 - dense_3_acc_5: 0.8188 - dense_3_acc_6: 0.0653 - dense_3_acc_7: 0.0091 - dense_3_acc_8: 0.8876 - dense_3_acc_9: 0.0415 - dense_3_acc_10: 0.022 - ETA: 26s - loss: 20.9753 - dense_3_loss_1: 2.0707 - dense_3_loss_2: 1.7401 - dense_3_loss_3: 2.2235 - dense_3_loss_4: 2.8375 - dense_3_loss_5: 1.4205 - dense_3_loss_6: 1.7376 - dense_3_loss_7: 2.7151 - dense_3_loss_8: 1.4339 - dense_3_loss_9: 2.0107 - dense_3_loss_10: 2.7857 - dense_3_acc_1: 0.0646 - dense_3_acc_2: 0.3783 - dense_3_acc_3: 0.1689 - dense_3_acc_4: 0.0520 - dense_3_acc_5: 0.8240 - dense_3_acc_6: 0.0634 - dense_3_acc_7: 0.0089 - dense_3_acc_8: 0.8909 - dense_3_acc_9: 0.0403 - dense_3_acc_10: 0.022 - ETA: 25s - loss: 20.9200 - dense_3_loss_1: 2.0582 - dense_3_loss_2: 1.7254 - dense_3_loss_3: 2.2128 - dense_3_loss_4: 2.8376 - dense_3_loss_5: 1.4093 - dense_3_loss_6: 1.7370 - dense_3_loss_7: 2.7185 - dense_3_loss_8: 1.4318 - dense_3_loss_9: 2.0055 - dense_3_loss_10: 2.7838 - dense_3_acc_1: 0.0803 - dense_3_acc_2: 0.3853 - dense_3_acc_3: 0.1711 - dense_3_acc_4: 0.0536 - dense_3_acc_5: 0.8289 - dense_3_acc_6: 0.0617 - dense_3_acc_7: 0.0086 - dense_3_acc_8: 0.8939 - dense_3_acc_9: 0.0392 - dense_3_acc_10: 0.025 - ETA: 24s - loss: 20.8527 - dense_3_loss_1: 2.0406 - dense_3_loss_2: 1.7092 - dense_3_loss_3: 2.2007 - dense_3_loss_4: 2.8372 - dense_3_loss_5: 1.4011 - dense_3_loss_6: 1.7376 - dense_3_loss_7: 2.7168 - dense_3_loss_8: 1.4316 - dense_3_loss_9: 1.9988 - dense_3_loss_10: 2.7792 - dense_3_acc_1: 0.0973 - dense_3_acc_2: 0.3941 - dense_3_acc_3: 0.1751 - dense_3_acc_4: 0.0557 - dense_3_acc_5: 0.8335 - dense_3_acc_6: 0.0600 - dense_3_acc_7: 0.0084 - dense_3_acc_8: 0.8968 - dense_3_acc_9: 0.0449 - dense_3_acc_10: 0.030 - ETA: 24s - loss: 20.8022 - dense_3_loss_1: 2.0268 - dense_3_loss_2: 1.6973 - dense_3_loss_3: 2.1864 - dense_3_loss_4: 2.8366 - dense_3_loss_5: 1.3963 - dense_3_loss_6: 1.7356 - dense_3_loss_7: 2.7170 - dense_3_loss_8: 1.4331 - dense_3_loss_9: 1.9922 - dense_3_loss_10: 2.7809 - dense_3_acc_1: 0.1108 - dense_3_acc_2: 0.3997 - dense_3_acc_3: 0.1800 - dense_3_acc_4: 0.0574 - dense_3_acc_5: 0.8379 - dense_3_acc_6: 0.0584 - dense_3_acc_7: 0.0082 - dense_3_acc_8: 0.8905 - dense_3_acc_9: 0.0508 - dense_3_acc_10: 0.031 - ETA: 23s - loss: 20.7514 - dense_3_loss_1: 2.0127 - dense_3_loss_2: 1.6839 - dense_3_loss_3: 2.1727 - dense_3_loss_4: 2.8413 - dense_3_loss_5: 1.3880 - dense_3_loss_6: 1.7352 - dense_3_loss_7: 2.7173 - dense_3_loss_8: 1.4323 - dense_3_loss_9: 1.9870 - dense_3_loss_10: 2.7810 - dense_3_acc_1: 0.1233 - dense_3_acc_2: 0.4049 - dense_3_acc_3: 0.1854 - dense_3_acc_4: 0.0572 - dense_3_acc_5: 0.8421 - dense_3_acc_6: 0.0569 - dense_3_acc_7: 0.0079 - dense_3_acc_8: 0.8931 - dense_3_acc_9: 0.0567 - dense_3_acc_10: 0.034 - ETA: 22s - loss: 20.6966 - dense_3_loss_1: 1.9984 - dense_3_loss_2: 1.6692 - dense_3_loss_3: 2.1666 - dense_3_loss_4: 2.8442 - dense_3_loss_5: 1.3762 - dense_3_loss_6: 1.7334 - dense_3_loss_7: 2.7212 - dense_3_loss_8: 1.4292 - dense_3_loss_9: 1.9832 - dense_3_loss_10: 2.7750 - dense_3_acc_1: 0.1355 - dense_3_acc_2: 0.4100 - dense_3_acc_3: 0.1863 - dense_3_acc_4: 0.0557 - dense_3_acc_5: 0.8460 - dense_3_acc_6: 0.0555 - dense_3_acc_7: 0.0078 - dense_3_acc_8: 0.8958 - dense_3_acc_9: 0.0570 - dense_3_acc_10: 0.039 - ETA: 21s - loss: 20.6379 - dense_3_loss_1: 1.9824 - dense_3_loss_2: 1.6534 - dense_3_loss_3: 2.1564 - dense_3_loss_4: 2.8465 - dense_3_loss_5: 1.3636 - dense_3_loss_6: 1.7344 - dense_3_loss_7: 2.7224 - dense_3_loss_8: 1.4252 - dense_3_loss_9: 1.9779 - dense_3_loss_10: 2.7757 - dense_3_acc_1: 0.1478 - dense_3_acc_2: 0.4159 - dense_3_acc_3: 0.1898 - dense_3_acc_4: 0.0544 - dense_3_acc_5: 0.8498 - dense_3_acc_6: 0.0541 - dense_3_acc_7: 0.0076 - dense_3_acc_8: 0.8983 - dense_3_acc_9: 0.0578 - dense_3_acc_10: 0.041 - ETA: 20s - loss: 20.5798 - dense_3_loss_1: 1.9660 - dense_3_loss_2: 1.6379 - dense_3_loss_3: 2.1464 - dense_3_loss_4: 2.8460 - dense_3_loss_5: 1.3516 - dense_3_loss_6: 1.7348 - dense_3_loss_7: 2.7273 - dense_3_loss_8: 1.4214 - dense_3_loss_9: 1.9720 - dense_3_loss_10: 2.7763 - dense_3_acc_1: 0.1593 - dense_3_acc_2: 0.4210 - dense_3_acc_3: 0.1945 - dense_3_acc_4: 0.0533 - dense_3_acc_5: 0.8533 - dense_3_acc_6: 0.0529 - dense_3_acc_7: 0.0074 - dense_3_acc_8: 0.9007 - dense_3_acc_9: 0.0619 - dense_3_acc_10: 0.043 - ETA: 20s - loss: 20.5246 - dense_3_loss_1: 1.9478 - dense_3_loss_2: 1.6216 - dense_3_loss_3: 2.1355 - dense_3_loss_4: 2.8483 - dense_3_loss_5: 1.3423 - dense_3_loss_6: 1.7364 - dense_3_loss_7: 2.7270 - dense_3_loss_8: 1.4189 - dense_3_loss_9: 1.9665 - dense_3_loss_10: 2.7802 - dense_3_acc_1: 0.1709 - dense_3_acc_2: 0.4265 - dense_3_acc_3: 0.1995 - dense_3_acc_4: 0.0528 - dense_3_acc_5: 0.8567 - dense_3_acc_6: 0.0516 - dense_3_acc_7: 0.0072 - dense_3_acc_8: 0.8988 - dense_3_acc_9: 0.0658 - dense_3_acc_10: 0.045 - ETA: 19s - loss: 20.4572 - dense_3_loss_1: 1.9285 - dense_3_loss_2: 1.6047 - dense_3_loss_3: 2.1252 - dense_3_loss_4: 2.8489 - dense_3_loss_5: 1.3312 - dense_3_loss_6: 1.7371 - dense_3_loss_7: 2.7257 - dense_3_loss_8: 1.4169 - dense_3_loss_9: 1.9610 - dense_3_loss_10: 2.7779 - dense_3_acc_1: 0.1832 - dense_3_acc_2: 0.4339 - dense_3_acc_3: 0.2041 - dense_3_acc_4: 0.0520 - dense_3_acc_5: 0.8600 - dense_3_acc_6: 0.0505 - dense_3_acc_7: 0.0070 - dense_3_acc_8: 0.8952 - dense_3_acc_9: 0.0707 - dense_3_acc_10: 0.048 - ETA: 18s - loss: 20.4009 - dense_3_loss_1: 1.9115 - dense_3_loss_2: 1.5894 - dense_3_loss_3: 2.1161 - dense_3_loss_4: 2.8496 - dense_3_loss_5: 1.3205 - dense_3_loss_6: 1.7358 - dense_3_loss_7: 2.7285 - dense_3_loss_8: 1.4169 - dense_3_loss_9: 1.9545 - dense_3_loss_10: 2.7780 - dense_3_acc_1: 0.1931 - dense_3_acc_2: 0.4409 - dense_3_acc_3: 0.2082 - dense_3_acc_4: 0.0509 - dense_3_acc_5: 0.8631 - dense_3_acc_6: 0.0493 - dense_3_acc_7: 0.0073 - dense_3_acc_8: 0.8871 - dense_3_acc_9: 0.0784 - dense_3_acc_10: 0.050 - ETA: 18s - loss: 20.3431 - dense_3_loss_1: 1.8934 - dense_3_loss_2: 1.5745 - dense_3_loss_3: 2.1066 - dense_3_loss_4: 2.8505 - dense_3_loss_5: 1.3101 - dense_3_loss_6: 1.7327 - dense_3_loss_7: 2.7326 - dense_3_loss_8: 1.4167 - dense_3_loss_9: 1.9482 - dense_3_loss_10: 2.7777 - dense_3_acc_1: 0.2035 - dense_3_acc_2: 0.4478 - dense_3_acc_3: 0.2124 - dense_3_acc_4: 0.0498 - dense_3_acc_5: 0.8661 - dense_3_acc_6: 0.0483 - dense_3_acc_7: 0.0078 - dense_3_acc_8: 0.8807 - dense_3_acc_9: 0.0846 - dense_3_acc_10: 0.052 - ETA: 17s - loss: 20.2944 - dense_3_loss_1: 1.8782 - dense_3_loss_2: 1.5620 - dense_3_loss_3: 2.0993 - dense_3_loss_4: 2.8484 - dense_3_loss_5: 1.2997 - dense_3_loss_6: 1.7316 - dense_3_loss_7: 2.7345 - dense_3_loss_8: 1.4153 - dense_3_loss_9: 1.9434 - dense_3_loss_10: 2.7820 - dense_3_acc_1: 0.2113 - dense_3_acc_2: 0.4506 - dense_3_acc_3: 0.2149 - dense_3_acc_4: 0.0487 - dense_3_acc_5: 0.8689 - dense_3_acc_6: 0.0472 - dense_3_acc_7: 0.0079 - dense_3_acc_8: 0.8760 - dense_3_acc_9: 0.0891 - dense_3_acc_10: 0.053 - ETA: 17s - loss: 20.2397 - dense_3_loss_1: 1.8636 - dense_3_loss_2: 1.5496 - dense_3_loss_3: 2.0924 - dense_3_loss_4: 2.8459 - dense_3_loss_5: 1.2892 - dense_3_loss_6: 1.7287 - dense_3_loss_7: 2.7369 - dense_3_loss_8: 1.4109 - dense_3_loss_9: 1.9407 - dense_3_loss_10: 2.7819 - dense_3_acc_1: 0.2183 - dense_3_acc_2: 0.4548 - dense_3_acc_3: 0.2171 - dense_3_acc_4: 0.0477 - dense_3_acc_5: 0.8717 - dense_3_acc_6: 0.0463 - dense_3_acc_7: 0.0077 - dense_3_acc_8: 0.8777 - dense_3_acc_9: 0.0915 - dense_3_acc_10: 0.0550 6400/10000 [==================>...........] - ETA: 16s - loss: 20.1808 - dense_3_loss_1: 1.8486 - dense_3_loss_2: 1.5354 - dense_3_loss_3: 2.0828 - dense_3_loss_4: 2.8442 - dense_3_loss_5: 1.2790 - dense_3_loss_6: 1.7250 - dense_3_loss_7: 2.7405 - dense_3_loss_8: 1.4074 - dense_3_loss_9: 1.9381 - dense_3_loss_10: 2.7795 - dense_3_acc_1: 0.2251 - dense_3_acc_2: 0.4616 - dense_3_acc_3: 0.2216 - dense_3_acc_4: 0.0467 - dense_3_acc_5: 0.8743 - dense_3_acc_6: 0.0453 - dense_3_acc_7: 0.0076 - dense_3_acc_8: 0.8800 - dense_3_acc_9: 0.0941 - dense_3_acc_10: 0.057 - ETA: 15s - loss: 20.1248 - dense_3_loss_1: 1.8316 - dense_3_loss_2: 1.5213 - dense_3_loss_3: 2.0756 - dense_3_loss_4: 2.8453 - dense_3_loss_5: 1.2697 - dense_3_loss_6: 1.7202 - dense_3_loss_7: 2.7428 - dense_3_loss_8: 1.4061 - dense_3_loss_9: 1.9343 - dense_3_loss_10: 2.7779 - dense_3_acc_1: 0.2334 - dense_3_acc_2: 0.4692 - dense_3_acc_3: 0.2246 - dense_3_acc_4: 0.0464 - dense_3_acc_5: 0.8768 - dense_3_acc_6: 0.0444 - dense_3_acc_7: 0.0074 - dense_3_acc_8: 0.8804 - dense_3_acc_9: 0.1006 - dense_3_acc_10: 0.058 - ETA: 15s - loss: 20.0638 - dense_3_loss_1: 1.8135 - dense_3_loss_2: 1.5066 - dense_3_loss_3: 2.0689 - dense_3_loss_4: 2.8420 - dense_3_loss_5: 1.2604 - dense_3_loss_6: 1.7150 - dense_3_loss_7: 2.7466 - dense_3_loss_8: 1.4048 - dense_3_loss_9: 1.9303 - dense_3_loss_10: 2.7757 - dense_3_acc_1: 0.2420 - dense_3_acc_2: 0.4767 - dense_3_acc_3: 0.2261 - dense_3_acc_4: 0.0465 - dense_3_acc_5: 0.8792 - dense_3_acc_6: 0.0435 - dense_3_acc_7: 0.0073 - dense_3_acc_8: 0.8804 - dense_3_acc_9: 0.1055 - dense_3_acc_10: 0.059 - ETA: 14s - loss: 19.9987 - dense_3_loss_1: 1.7967 - dense_3_loss_2: 1.4932 - dense_3_loss_3: 2.0610 - dense_3_loss_4: 2.8407 - dense_3_loss_5: 1.2505 - dense_3_loss_6: 1.7098 - dense_3_loss_7: 2.7460 - dense_3_loss_8: 1.4011 - dense_3_loss_9: 1.9258 - dense_3_loss_10: 2.7740 - dense_3_acc_1: 0.2494 - dense_3_acc_2: 0.4819 - dense_3_acc_3: 0.2288 - dense_3_acc_4: 0.0469 - dense_3_acc_5: 0.8815 - dense_3_acc_6: 0.0435 - dense_3_acc_7: 0.0071 - dense_3_acc_8: 0.8823 - dense_3_acc_9: 0.1085 - dense_3_acc_10: 0.060 - ETA: 14s - loss: 19.9341 - dense_3_loss_1: 1.7805 - dense_3_loss_2: 1.4800 - dense_3_loss_3: 2.0529 - dense_3_loss_4: 2.8408 - dense_3_loss_5: 1.2396 - dense_3_loss_6: 1.7037 - dense_3_loss_7: 2.7464 - dense_3_loss_8: 1.3956 - dense_3_loss_9: 1.9230 - dense_3_loss_10: 2.7715 - dense_3_acc_1: 0.2560 - dense_3_acc_2: 0.4866 - dense_3_acc_3: 0.2321 - dense_3_acc_4: 0.0468 - dense_3_acc_5: 0.8838 - dense_3_acc_6: 0.0451 - dense_3_acc_7: 0.0070 - dense_3_acc_8: 0.8842 - dense_3_acc_9: 0.1085 - dense_3_acc_10: 0.062 - ETA: 13s - loss: 19.8715 - dense_3_loss_1: 1.7645 - dense_3_loss_2: 1.4671 - dense_3_loss_3: 2.0455 - dense_3_loss_4: 2.8401 - dense_3_loss_5: 1.2290 - dense_3_loss_6: 1.6957 - dense_3_loss_7: 2.7502 - dense_3_loss_8: 1.3905 - dense_3_loss_9: 1.9199 - dense_3_loss_10: 2.7689 - dense_3_acc_1: 0.2624 - dense_3_acc_2: 0.4917 - dense_3_acc_3: 0.2339 - dense_3_acc_4: 0.0461 - dense_3_acc_5: 0.8859 - dense_3_acc_6: 0.0524 - dense_3_acc_7: 0.0072 - dense_3_acc_8: 0.8861 - dense_3_acc_9: 0.1128 - dense_3_acc_10: 0.063 - ETA: 13s - loss: 19.7999 - dense_3_loss_1: 1.7456 - dense_3_loss_2: 1.4528 - dense_3_loss_3: 2.0358 - dense_3_loss_4: 2.8391 - dense_3_loss_5: 1.2194 - dense_3_loss_6: 1.6859 - dense_3_loss_7: 2.7515 - dense_3_loss_8: 1.3843 - dense_3_loss_9: 1.9161 - dense_3_loss_10: 2.7694 - dense_3_acc_1: 0.2711 - dense_3_acc_2: 0.4989 - dense_3_acc_3: 0.2355 - dense_3_acc_4: 0.0453 - dense_3_acc_5: 0.8880 - dense_3_acc_6: 0.0644 - dense_3_acc_7: 0.0089 - dense_3_acc_8: 0.8875 - dense_3_acc_9: 0.1182 - dense_3_acc_10: 0.065 - ETA: 12s - loss: 19.7282 - dense_3_loss_1: 1.7291 - dense_3_loss_2: 1.4399 - dense_3_loss_3: 2.0270 - dense_3_loss_4: 2.8349 - dense_3_loss_5: 1.2107 - dense_3_loss_6: 1.6733 - dense_3_loss_7: 2.7553 - dense_3_loss_8: 1.3752 - dense_3_loss_9: 1.9120 - dense_3_loss_10: 2.7708 - dense_3_acc_1: 0.2775 - dense_3_acc_2: 0.5046 - dense_3_acc_3: 0.2388 - dense_3_acc_4: 0.0445 - dense_3_acc_5: 0.8896 - dense_3_acc_6: 0.0775 - dense_3_acc_7: 0.0100 - dense_3_acc_8: 0.8884 - dense_3_acc_9: 0.1180 - dense_3_acc_10: 0.065 - ETA: 12s - loss: 19.6591 - dense_3_loss_1: 1.7146 - dense_3_loss_2: 1.4287 - dense_3_loss_3: 2.0224 - dense_3_loss_4: 2.8315 - dense_3_loss_5: 1.2018 - dense_3_loss_6: 1.6608 - dense_3_loss_7: 2.7566 - dense_3_loss_8: 1.3638 - dense_3_loss_9: 1.9085 - dense_3_loss_10: 2.7703 - dense_3_acc_1: 0.2823 - dense_3_acc_2: 0.5082 - dense_3_acc_3: 0.2400 - dense_3_acc_4: 0.0439 - dense_3_acc_5: 0.8916 - dense_3_acc_6: 0.0895 - dense_3_acc_7: 0.0114 - dense_3_acc_8: 0.8904 - dense_3_acc_9: 0.1221 - dense_3_acc_10: 0.066 - ETA: 12s - loss: 19.5806 - dense_3_loss_1: 1.6975 - dense_3_loss_2: 1.4155 - dense_3_loss_3: 2.0153 - dense_3_loss_4: 2.8325 - dense_3_loss_5: 1.1914 - dense_3_loss_6: 1.6496 - dense_3_loss_7: 2.7576 - dense_3_loss_8: 1.3511 - dense_3_loss_9: 1.9025 - dense_3_loss_10: 2.7676 - dense_3_acc_1: 0.2890 - dense_3_acc_2: 0.5141 - dense_3_acc_3: 0.2416 - dense_3_acc_4: 0.0431 - dense_3_acc_5: 0.8934 - dense_3_acc_6: 0.1007 - dense_3_acc_7: 0.0122 - dense_3_acc_8: 0.8922 - dense_3_acc_9: 0.1279 - dense_3_acc_10: 0.067 - ETA: 11s - loss: 19.4983 - dense_3_loss_1: 1.6806 - dense_3_loss_2: 1.4030 - dense_3_loss_3: 2.0093 - dense_3_loss_4: 2.8317 - dense_3_loss_5: 1.1794 - dense_3_loss_6: 1.6368 - dense_3_loss_7: 2.7606 - dense_3_loss_8: 1.3364 - dense_3_loss_9: 1.8959 - dense_3_loss_10: 2.7647 - dense_3_acc_1: 0.2958 - dense_3_acc_2: 0.5195 - dense_3_acc_3: 0.2439 - dense_3_acc_4: 0.0425 - dense_3_acc_5: 0.8953 - dense_3_acc_6: 0.1115 - dense_3_acc_7: 0.0129 - dense_3_acc_8: 0.8941 - dense_3_acc_9: 0.1329 - dense_3_acc_10: 0.067 - ETA: 11s - loss: 19.4101 - dense_3_loss_1: 1.6639 - dense_3_loss_2: 1.3902 - dense_3_loss_3: 2.0027 - dense_3_loss_4: 2.8286 - dense_3_loss_5: 1.1664 - dense_3_loss_6: 1.6241 - dense_3_loss_7: 2.7610 - dense_3_loss_8: 1.3212 - dense_3_loss_9: 1.8903 - dense_3_loss_10: 2.7615 - dense_3_acc_1: 0.3022 - dense_3_acc_2: 0.5250 - dense_3_acc_3: 0.2453 - dense_3_acc_4: 0.0428 - dense_3_acc_5: 0.8970 - dense_3_acc_6: 0.1218 - dense_3_acc_7: 0.0142 - dense_3_acc_8: 0.8958 - dense_3_acc_9: 0.1367 - dense_3_acc_10: 0.068 - ETA: 10s - loss: 19.3284 - dense_3_loss_1: 1.6497 - dense_3_loss_2: 1.3791 - dense_3_loss_3: 2.0000 - dense_3_loss_4: 2.8272 - dense_3_loss_5: 1.1528 - dense_3_loss_6: 1.6113 - dense_3_loss_7: 2.7627 - dense_3_loss_8: 1.3055 - dense_3_loss_9: 1.8848 - dense_3_loss_10: 2.7553 - dense_3_acc_1: 0.3064 - dense_3_acc_2: 0.5295 - dense_3_acc_3: 0.2459 - dense_3_acc_4: 0.0425 - dense_3_acc_5: 0.8987 - dense_3_acc_6: 0.1315 - dense_3_acc_7: 0.0151 - dense_3_acc_8: 0.8975 - dense_3_acc_9: 0.1403 - dense_3_acc_10: 0.070 - ETA: 10s - loss: 19.2338 - dense_3_loss_1: 1.6337 - dense_3_loss_2: 1.3666 - dense_3_loss_3: 1.9929 - dense_3_loss_4: 2.8236 - dense_3_loss_5: 1.1388 - dense_3_loss_6: 1.6004 - dense_3_loss_7: 2.7635 - dense_3_loss_8: 1.2886 - dense_3_loss_9: 1.8768 - dense_3_loss_10: 2.7489 - dense_3_acc_1: 0.3121 - dense_3_acc_2: 0.5345 - dense_3_acc_3: 0.2481 - dense_3_acc_4: 0.0421 - dense_3_acc_5: 0.9003 - dense_3_acc_6: 0.1406 - dense_3_acc_7: 0.0158 - dense_3_acc_8: 0.8992 - dense_3_acc_9: 0.1452 - dense_3_acc_10: 0.071 - ETA: 9s - loss: 19.1404 - dense_3_loss_1: 1.6180 - dense_3_loss_2: 1.3543 - dense_3_loss_3: 1.9855 - dense_3_loss_4: 2.8196 - dense_3_loss_5: 1.1248 - dense_3_loss_6: 1.5886 - dense_3_loss_7: 2.7638 - dense_3_loss_8: 1.2718 - dense_3_loss_9: 1.8707 - dense_3_loss_10: 2.7432 - dense_3_acc_1: 0.3173 - dense_3_acc_2: 0.5398 - dense_3_acc_3: 0.2508 - dense_3_acc_4: 0.0430 - dense_3_acc_5: 0.9019 - dense_3_acc_6: 0.1495 - dense_3_acc_7: 0.0163 - dense_3_acc_8: 0.9008 - dense_3_acc_9: 0.1484 - dense_3_acc_10: 0.072 - ETA: 9s - loss: 19.0492 - dense_3_loss_1: 1.6037 - dense_3_loss_2: 1.3435 - dense_3_loss_3: 1.9802 - dense_3_loss_4: 2.8163 - dense_3_loss_5: 1.1106 - dense_3_loss_6: 1.5763 - dense_3_loss_7: 2.7619 - dense_3_loss_8: 1.2549 - dense_3_loss_9: 1.8648 - dense_3_loss_10: 2.7371 - dense_3_acc_1: 0.3217 - dense_3_acc_2: 0.5447 - dense_3_acc_3: 0.2522 - dense_3_acc_4: 0.0433 - dense_3_acc_5: 0.9034 - dense_3_acc_6: 0.1588 - dense_3_acc_7: 0.0163 - dense_3_acc_8: 0.9023 - dense_3_acc_9: 0.1530 - dense_3_acc_10: 0.0739 8000/10000 [=======================>......] - ETA: 9s - loss: 18.9504 - dense_3_loss_1: 1.5888 - dense_3_loss_2: 1.3321 - dense_3_loss_3: 1.9724 - dense_3_loss_4: 2.8125 - dense_3_loss_5: 1.0965 - dense_3_loss_6: 1.5641 - dense_3_loss_7: 2.7574 - dense_3_loss_8: 1.2384 - dense_3_loss_9: 1.8572 - dense_3_loss_10: 2.7310 - dense_3_acc_1: 0.3272 - dense_3_acc_2: 0.5502 - dense_3_acc_3: 0.2551 - dense_3_acc_4: 0.0434 - dense_3_acc_5: 0.9049 - dense_3_acc_6: 0.1683 - dense_3_acc_7: 0.0166 - dense_3_acc_8: 0.9038 - dense_3_acc_9: 0.1568 - dense_3_acc_10: 0.07 - ETA: 8s - loss: 18.8554 - dense_3_loss_1: 1.5760 - dense_3_loss_2: 1.3225 - dense_3_loss_3: 1.9655 - dense_3_loss_4: 2.8070 - dense_3_loss_5: 1.0824 - dense_3_loss_6: 1.5524 - dense_3_loss_7: 2.7529 - dense_3_loss_8: 1.2220 - dense_3_loss_9: 1.8499 - dense_3_loss_10: 2.7246 - dense_3_acc_1: 0.3300 - dense_3_acc_2: 0.5539 - dense_3_acc_3: 0.2571 - dense_3_acc_4: 0.0453 - dense_3_acc_5: 0.9064 - dense_3_acc_6: 0.1768 - dense_3_acc_7: 0.0183 - dense_3_acc_8: 0.9053 - dense_3_acc_9: 0.1606 - dense_3_acc_10: 0.07 - ETA: 8s - loss: 18.7602 - dense_3_loss_1: 1.5623 - dense_3_loss_2: 1.3123 - dense_3_loss_3: 1.9608 - dense_3_loss_4: 2.8016 - dense_3_loss_5: 1.0684 - dense_3_loss_6: 1.5397 - dense_3_loss_7: 2.7484 - dense_3_loss_8: 1.2057 - dense_3_loss_9: 1.8425 - dense_3_loss_10: 2.7184 - dense_3_acc_1: 0.3360 - dense_3_acc_2: 0.5588 - dense_3_acc_3: 0.2579 - dense_3_acc_4: 0.0457 - dense_3_acc_5: 0.9078 - dense_3_acc_6: 0.1857 - dense_3_acc_7: 0.0200 - dense_3_acc_8: 0.9067 - dense_3_acc_9: 0.1640 - dense_3_acc_10: 0.07 - ETA: 8s - loss: 18.6657 - dense_3_loss_1: 1.5495 - dense_3_loss_2: 1.3021 - dense_3_loss_3: 1.9540 - dense_3_loss_4: 2.7975 - dense_3_loss_5: 1.0543 - dense_3_loss_6: 1.5275 - dense_3_loss_7: 2.7454 - dense_3_loss_8: 1.1898 - dense_3_loss_9: 1.8334 - dense_3_loss_10: 2.7122 - dense_3_acc_1: 0.3404 - dense_3_acc_2: 0.5637 - dense_3_acc_3: 0.2600 - dense_3_acc_4: 0.0465 - dense_3_acc_5: 0.9091 - dense_3_acc_6: 0.1932 - dense_3_acc_7: 0.0212 - dense_3_acc_8: 0.9081 - dense_3_acc_9: 0.1668 - dense_3_acc_10: 0.07 - ETA: 7s - loss: 18.5692 - dense_3_loss_1: 1.5367 - dense_3_loss_2: 1.2913 - dense_3_loss_3: 1.9451 - dense_3_loss_4: 2.7917 - dense_3_loss_5: 1.0405 - dense_3_loss_6: 1.5155 - dense_3_loss_7: 2.7404 - dense_3_loss_8: 1.1745 - dense_3_loss_9: 1.8278 - dense_3_loss_10: 2.7058 - dense_3_acc_1: 0.3459 - dense_3_acc_2: 0.5690 - dense_3_acc_3: 0.2635 - dense_3_acc_4: 0.0477 - dense_3_acc_5: 0.9104 - dense_3_acc_6: 0.2014 - dense_3_acc_7: 0.0226 - dense_3_acc_8: 0.9094 - dense_3_acc_9: 0.1687 - dense_3_acc_10: 0.07 - ETA: 7s - loss: 18.4761 - dense_3_loss_1: 1.5235 - dense_3_loss_2: 1.2810 - dense_3_loss_3: 1.9384 - dense_3_loss_4: 2.7875 - dense_3_loss_5: 1.0267 - dense_3_loss_6: 1.5021 - dense_3_loss_7: 2.7356 - dense_3_loss_8: 1.1597 - dense_3_loss_9: 1.8224 - dense_3_loss_10: 2.6992 - dense_3_acc_1: 0.3533 - dense_3_acc_2: 0.5740 - dense_3_acc_3: 0.2653 - dense_3_acc_4: 0.0481 - dense_3_acc_5: 0.9117 - dense_3_acc_6: 0.2107 - dense_3_acc_7: 0.0239 - dense_3_acc_8: 0.9107 - dense_3_acc_9: 0.1707 - dense_3_acc_10: 0.07 - ETA: 7s - loss: 18.3818 - dense_3_loss_1: 1.5101 - dense_3_loss_2: 1.2702 - dense_3_loss_3: 1.9301 - dense_3_loss_4: 2.7842 - dense_3_loss_5: 1.0133 - dense_3_loss_6: 1.4901 - dense_3_loss_7: 2.7306 - dense_3_loss_8: 1.1447 - dense_3_loss_9: 1.8154 - dense_3_loss_10: 2.6931 - dense_3_acc_1: 0.3593 - dense_3_acc_2: 0.5789 - dense_3_acc_3: 0.2676 - dense_3_acc_4: 0.0483 - dense_3_acc_5: 0.9130 - dense_3_acc_6: 0.2187 - dense_3_acc_7: 0.0259 - dense_3_acc_8: 0.9120 - dense_3_acc_9: 0.1723 - dense_3_acc_10: 0.07 - ETA: 6s - loss: 18.2898 - dense_3_loss_1: 1.4976 - dense_3_loss_2: 1.2593 - dense_3_loss_3: 1.9220 - dense_3_loss_4: 2.7816 - dense_3_loss_5: 1.0000 - dense_3_loss_6: 1.4779 - dense_3_loss_7: 2.7264 - dense_3_loss_8: 1.1298 - dense_3_loss_9: 1.8079 - dense_3_loss_10: 2.6873 - dense_3_acc_1: 0.3667 - dense_3_acc_2: 0.5836 - dense_3_acc_3: 0.2703 - dense_3_acc_4: 0.0487 - dense_3_acc_5: 0.9142 - dense_3_acc_6: 0.2265 - dense_3_acc_7: 0.0271 - dense_3_acc_8: 0.9132 - dense_3_acc_9: 0.1757 - dense_3_acc_10: 0.07 - ETA: 6s - loss: 18.1983 - dense_3_loss_1: 1.4852 - dense_3_loss_2: 1.2481 - dense_3_loss_3: 1.9149 - dense_3_loss_4: 2.7765 - dense_3_loss_5: 0.9870 - dense_3_loss_6: 1.4660 - dense_3_loss_7: 2.7218 - dense_3_loss_8: 1.1153 - dense_3_loss_9: 1.8011 - dense_3_loss_10: 2.6822 - dense_3_acc_1: 0.3741 - dense_3_acc_2: 0.5885 - dense_3_acc_3: 0.2718 - dense_3_acc_4: 0.0497 - dense_3_acc_5: 0.9153 - dense_3_acc_6: 0.2340 - dense_3_acc_7: 0.0282 - dense_3_acc_8: 0.9144 - dense_3_acc_9: 0.1801 - dense_3_acc_10: 0.08 - ETA: 6s - loss: 18.1061 - dense_3_loss_1: 1.4727 - dense_3_loss_2: 1.2375 - dense_3_loss_3: 1.9076 - dense_3_loss_4: 2.7718 - dense_3_loss_5: 0.9743 - dense_3_loss_6: 1.4541 - dense_3_loss_7: 2.7162 - dense_3_loss_8: 1.1010 - dense_3_loss_9: 1.7945 - dense_3_loss_10: 2.6764 - dense_3_acc_1: 0.3814 - dense_3_acc_2: 0.5927 - dense_3_acc_3: 0.2735 - dense_3_acc_4: 0.0503 - dense_3_acc_5: 0.9165 - dense_3_acc_6: 0.2418 - dense_3_acc_7: 0.0297 - dense_3_acc_8: 0.9155 - dense_3_acc_9: 0.1832 - dense_3_acc_10: 0.08 - ETA: 6s - loss: 18.0140 - dense_3_loss_1: 1.4598 - dense_3_loss_2: 1.2271 - dense_3_loss_3: 1.9007 - dense_3_loss_4: 2.7657 - dense_3_loss_5: 0.9621 - dense_3_loss_6: 1.4433 - dense_3_loss_7: 2.7103 - dense_3_loss_8: 1.0871 - dense_3_loss_9: 1.7879 - dense_3_loss_10: 2.6701 - dense_3_acc_1: 0.3879 - dense_3_acc_2: 0.5969 - dense_3_acc_3: 0.2747 - dense_3_acc_4: 0.0509 - dense_3_acc_5: 0.9176 - dense_3_acc_6: 0.2487 - dense_3_acc_7: 0.0323 - dense_3_acc_8: 0.9167 - dense_3_acc_9: 0.1860 - dense_3_acc_10: 0.08 - ETA: 5s - loss: 17.9255 - dense_3_loss_1: 1.4468 - dense_3_loss_2: 1.2161 - dense_3_loss_3: 1.8932 - dense_3_loss_4: 2.7613 - dense_3_loss_5: 0.9500 - dense_3_loss_6: 1.4333 - dense_3_loss_7: 2.7045 - dense_3_loss_8: 1.0736 - dense_3_loss_9: 1.7810 - dense_3_loss_10: 2.6658 - dense_3_acc_1: 0.3945 - dense_3_acc_2: 0.6013 - dense_3_acc_3: 0.2766 - dense_3_acc_4: 0.0513 - dense_3_acc_5: 0.9187 - dense_3_acc_6: 0.2550 - dense_3_acc_7: 0.0351 - dense_3_acc_8: 0.9178 - dense_3_acc_9: 0.1889 - dense_3_acc_10: 0.08 - ETA: 5s - loss: 17.8414 - dense_3_loss_1: 1.4357 - dense_3_loss_2: 1.2062 - dense_3_loss_3: 1.8860 - dense_3_loss_4: 2.7562 - dense_3_loss_5: 0.9381 - dense_3_loss_6: 1.4240 - dense_3_loss_7: 2.6982 - dense_3_loss_8: 1.0607 - dense_3_loss_9: 1.7750 - dense_3_loss_10: 2.6612 - dense_3_acc_1: 0.4013 - dense_3_acc_2: 0.6055 - dense_3_acc_3: 0.2787 - dense_3_acc_4: 0.0525 - dense_3_acc_5: 0.9197 - dense_3_acc_6: 0.2614 - dense_3_acc_7: 0.0366 - dense_3_acc_8: 0.9188 - dense_3_acc_9: 0.1926 - dense_3_acc_10: 0.08 - ETA: 5s - loss: 17.7570 - dense_3_loss_1: 1.4246 - dense_3_loss_2: 1.1975 - dense_3_loss_3: 1.8797 - dense_3_loss_4: 2.7503 - dense_3_loss_5: 0.9264 - dense_3_loss_6: 1.4137 - dense_3_loss_7: 2.6916 - dense_3_loss_8: 1.0481 - dense_3_loss_9: 1.7694 - dense_3_loss_10: 2.6556 - dense_3_acc_1: 0.4068 - dense_3_acc_2: 0.6085 - dense_3_acc_3: 0.2803 - dense_3_acc_4: 0.0540 - dense_3_acc_5: 0.9208 - dense_3_acc_6: 0.2681 - dense_3_acc_7: 0.0390 - dense_3_acc_8: 0.9199 - dense_3_acc_9: 0.1955 - dense_3_acc_10: 0.08 - ETA: 4s - loss: 17.6791 - dense_3_loss_1: 1.4132 - dense_3_loss_2: 1.1874 - dense_3_loss_3: 1.8727 - dense_3_loss_4: 2.7458 - dense_3_loss_5: 0.9151 - dense_3_loss_6: 1.4035 - dense_3_loss_7: 2.6866 - dense_3_loss_8: 1.0356 - dense_3_loss_9: 1.7671 - dense_3_loss_10: 2.6521 - dense_3_acc_1: 0.4124 - dense_3_acc_2: 0.6124 - dense_3_acc_3: 0.2816 - dense_3_acc_4: 0.0544 - dense_3_acc_5: 0.9218 - dense_3_acc_6: 0.2744 - dense_3_acc_7: 0.0401 - dense_3_acc_8: 0.9209 - dense_3_acc_9: 0.1972 - dense_3_acc_10: 0.08 - ETA: 4s - loss: 17.5956 - dense_3_loss_1: 1.4016 - dense_3_loss_2: 1.1769 - dense_3_loss_3: 1.8652 - dense_3_loss_4: 2.7414 - dense_3_loss_5: 0.9039 - dense_3_loss_6: 1.3942 - dense_3_loss_7: 2.6804 - dense_3_loss_8: 1.0233 - dense_3_loss_9: 1.7616 - dense_3_loss_10: 2.6470 - dense_3_acc_1: 0.4183 - dense_3_acc_2: 0.6166 - dense_3_acc_3: 0.2835 - dense_3_acc_4: 0.0546 - dense_3_acc_5: 0.9228 - dense_3_acc_6: 0.2802 - dense_3_acc_7: 0.0422 - dense_3_acc_8: 0.9219 - dense_3_acc_9: 0.1995 - dense_3_acc_10: 0.0856 9600/10000 [===========================>..] - ETA: 4s - loss: 17.5139 - dense_3_loss_1: 1.3906 - dense_3_loss_2: 1.1668 - dense_3_loss_3: 1.8582 - dense_3_loss_4: 2.7362 - dense_3_loss_5: 0.8931 - dense_3_loss_6: 1.3831 - dense_3_loss_7: 2.6754 - dense_3_loss_8: 1.0113 - dense_3_loss_9: 1.7562 - dense_3_loss_10: 2.6430 - dense_3_acc_1: 0.4241 - dense_3_acc_2: 0.6204 - dense_3_acc_3: 0.2851 - dense_3_acc_4: 0.0558 - dense_3_acc_5: 0.9237 - dense_3_acc_6: 0.2873 - dense_3_acc_7: 0.0435 - dense_3_acc_8: 0.9228 - dense_3_acc_9: 0.2014 - dense_3_acc_10: 0.08 - ETA: 4s - loss: 17.4347 - dense_3_loss_1: 1.3802 - dense_3_loss_2: 1.1560 - dense_3_loss_3: 1.8502 - dense_3_loss_4: 2.7318 - dense_3_loss_5: 0.8825 - dense_3_loss_6: 1.3753 - dense_3_loss_7: 2.6701 - dense_3_loss_8: 0.9995 - dense_3_loss_9: 1.7503 - dense_3_loss_10: 2.6389 - dense_3_acc_1: 0.4302 - dense_3_acc_2: 0.6246 - dense_3_acc_3: 0.2877 - dense_3_acc_4: 0.0567 - dense_3_acc_5: 0.9246 - dense_3_acc_6: 0.2920 - dense_3_acc_7: 0.0451 - dense_3_acc_8: 0.9238 - dense_3_acc_9: 0.2035 - dense_3_acc_10: 0.08 - ETA: 3s - loss: 17.3564 - dense_3_loss_1: 1.3700 - dense_3_loss_2: 1.1468 - dense_3_loss_3: 1.8430 - dense_3_loss_4: 2.7274 - dense_3_loss_5: 0.8721 - dense_3_loss_6: 1.3646 - dense_3_loss_7: 2.6647 - dense_3_loss_8: 0.9879 - dense_3_loss_9: 1.7448 - dense_3_loss_10: 2.6351 - dense_3_acc_1: 0.4358 - dense_3_acc_2: 0.6278 - dense_3_acc_3: 0.2899 - dense_3_acc_4: 0.0578 - dense_3_acc_5: 0.9255 - dense_3_acc_6: 0.2984 - dense_3_acc_7: 0.0463 - dense_3_acc_8: 0.9247 - dense_3_acc_9: 0.2063 - dense_3_acc_10: 0.08 - ETA: 3s - loss: 17.2802 - dense_3_loss_1: 1.3600 - dense_3_loss_2: 1.1375 - dense_3_loss_3: 1.8366 - dense_3_loss_4: 2.7215 - dense_3_loss_5: 0.8621 - dense_3_loss_6: 1.3558 - dense_3_loss_7: 2.6588 - dense_3_loss_8: 0.9767 - dense_3_loss_9: 1.7408 - dense_3_loss_10: 2.6303 - dense_3_acc_1: 0.4413 - dense_3_acc_2: 0.6314 - dense_3_acc_3: 0.2914 - dense_3_acc_4: 0.0586 - dense_3_acc_5: 0.9264 - dense_3_acc_6: 0.3036 - dense_3_acc_7: 0.0488 - dense_3_acc_8: 0.9256 - dense_3_acc_9: 0.2089 - dense_3_acc_10: 0.08 - ETA: 3s - loss: 17.2061 - dense_3_loss_1: 1.3496 - dense_3_loss_2: 1.1279 - dense_3_loss_3: 1.8297 - dense_3_loss_4: 2.7174 - dense_3_loss_5: 0.8522 - dense_3_loss_6: 1.3478 - dense_3_loss_7: 2.6531 - dense_3_loss_8: 0.9657 - dense_3_loss_9: 1.7356 - dense_3_loss_10: 2.6270 - dense_3_acc_1: 0.4468 - dense_3_acc_2: 0.6354 - dense_3_acc_3: 0.2931 - dense_3_acc_4: 0.0594 - dense_3_acc_5: 0.9273 - dense_3_acc_6: 0.3081 - dense_3_acc_7: 0.0505 - dense_3_acc_8: 0.9265 - dense_3_acc_9: 0.2115 - dense_3_acc_10: 0.08 - ETA: 3s - loss: 17.1318 - dense_3_loss_1: 1.3396 - dense_3_loss_2: 1.1188 - dense_3_loss_3: 1.8239 - dense_3_loss_4: 2.7118 - dense_3_loss_5: 0.8426 - dense_3_loss_6: 1.3397 - dense_3_loss_7: 2.6482 - dense_3_loss_8: 0.9550 - dense_3_loss_9: 1.7302 - dense_3_loss_10: 2.6220 - dense_3_acc_1: 0.4520 - dense_3_acc_2: 0.6388 - dense_3_acc_3: 0.2941 - dense_3_acc_4: 0.0606 - dense_3_acc_5: 0.9281 - dense_3_acc_6: 0.3128 - dense_3_acc_7: 0.0524 - dense_3_acc_8: 0.9273 - dense_3_acc_9: 0.2134 - dense_3_acc_10: 0.08 - ETA: 2s - loss: 17.0567 - dense_3_loss_1: 1.3292 - dense_3_loss_2: 1.1093 - dense_3_loss_3: 1.8168 - dense_3_loss_4: 2.7070 - dense_3_loss_5: 0.8332 - dense_3_loss_6: 1.3324 - dense_3_loss_7: 2.6425 - dense_3_loss_8: 0.9445 - dense_3_loss_9: 1.7249 - dense_3_loss_10: 2.6170 - dense_3_acc_1: 0.4576 - dense_3_acc_2: 0.6423 - dense_3_acc_3: 0.2960 - dense_3_acc_4: 0.0617 - dense_3_acc_5: 0.9290 - dense_3_acc_6: 0.3168 - dense_3_acc_7: 0.0534 - dense_3_acc_8: 0.9282 - dense_3_acc_9: 0.2148 - dense_3_acc_10: 0.09 - ETA: 2s - loss: 16.9840 - dense_3_loss_1: 1.3192 - dense_3_loss_2: 1.0996 - dense_3_loss_3: 1.8102 - dense_3_loss_4: 2.7028 - dense_3_loss_5: 0.8241 - dense_3_loss_6: 1.3239 - dense_3_loss_7: 2.6360 - dense_3_loss_8: 0.9344 - dense_3_loss_9: 1.7203 - dense_3_loss_10: 2.6136 - dense_3_acc_1: 0.4631 - dense_3_acc_2: 0.6461 - dense_3_acc_3: 0.2970 - dense_3_acc_4: 0.0624 - dense_3_acc_5: 0.9298 - dense_3_acc_6: 0.3212 - dense_3_acc_7: 0.0555 - dense_3_acc_8: 0.9290 - dense_3_acc_9: 0.2165 - dense_3_acc_10: 0.09 - ETA: 2s - loss: 16.9097 - dense_3_loss_1: 1.3094 - dense_3_loss_2: 1.0905 - dense_3_loss_3: 1.8034 - dense_3_loss_4: 2.6981 - dense_3_loss_5: 0.8151 - dense_3_loss_6: 1.3146 - dense_3_loss_7: 2.6308 - dense_3_loss_8: 0.9242 - dense_3_loss_9: 1.7142 - dense_3_loss_10: 2.6094 - dense_3_acc_1: 0.4678 - dense_3_acc_2: 0.6494 - dense_3_acc_3: 0.2990 - dense_3_acc_4: 0.0633 - dense_3_acc_5: 0.9306 - dense_3_acc_6: 0.3263 - dense_3_acc_7: 0.0570 - dense_3_acc_8: 0.9298 - dense_3_acc_9: 0.2191 - dense_3_acc_10: 0.09 - ETA: 2s - loss: 16.8394 - dense_3_loss_1: 1.2993 - dense_3_loss_2: 1.0816 - dense_3_loss_3: 1.7976 - dense_3_loss_4: 2.6938 - dense_3_loss_5: 0.8063 - dense_3_loss_6: 1.3066 - dense_3_loss_7: 2.6256 - dense_3_loss_8: 0.9142 - dense_3_loss_9: 1.7079 - dense_3_loss_10: 2.6064 - dense_3_acc_1: 0.4729 - dense_3_acc_2: 0.6527 - dense_3_acc_3: 0.3000 - dense_3_acc_4: 0.0639 - dense_3_acc_5: 0.9313 - dense_3_acc_6: 0.3307 - dense_3_acc_7: 0.0589 - dense_3_acc_8: 0.9306 - dense_3_acc_9: 0.2213 - dense_3_acc_10: 0.09 - ETA: 1s - loss: 16.7707 - dense_3_loss_1: 1.2901 - dense_3_loss_2: 1.0730 - dense_3_loss_3: 1.7916 - dense_3_loss_4: 2.6885 - dense_3_loss_5: 0.7978 - dense_3_loss_6: 1.2981 - dense_3_loss_7: 2.6208 - dense_3_loss_8: 0.9045 - dense_3_loss_9: 1.7033 - dense_3_loss_10: 2.6031 - dense_3_acc_1: 0.4774 - dense_3_acc_2: 0.6555 - dense_3_acc_3: 0.3005 - dense_3_acc_4: 0.0649 - dense_3_acc_5: 0.9321 - dense_3_acc_6: 0.3360 - dense_3_acc_7: 0.0601 - dense_3_acc_8: 0.9313 - dense_3_acc_9: 0.2219 - dense_3_acc_10: 0.09 - ETA: 1s - loss: 16.7024 - dense_3_loss_1: 1.2808 - dense_3_loss_2: 1.0639 - dense_3_loss_3: 1.7848 - dense_3_loss_4: 2.6843 - dense_3_loss_5: 0.7895 - dense_3_loss_6: 1.2906 - dense_3_loss_7: 2.6155 - dense_3_loss_8: 0.8949 - dense_3_loss_9: 1.6983 - dense_3_loss_10: 2.5995 - dense_3_acc_1: 0.4822 - dense_3_acc_2: 0.6588 - dense_3_acc_3: 0.3021 - dense_3_acc_4: 0.0658 - dense_3_acc_5: 0.9328 - dense_3_acc_6: 0.3401 - dense_3_acc_7: 0.0623 - dense_3_acc_8: 0.9321 - dense_3_acc_9: 0.2235 - dense_3_acc_10: 0.09 - ETA: 1s - loss: 16.6357 - dense_3_loss_1: 1.2713 - dense_3_loss_2: 1.0554 - dense_3_loss_3: 1.7779 - dense_3_loss_4: 2.6797 - dense_3_loss_5: 0.7812 - dense_3_loss_6: 1.2826 - dense_3_loss_7: 2.6100 - dense_3_loss_8: 0.8856 - dense_3_loss_9: 1.6961 - dense_3_loss_10: 2.5960 - dense_3_acc_1: 0.4872 - dense_3_acc_2: 0.6616 - dense_3_acc_3: 0.3044 - dense_3_acc_4: 0.0672 - dense_3_acc_5: 0.9335 - dense_3_acc_6: 0.3445 - dense_3_acc_7: 0.0644 - dense_3_acc_8: 0.9328 - dense_3_acc_9: 0.2244 - dense_3_acc_10: 0.09 - ETA: 1s - loss: 16.5697 - dense_3_loss_1: 1.2621 - dense_3_loss_2: 1.0473 - dense_3_loss_3: 1.7716 - dense_3_loss_4: 2.6753 - dense_3_loss_5: 0.7732 - dense_3_loss_6: 1.2752 - dense_3_loss_7: 2.6061 - dense_3_loss_8: 0.8764 - dense_3_loss_9: 1.6906 - dense_3_loss_10: 2.5919 - dense_3_acc_1: 0.4916 - dense_3_acc_2: 0.6646 - dense_3_acc_3: 0.3065 - dense_3_acc_4: 0.0684 - dense_3_acc_5: 0.9343 - dense_3_acc_6: 0.3487 - dense_3_acc_7: 0.0654 - dense_3_acc_8: 0.9335 - dense_3_acc_9: 0.2262 - dense_3_acc_10: 0.09 - ETA: 1s - loss: 16.5037 - dense_3_loss_1: 1.2523 - dense_3_loss_2: 1.0385 - dense_3_loss_3: 1.7651 - dense_3_loss_4: 2.6716 - dense_3_loss_5: 0.7653 - dense_3_loss_6: 1.2675 - dense_3_loss_7: 2.6015 - dense_3_loss_8: 0.8675 - dense_3_loss_9: 1.6854 - dense_3_loss_10: 2.5891 - dense_3_acc_1: 0.4966 - dense_3_acc_2: 0.6678 - dense_3_acc_3: 0.3084 - dense_3_acc_4: 0.0691 - dense_3_acc_5: 0.9349 - dense_3_acc_6: 0.3534 - dense_3_acc_7: 0.0665 - dense_3_acc_8: 0.9342 - dense_3_acc_9: 0.2288 - dense_3_acc_10: 0.09 - ETA: 0s - loss: 16.4380 - dense_3_loss_1: 1.2426 - dense_3_loss_2: 1.0298 - dense_3_loss_3: 1.7590 - dense_3_loss_4: 2.6674 - dense_3_loss_5: 0.7575 - dense_3_loss_6: 1.2595 - dense_3_loss_7: 2.5968 - dense_3_loss_8: 0.8587 - dense_3_loss_9: 1.6805 - dense_3_loss_10: 2.5863 - dense_3_acc_1: 0.5015 - dense_3_acc_2: 0.6708 - dense_3_acc_3: 0.3103 - dense_3_acc_4: 0.0696 - dense_3_acc_5: 0.9356 - dense_3_acc_6: 0.3577 - dense_3_acc_7: 0.0677 - dense_3_acc_8: 0.9349 - dense_3_acc_9: 0.2312 - dense_3_acc_10: 0.094110000/10000 [==============================] - ETA: 0s - loss: 16.3737 - dense_3_loss_1: 1.2333 - dense_3_loss_2: 1.0211 - dense_3_loss_3: 1.7526 - dense_3_loss_4: 2.6640 - dense_3_loss_5: 0.7498 - dense_3_loss_6: 1.2508 - dense_3_loss_7: 2.5925 - dense_3_loss_8: 0.8501 - dense_3_loss_9: 1.6760 - dense_3_loss_10: 2.5834 - dense_3_acc_1: 0.5063 - dense_3_acc_2: 0.6739 - dense_3_acc_3: 0.3122 - dense_3_acc_4: 0.0705 - dense_3_acc_5: 0.9363 - dense_3_acc_6: 0.3627 - dense_3_acc_7: 0.0682 - dense_3_acc_8: 0.9356 - dense_3_acc_9: 0.2329 - dense_3_acc_10: 0.09 - ETA: 0s - loss: 16.3126 - dense_3_loss_1: 1.2241 - dense_3_loss_2: 1.0127 - dense_3_loss_3: 1.7469 - dense_3_loss_4: 2.6610 - dense_3_loss_5: 0.7424 - dense_3_loss_6: 1.2442 - dense_3_loss_7: 2.5886 - dense_3_loss_8: 0.8417 - dense_3_loss_9: 1.6708 - dense_3_loss_10: 2.5803 - dense_3_acc_1: 0.5109 - dense_3_acc_2: 0.6769 - dense_3_acc_3: 0.3135 - dense_3_acc_4: 0.0708 - dense_3_acc_5: 0.9369 - dense_3_acc_6: 0.3660 - dense_3_acc_7: 0.0693 - dense_3_acc_8: 0.9362 - dense_3_acc_9: 0.2350 - dense_3_acc_10: 0.09 - ETA: 0s - loss: 16.2541 - dense_3_loss_1: 1.2160 - dense_3_loss_2: 1.0061 - dense_3_loss_3: 1.7407 - dense_3_loss_4: 2.6575 - dense_3_loss_5: 0.7351 - dense_3_loss_6: 1.2370 - dense_3_loss_7: 2.5843 - dense_3_loss_8: 0.8334 - dense_3_loss_9: 1.6670 - dense_3_loss_10: 2.5769 - dense_3_acc_1: 0.5148 - dense_3_acc_2: 0.6792 - dense_3_acc_3: 0.3154 - dense_3_acc_4: 0.0711 - dense_3_acc_5: 0.9376 - dense_3_acc_6: 0.3696 - dense_3_acc_7: 0.0698 - dense_3_acc_8: 0.9369 - dense_3_acc_9: 0.2364 - dense_3_acc_10: 0.09 - 20s 2ms/step - loss: 16.1926 - dense_3_loss_1: 1.2069 - dense_3_loss_2: 0.9976 - dense_3_loss_3: 1.7341 - dense_3_loss_4: 2.6537 - dense_3_loss_5: 0.7279 - dense_3_loss_6: 1.2297 - dense_3_loss_7: 2.5799 - dense_3_loss_8: 0.8255 - dense_3_loss_9: 1.6625 - dense_3_loss_10: 2.5748 - dense_3_acc_1: 0.5195 - dense_3_acc_2: 0.6824 - dense_3_acc_3: 0.3172 - dense_3_acc_4: 0.0715 - dense_3_acc_5: 0.9382 - dense_3_acc_6: 0.3739 - dense_3_acc_7: 0.0704 - dense_3_acc_8: 0.9375 - dense_3_acc_9: 0.2374 - dense_3_acc_10: 0.0964
    




    <keras.callbacks.History at 0x20d7f3e8dd8>



While training you can see the loss as well as the accuracy on each of the 10 positions of the output. The table below gives you an example of what the accuracies could be if the batch had 2 examples: 

<img src="images/table.png" style="width:700;height:200px;"> <br>
<caption><center>Thus, `dense_2_acc_8: 0.89` means that you are predicting the 7th character of the output correctly 89% of the time in the current batch of data. </center></caption>


We have run this model for longer, and saved the weights. Run the next cell to load our weights. (By training a model for several minutes, you should be able to obtain a model of similar accuracy, but loading our model will save you time.) 


```python
model.load_weights('models/model.h5')
```

You can now see the results on new examples.


```python
EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018', 'March 3 2001', 'March 3rd 2001', '1 March 2001']
for example in EXAMPLES:
    
    source = string_to_int(example, Tx, human_vocab)
    # source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0,1)
    
    source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source)))
    source = source.reshape((1,source.shape[0],source.shape[1]))
    
    prediction = model.predict([source, s0, c0])
    prediction = np.argmax(prediction, axis = -1)
    output = [inv_machine_vocab[int(i)] for i in prediction]
    
    print("source:", example)
    print("output:", ''.join(output))
```

    source: 3 May 1979
    output: 1979-05-03
    source: 5 April 09
    output: 2009-05-05
    source: 21th of August 2016
    output: 2016-08-21
    source: Tue 10 Jul 2007
    output: 2007-07-10
    source: Saturday May 9 2018
    output: 2018-05-09
    source: March 3 2001
    output: 2001-03-03
    source: March 3rd 2001
    output: 2001-03-03
    source: 1 March 2001
    output: 2001-03-01
    

错误：ValueError: Error when checking : expected input_2 to have 3 dimensions, but got array with shape (37, 30)

https://blog.csdn.net/Exupery_/article/details/79548104 这篇 blog 中分析的原因是 Keras 的版本问题，修改原代码后可正常运行。


You can also change these examples to test with your own examples. The next part will give you a better sense on what the attention mechanism is doing--i.e., what part of the input the network is paying attention to when generating a particular output character. 


## 3 - Visualizing Attention (Optional / Ungraded)

Since the problem has a fixed output length of 10, it is also possible to carry out this task using 10 different softmax units to generate the 10 characters of the output. But one advantage of the attention model is that each part of the output (say the month) knows it needs to depend only on a small part of the input (the characters in the input giving the month). We can  visualize what part of the output is looking at what part of the input.

由于问题的输出长度固定为 10，因此也可以使用 10 个不同的 softmax 单位执行此任务以生成输出的 10 个字符。 但是，注意模型的一个优点是输出的每个部分（比如说月份）都知道它只需要依赖一小部分输入（输入给出月份的字符）。 我们可以看到输出的哪一部分正在查看输入的哪一部分。

Consider the task of translating "Saturday 9 May 2018" to "2018-05-09". If we visualize the computed $\alpha^{\langle t, t' \rangle}$ we get this: 

<img src="images/date_attention.png" style="width:600;height:300px;"> <br>
<caption><center> **Figure 8**: Full Attention Map</center></caption>

Notice how the output ignores the "Saturday" portion of the input. None of the output timesteps are paying much attention to that portion of the input. We see also that 9 has been translated as 09 and May has been correctly translated into 05, with the output paying attention to the parts of the input it needs to to make the translation. The year mostly requires it to pay attention to the input's "18" in order to generate "2018." 

注意输出如何忽略输入的“星期六”部分。 没有一个输出时间步骤对输入的那部分非常重视。 我们还看到，9被翻译为09，5月被正确翻译成05，输出注意输入翻译所需的部分。 这一年大多需要注意输入的“18”以产生“2018”。


### 3.1 - Getting the activations from the network

Lets now visualize the attention values in your network. We'll propagate an example through the network, then visualize the values of $\alpha^{\langle t, t' \rangle}$. 

To figure out where the attention values are located, let's start by printing a summary of the model .

现在让我们看到您网络中的关注值。 我们将通过网络传播一个例子，然后可视化 $\alpha^{\langle t, t' \rangle}$ 的值。

要计算注意力值的位置，我们首先打印模型的摘要。


```python
model.summary()
```

    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_2 (InputLayer)            (None, 30, 37)       0                                            
    __________________________________________________________________________________________________
    s0 (InputLayer)                 (None, 64)           0                                            
    __________________________________________________________________________________________________
    bidirectional_2 (Bidirectional) (None, 30, 64)       17920       input_2[0][0]                    
    __________________________________________________________________________________________________
    repeat_vector_1 (RepeatVector)  (None, 30, 64)       0           s0[0][0]                         
                                                                     lstm_1[0][0]                     
                                                                     lstm_1[1][0]                     
                                                                     lstm_1[2][0]                     
                                                                     lstm_1[3][0]                     
                                                                     lstm_1[4][0]                     
                                                                     lstm_1[5][0]                     
                                                                     lstm_1[6][0]                     
                                                                     lstm_1[7][0]                     
                                                                     lstm_1[8][0]                     
    __________________________________________________________________________________________________
    concatenate_1 (Concatenate)     (None, 30, 128)      0           bidirectional_2[0][0]            
                                                                     repeat_vector_1[1][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[2][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[3][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[4][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[5][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[6][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[7][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[8][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[9][0]            
                                                                     bidirectional_2[0][0]            
                                                                     repeat_vector_1[10][0]           
    __________________________________________________________________________________________________
    dense_1 (Dense)                 (None, 30, 10)       1290        concatenate_1[0][0]              
                                                                     concatenate_1[1][0]              
                                                                     concatenate_1[2][0]              
                                                                     concatenate_1[3][0]              
                                                                     concatenate_1[4][0]              
                                                                     concatenate_1[5][0]              
                                                                     concatenate_1[6][0]              
                                                                     concatenate_1[7][0]              
                                                                     concatenate_1[8][0]              
                                                                     concatenate_1[9][0]              
    __________________________________________________________________________________________________
    dense_2 (Dense)                 (None, 30, 1)        11          dense_1[0][0]                    
                                                                     dense_1[1][0]                    
                                                                     dense_1[2][0]                    
                                                                     dense_1[3][0]                    
                                                                     dense_1[4][0]                    
                                                                     dense_1[5][0]                    
                                                                     dense_1[6][0]                    
                                                                     dense_1[7][0]                    
                                                                     dense_1[8][0]                    
                                                                     dense_1[9][0]                    
    __________________________________________________________________________________________________
    attention_weights (Activation)  (None, 30, 1)        0           dense_2[0][0]                    
                                                                     dense_2[1][0]                    
                                                                     dense_2[2][0]                    
                                                                     dense_2[3][0]                    
                                                                     dense_2[4][0]                    
                                                                     dense_2[5][0]                    
                                                                     dense_2[6][0]                    
                                                                     dense_2[7][0]                    
                                                                     dense_2[8][0]                    
                                                                     dense_2[9][0]                    
    __________________________________________________________________________________________________
    dot_1 (Dot)                     (None, 1, 64)        0           attention_weights[0][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[1][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[2][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[3][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[4][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[5][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[6][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[7][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[8][0]          
                                                                     bidirectional_2[0][0]            
                                                                     attention_weights[9][0]          
                                                                     bidirectional_2[0][0]            
    __________________________________________________________________________________________________
    c0 (InputLayer)                 (None, 64)           0                                            
    __________________________________________________________________________________________________
    lstm_1 (LSTM)                   [(None, 64), (None,  33024       dot_1[0][0]                      
                                                                     s0[0][0]                         
                                                                     c0[0][0]                         
                                                                     dot_1[1][0]                      
                                                                     lstm_1[0][0]                     
                                                                     lstm_1[0][2]                     
                                                                     dot_1[2][0]                      
                                                                     lstm_1[1][0]                     
                                                                     lstm_1[1][2]                     
                                                                     dot_1[3][0]                      
                                                                     lstm_1[2][0]                     
                                                                     lstm_1[2][2]                     
                                                                     dot_1[4][0]                      
                                                                     lstm_1[3][0]                     
                                                                     lstm_1[3][2]                     
                                                                     dot_1[5][0]                      
                                                                     lstm_1[4][0]                     
                                                                     lstm_1[4][2]                     
                                                                     dot_1[6][0]                      
                                                                     lstm_1[5][0]                     
                                                                     lstm_1[5][2]                     
                                                                     dot_1[7][0]                      
                                                                     lstm_1[6][0]                     
                                                                     lstm_1[6][2]                     
                                                                     dot_1[8][0]                      
                                                                     lstm_1[7][0]                     
                                                                     lstm_1[7][2]                     
                                                                     dot_1[9][0]                      
                                                                     lstm_1[8][0]                     
                                                                     lstm_1[8][2]                     
    __________________________________________________________________________________________________
    dense_3 (Dense)                 (None, 11)           715         lstm_1[0][0]                     
                                                                     lstm_1[1][0]                     
                                                                     lstm_1[2][0]                     
                                                                     lstm_1[3][0]                     
                                                                     lstm_1[4][0]                     
                                                                     lstm_1[5][0]                     
                                                                     lstm_1[6][0]                     
                                                                     lstm_1[7][0]                     
                                                                     lstm_1[8][0]                     
                                                                     lstm_1[9][0]                     
    ==================================================================================================
    Total params: 52,960
    Trainable params: 52,960
    Non-trainable params: 0
    __________________________________________________________________________________________________
    

Navigate through the output of `model.summary()` above. You can see that the layer named `attention_weights` outputs the `alphas` of shape (m, 30, 1) before `dot_2` computes the context vector for every time step $t = 0, \ldots, T_y-1$. Lets get the activations from this layer.

The function `attention_map()` pulls out the attention values from your model and plots them.


```python
attention_map = plot_attention_map(model, human_vocab, inv_machine_vocab, "Tuesday 09 Oct 1993", num = 7, n_s = 64)
```


    <matplotlib.figure.Figure at 0x20d10aa38d0>



![png](output_42_1.png)


On the generated plot you can observe the values of the attention weights for each character of the predicted output. Examine this plot and check that where the network is paying attention makes sense to you.

In the date translation application, you will observe that most of the time attention helps predict the year, and hasn't much impact on predicting the day/month.

在生成的图上，您可以观察预测输出的每个字符的注意力权重值。 检查此图并检查网络注意力在哪里对您有意义。

在日期翻译应用程序中，您会发现大多数时间注意有助于预测年份，并且对预测日/月没有太大影响。

### Congratulations!


You have come to the end of this assignment 

<font color='blue'> **Here's what you should remember from this notebook**:

- Machine translation models can be used to map from one sequence to another. They are useful not just for translating human languages (like French->English) but also for tasks like date format translation. 
- An attention mechanism allows a network to focus on the most relevant parts of the input when producing a specific part of the output. 
- A network using an attention mechanism can translate from inputs of length $T_x$ to outputs of length $T_y$, where $T_x$ and $T_y$ can be different. 
- You can visualize attention weights $\alpha^{\langle t,t' \rangle}$ to see what the network is paying attention to while generating each output.

- 机器翻译模型可用于从一个序列映射到另一个序列。 它们不仅用于翻译人类语言（如法语 - >英语），还用于日期格式翻译等任务。

- 注意机制允许网络在产生输出的特定部分时专注于输入的最相关部分。

- 使用注意机制的网络可以从长度为 $T_x$ 的输入转换为长度为$T_y$的输出，其中 $T_x$ 和 $T_y$ 可以不同。

- 您可以将注意力权重$\alpha^{\langle t,t' \rangle}$可视化，以便在生成每个输出时查看网络正在关注的内容。


Congratulations on finishing this assignment! You are now able to implement an attention model and use it to learn complex mappings from one sequence to another. 
