Complete DavidNet code can be found at:


# Step by step DavidNet model explanation:





## Import libraries


```python
 import numpy as np
 import time, math
 from tqdm import tqdm_notebook as tqdm
 import tensorflow as tf
 import tensorflow.contrib.eager as tfe
```


​    

## Eagermode.

#### Eager execution is an programming environment that allows the users to run the code without creating graphs. Consider the below example:

```python
a = tf.constant([1,2],[3,4])

print(a) 
```

###### The output of the above code is a tensor, in order to print the values you need to make use of the tfsession.

Where as if you use eager execution like.

```python
tf.enable_eager_execution()
```

You can able to print the actual value itself.

Sources: https://www.tensorflow.org/guide/eager



## weight initialization algorithm 

Davidnet is written in PyTorch, and for weight intialization they have used "Kaiming He init". And Keras uses Xavier Glorot init, in order to make keras intialization similar to PyTorch we have used the below function.  

    def init_pytorch(shape, dtype=tf.float32, partition_info=None):
         fan = np.prod(shape[:-1])
         bound = 1 / math.sqrt(fan)
         return tf.random.uniform(shape, minval=-bound, maxval=bound, dtype=dtype)

## Keras Model subclassing 

#### In this program we have used a new type of model creation "Model subclassing"

Sources: 

https://www.pyimagesearch.com/2019/10/28/3-ways-to-create-a-keras-model-with-tensorflow-2-0-sequential-functional-and-model-subclassing/

https://www.tensorflow.org/guide/keras/custom_layers_and_models

ConvBN class comprised of 2D Convolution layer followed by BatchNormalization, Dropout & Relu activation. 

c_out: Total number of kernels. 

```python
class ConvBN(tf.keras.Model):
  def __init__(self, c_out):
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=3, padding="SAME", kernel_initializer=init_pytorch, use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)
    self.drop = tf.keras.layers.Dropout(0.05)

  def call(self, inputs):
    return tf.nn.relu(self.bn(self.drop(self.conv(inputs))))
```

## ResNET block. 

Simialr to __ConvBN__ block, ResNet is defined in the below code. 

    class ResBlk(tf.keras.Model):
      def __init__(self, c_out, pool, res = False):
        super().__init__()
        self.conv_bn = ConvBN(c_out)
        self.pool = pool
        self.res = res
        if self.res:
          self.res1 = ConvBN(c_out)
          self.res2 = ConvBN(c_out)
    
      def call(self, inputs):
        h = self.pool(self.conv_bn(inputs))
        if self.res:
          h = h + self.res2(self.res1(h))
        return h



Total numeber of kernels(c_out), pooling layer(pool) and residual connection(res) are passed as a parameter.



## DavidNet model.

Similar to ConvBN and ResBlk blocks, DavidNet class is defined as below. 


     class DavidNet(tf.keras.Model):
       def __init__(self, c=64, weight=0.125):
         super().__init__()
         pool = tf.keras.layers.MaxPooling2D()
         self.init_conv_bn = ConvBN(c)
         self.blk1 = ResBlk(c*2, pool, res = True)
         self.blk2 = ResBlk(c*4, pool)
         self.blk3 = ResBlk(c*8, pool, res = True)
         self.pool = tf.keras.layers.GlobalMaxPool2D()
         self.linear = tf.keras.layers.Dense(10, kernel_initializer=init_pytorch, use_bias=False)
         self.weight = weight
    
      def call(self, x, y):
        h = self.pool(self.blk3(self.blk2(self.blk1(self.init_conv_bn(x)))))
        h = self.linear(h) * self.weight
        ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)
        loss = tf.reduce_sum(ce)
        correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis = 1), y), tf.float32))
        return loss, correct



<img src="https://github.com/davidcpage/cifar10-fast/blob/d31ad8d393dd75147b65f261dbf78670a97e48a8/net.svg">

1. Initial ConvBN block is made up of with 64 kernels.
2. followed by a ResNet block with residual connection.
3. Again a ResNet block with 256 kernels and no residual connection. 
4. This is followed by ResNet block with 512 kernels and residual connection. 
5. At the end we have globalmMaxPooling layer and output of GlobalMaxPooling is passed to a Dense layer to get 10x1x1 output. 
6. Outputs are scaled by 0.125 and passed to a softmax layer to get the results

Note: 

```python
h = self.linear(h) * self.weight
```

* This step applies scaling to the output of dense layer. *

* self.weight is a hyperparameter which is tuned to 0.125 by DavidNet.* 

  

```
ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=h, labels=y)
```

This function computes the sparse softmax cross entropy between logits and labels. Function takes 2 parameters "predicted y values" and "actual y values".



```
loss = tf.reduce_sum(ce)
```

This step sums the error cross entropy computed in previous step.



```
correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.argmax(h, axis = 1), y), tf.float32))
```

In this step _tf.argmax()_ gets actual class id from one-hot encoded vector. 

tf.math.equal() compares the predicted and true labels.

tf.reduce_sum() is used to sum the numbers to in effect find total number of true predictions. 

call() returns loss and total correct predictions.



## Data preprocessing.

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    len_train, len_test = len(x_train), len(x_test)
    y_train = y_train.astype('int64').reshape(len_train)
    y_test = y_test.astype('int64').reshape(len_test)
    
    train_mean = np.mean(x_train, axis=(0,1,2))
    train_std = np.std(x_train, axis=(0,1,2))
    
    normalize = lambda x: ((x - train_mean) / train_std).astype('float32') # todo: check here
    pad4 = lambda x: np.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], mode='reflect')
    
    x_train = normalize(pad4(x_train))
    x_test = normalize(x_test)



We load cifar10 dataset and reshape it. 

Later we have to do padding by 4 cells. 

```python
np.pad(x, [(0, 0), (4, 4), (4, 4), (0, 0)], mode='reflect')
```

Normalize the data by mean and standard deviation.

```
train_mean = np.mean(x_train, axis=(0,1,2))
train_std = np.std(x_train, axis=(0,1,2))
```



## Learning rate scheduler

```
lr_schedule = lambda t: np.interp([t], [0, (EPOCHS+1)//5, EPOCHS], [0, LEARNING_RATE, 0])[0]
```

We are passing 3 arguments i.e

(0,0), ((EPOCHS+1)//5,LEARNING_RATE) and (EPOCHS,0) to the _np.interp()_ and expecting interpolated values of x coordinates [t]. 

Here _[t]_ represents x coordinates at which we need interpolated values. 

[0, (EPOCHS+1)//5, EPOCHS] represents known x coordinates

[0, LEARNING_RATE, 0] represents known y coordinates. 



## Momentum optimizer 

```
opt = tf.train.MomentumOptimizer(lr_func, momentum=MOMENTUM, use_nesterov=True)
```

It is a inbuilt tensorflow momentum optimizer. 



Data augmentation 



```
data_aug = lambda x, y: (tf.image.random_flip_left_right(tf.random_crop(x, [32, 32, 3])), y)
```

Random crop of 32x32x3 image and an random horizontal flips.



## Model Training

```
test_set = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(BATCH_SIZE)
```

From tensor_slices((x_test, y_test))  we will create a Dataset whose elements are slices of tensors x_test and y_test.

Similarly we get train data set from the following code: 

```
train_set = tf.data.Dataset.from_tensor_slices((x_train, y_train)).map(data_aug).shuffle(len_train).batch(BATCH_SIZE).prefetch(1)
```

Note: To speed up the training process we make use of multithreading functionality of tensorflow. 

​		prefetch() method, gets the data ready for the next step.

​		1 batch of data be prefetched when other batch of data is being consumed by training process.

Learning phase 

```
tf.keras.backend.set_learning_phase(1)
```

Weight decay:

We sum up the multiplication of weight,weight decay coefficient and BATCH_SIZE to gradients. 

    for g, v in zip(grads, var):
       g += v * WEIGHT_DECAY * BATCH_SIZE
    


Calculate test loss and accuracy:

    tf.keras.backend.set_learning_phase(0)
    for (x, y) in test_set:
      loss, correct = model(x, y)
      test_loss += loss.numpy()
      test_acc += correct.numpy()
    




