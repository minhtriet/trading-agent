
# coding: utf-8

# In[1]:
import pandas as pd

import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.examples.tutorials.mnist import input_data

import numpy as np

import collections


# In[2]:
timestep = 28
n_input = 28
n_unit = 1024
n_classes = 10
batch_size = 69
lr = 0.0001


# %%
trp = pd.read_csv('ncg.csv')

# In[3]:


tf.reset_default_graph()

xx = tf.placeholder(tf.float32, shape=(None, 784))
yy = tf.placeholder(tf.float32, shape=(None, n_classes))

out_weights=tf.Variable(tf.random_normal([n_unit,10]))
out_bias=tf.Variable(tf.random_normal([10]))

output, state = rnn.static_rnn(rnn.BasicLSTMCell(n_unit, forget_bias=1), inputs=[xx], dtype=tf.float32)
predict = tf.matmul(output[-1], out_weights) + out_bias
loss = tf.losses.softmax_cross_entropy(logits=predict, onehot_labels=yy)


# In[4]:


opt = tf.train.RMSPropOptimizer(learning_rate=lr).minimize(loss)


# In[5]:


correct_prediction=tf.equal(tf.argmax(predict,1),tf.argmax(yy,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))


# In[6]:


mnist=input_data.read_data_sets("/tmp/data/",one_hot=True)


# In[7]:


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())    
    iter=1
    while iter<80:        
        batch_x,batch_y=mnist.train.next_batch(batch_size=batch_size)        
        sess.run(opt,feed_dict={xx:batch_x,yy:batch_y})        
        if iter %10==0:
            acc=sess.run(accuracy,feed_dict={xx:batch_x,yy:batch_y})
            los=sess.run(loss,feed_dict={xx:batch_x,yy:batch_y})            
            print("Accuracy %f Loss %f" % (acc, los))            
            print("__________________")

        iter=iter+1
    tf.saved_model.simple_save(sess, 'lstm_mnist', inputs={"xx": xx}, outputs={"yy": yy})


# In[10]:


with tf.Session() as sess:
    tf.saved_model.loader.load(sess=sess, export_dir='lstm_mnist')
    test_data = mnist.test.images
    test_label = mnist.test.labels
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={xx: test_data, yy: test_label}))

