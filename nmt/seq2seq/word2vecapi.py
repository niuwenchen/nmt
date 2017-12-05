#encoding:utf-8
#@Time : 2017/12/5 15:05
#@Author : JackNiu
import tensorflow as tf
vocabulary_size=10
batch_size=10
embedding_size=20

x=[0,0,0,1,0,1,0,0,0,0]

embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
embed = tf.nn.embedding_lookup(embeddings, train_inputs)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    y=sess.run(embed,feed_dict={train_inputs:x})
    print(y[5])