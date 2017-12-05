#encoding:utf-8
#@Time : 2017/12/4 17:24
#@Author : JackNiu
import   nmt.seq2seq.helper as helper
import numpy as np
import tensorflow as tf


tf.reset_default_graph()
sess = tf.InteractiveSession()

PAD = 0
EOS = 1

vocab_size = 10
input_embedding_size = 20

encoder_hidden_units = 20
decoder_hidden_units = encoder_hidden_units

encoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
decoder_targets = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')

decoder_inputs = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_inputs')

'''
因为输入max_time和batch_size 都不知道，但是这里embedding是确定的， [10,20]
encoder_inputs  : 8*100

'''
embeddings = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)


# A `Tensor` with the same type as the tensors in `params`. 返回的格式是[10,20],encoder_inputs[8,100]
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)
decoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, decoder_inputs)


encoder_cell = tf.contrib.rnn.LSTMCell(encoder_hidden_units)

encoder_outputs, encoder_final_state = tf.nn.dynamic_rnn(
    encoder_cell, encoder_inputs_embedded,
    dtype=tf.float32, time_major=True,
)

del encoder_outputs


decoder_cell = tf.contrib.rnn.LSTMCell(decoder_hidden_units)

decoder_outputs, decoder_final_state = tf.nn.dynamic_rnn(
    decoder_cell, decoder_inputs_embedded,

    initial_state=encoder_final_state,

    dtype=tf.float32, time_major=True, scope="plain_decoder",
)

# 线性优化层，没有激活函数的全连接层
decoder_logits = tf.contrib.layers.linear(decoder_outputs, vocab_size)

# [max_time,batch_size,hidden_units] -->[max_time,batch_size,vocab_size]
# decoder_prediction 就是最后的输出吗？
decoder_prediction = tf.argmax(decoder_logits, 2)


# 计算loss: 将输入用one_hot编码
stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits,
)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess.run(tf.global_variables_initializer())

# batch_ = [[6], [3, 4], [9, 8, 7]]
#
# batch_, batch_length_ = helper.batch(batch_)
# print('batch_encoded:\n' + str(batch_))
#
# din_, dlen_ = helper.batch(np.ones(shape=(3, 1), dtype=np.int32),
#                             max_sequence_length=4)
# print('decoder inputs:\n' + str(din_))
#
# pred_ = sess.run(decoder_prediction,
#     feed_dict={
#         encoder_inputs: batch_,
#         decoder_inputs: din_,
#     })
# print('decoder predictions:\n' + str(pred_))



batch_size = 100
# 到底什么是batch_size: 一般输入训练的数据一个batch代表多个数据样本的集合，那么这里的batch_size 就是多个数据样本同时进行训练
batches = helper.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)

# print('head of the batch:')
# for seq in next(batches)[:batch_size]:
#     print(seq)

def next_feed():
    batch = next(batches)
    # 输入数据吗？构造随机数据，对这些随机数据进行包装成数组 [max_time,batch_size]
    encoder_inputs_, _ = helper.batch(batch)
    # decoder输出数据: [添加了EOS格式的输出数据]
    decoder_targets_, _ = helper.batch(
        [(sequence) + [EOS] for sequence in batch]
    )
    # decoder_input: EOS + sequence
    decoder_inputs_, _ = helper.batch(
        [[EOS] + (sequence) for sequence in batch]
    )

    '''
    输入ABC: 作为encoder的输入，在读取到EOS时终止，并输出一个向量作为”ABC”这个输入项链的语义表示向量，此过程称为”Encoder”
    第二个RNN接受第一个RNN产生的输入序列的语义向量，并且每个时刻t输出词的概率都与前t-1时刻的输出有关
    因此： 其实encoder 就是一个输入
    decoder： 有两个输入，一个是中间状态，一个是结果
    decoder: 输出，就是结果

    Given encoder_inputs [5, 6, 7],  输入可以是英文
    decoder_targets would be [5, 6, 7, 1], where 1 is for EOS, 输出是French
    and decoder_inputs would be [1, 5, 6, 7] - decoder_inputs are lagged by 1 step, passing previous token as input at current step.
        deocder_inputs: 是French?
    '''
    return {
        encoder_inputs: encoder_inputs_,
        decoder_inputs: decoder_inputs_,
        decoder_targets: decoder_targets_,
    }
loss_track = []

max_batches = 3001
batches_in_epoch = 1000

try:
    for batch in range(max_batches):
        fd = next_feed()
        # 生成batch块，经过encoder的解析。

        _, l = sess.run([train_op, loss], fd)
        loss_track.append(l)

        if batch == 0 or batch % batches_in_epoch == 0:
            embed=sess.run(encoder_inputs_embedded,fd)
            print(embed[0],embed.shape)
            print('batch {}'.format(batch))
            print('  minibatch loss: {}'.format(sess.run(loss, fd)))
            predict_ = sess.run(decoder_prediction, fd)
            for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                print('  sample {}:'.format(i + 1))
                print('    input     > {}'.format(inp))
                print('    predicted > {}'.format(pred))
                if i >= 2:
                    break
            print()
except KeyboardInterrupt:
    print('training interrupted')