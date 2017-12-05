#encoding:utf-8
#@Time : 2017/12/5 13:19
#@Author : JackNiu
import   nmt.seq2seq.helper as helper
import tensorflow as tf

batch_size = 100
EOS=1

batches = helper.random_sequences(length_from=3, length_to=8,
                                   vocab_lower=2, vocab_upper=10,
                                   batch_size=batch_size)

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
    return encoder_inputs_,decoder_inputs_,decoder_targets_


'''
   输入数据格式: [max_time,batch_size]

'''


x,y,z= next_feed()
print(x.shape,type(x))
print(x[0])
print(z.shape,type(z))
print(z[0])
print(y.shape,type(y))
print(y[1])
