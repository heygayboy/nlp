#!/usr/bin/python
# -*- coding: UTF-8 -*-

import tensorflow as tf
from tensorflow.python.layers.core import Dense


class config():
    # 超参数
    # Number of Epochs
    epochs = 60
    # Batch Size
    batch_size = 128
    # RNN Size
    rnn_size = 50
    # Number of Layers
    num_layers = 2
    # Embedding Size
    encoding_embedding_size = 15
    decoding_embedding_size = 15
    # Learning Rate
    learning_rate = 0.001


class Seq2Seq(object):
    def __init__(self, config, target_letter_to_int, source_letter_to_int):

        self.target_letter_to_int = target_letter_to_int
        self.source_letter_to_int = source_letter_to_int
        self.config = config

        self.rnn_size, self.num_layers, self.batch_size = \
            config.rnn_size, config.num_layers, config.batch_size
        self.encoding_embedding_size = config.encoding_embedding_size
        self.decoding_embedding_size = config.decoding_embedding_size

        self.inputs, self.targets, self.learning_rate, self.target_sequence_length, \
            self.max_target_sequence_length, self.source_sequence_length \
            = self.get_inputs()

        with tf.name_scope("output"):
            self.training_decoder_output, self.predicting_decoder_output = \
                self.seq2seq_model(self.rnn_size, self.num_layers)

        training_logits = tf.identity(self.training_decoder_output.rnn_output, 'logits')
        predicting_logits = tf.identity(self.predicting_decoder_output.sample_id, name='predictions')

        masks = tf.sequence_mask(self.target_sequence_length, self.max_target_sequence_length, dtype=tf.float32, name='masks')

        with tf.name_scope("optimization"):
            # Loss function
            self.cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                self.targets,
                masks)

            # Optimizer
            optimizer = tf.train.AdamOptimizer(self.learning_rate)

            # Gradient Clipping
            gradients = optimizer.compute_gradients(self.cost)
            capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if
                                grad is not None]
            self.train_op = optimizer.apply_gradients(capped_gradients)

    def get_inputs(self):
        '''
        模型输入tensor
        '''
        inputs = tf.placeholder(tf.int32, [None, None], name='inputs')
        targets = tf.placeholder(tf.int32, [None, None], name='targets')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        # 定义target序列最大长度（之后target_sequence_length和source_sequence_length会作为feed_dict的参数）
        target_sequence_length = tf.placeholder(tf.int32, (None,), name='target_sequence_length')
        max_target_sequence_length = tf.reduce_max(target_sequence_length, name='max_target_len')
        source_sequence_length = tf.placeholder(tf.int32, (None,), name='source_sequence_length')

        return inputs, targets, learning_rate, target_sequence_length, max_target_sequence_length, source_sequence_length

    def get_encoder_layer(self, rnn_size, num_layers):
        '''
        Encoder
        - rnn_size: rnn隐层结点数量
        - num_layers: 堆叠的rnn cell数量
        '''
        source_vocab_size = len(self.source_letter_to_int)
        # Encoder embedding
        encoder_embed_input = tf.contrib.layers.embed_sequence(self.inputs, source_vocab_size, self.encoding_embedding_size)

        # RNN cell
        def get_lstm_cell(rnn_s):
            lstm_cell = tf.contrib.rnn.LSTMCell(rnn_s, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return lstm_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_lstm_cell(rnn_size) for _ in range(num_layers)])

        encoder_output, encoder_state = tf.nn.dynamic_rnn(cell, encoder_embed_input,
                                                          sequence_length=self.source_sequence_length, dtype=tf.float32)

        return encoder_output, encoder_state

    def process_decoder_input(self, data, vocab_to_int):
        '''
        补充<GO>，并移除最后一个字符
        '''
        # cut掉最后一个字符
        ending = tf.strided_slice(data, [0, 0], [self.batch_size, -1], [1, 1])
        decoder_input = tf.concat([tf.fill([self.batch_size, 1], vocab_to_int['<GO>']), ending], 1)

        return decoder_input

    def decoding_layer(self, num_layers, rnn_size, encoder_state, decoder_input):
        '''
        构造Decoder层

        参数：
        - num_layers: 堆叠的RNN单元数量
        - rnn_size: RNN单元的隐层结点数量
        - encoder_state: encoder端编码的状态向量
        - decoder_input: decoder端输入
        '''
        # 1. Embedding
        target_vocab_size = len(self.target_letter_to_int)
        decoder_embeddings = tf.Variable(tf.random_uniform([target_vocab_size, self.decoding_embedding_size]))
        decoder_embed_input = tf.nn.embedding_lookup(decoder_embeddings, decoder_input)

        # 2. 构造Decoder中的RNN单元
        def get_decoder_cell(rnn_size):
            decoder_cell = tf.contrib.rnn.LSTMCell(rnn_size,
                                                   initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
            return decoder_cell

        cell = tf.contrib.rnn.MultiRNNCell([get_decoder_cell(rnn_size) for _ in range(num_layers)])

        # 3. Output全连接层
        output_layer = Dense(target_vocab_size,
                             kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1))

        # 4. Training decoder
        with tf.variable_scope("decode"):
            # 得到help对象
            training_helper = tf.contrib.seq2seq.TrainingHelper(inputs=decoder_embed_input,
                                                                sequence_length=self.target_sequence_length,
                                                                time_major=False)
            # 构造decoder
            training_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                               training_helper,
                                                               encoder_state,
                                                               output_layer)
            training_decoder_output, final_training_state, final_training_sequence_lengths =\
                tf.contrib.seq2seq.dynamic_decode(
                    training_decoder,
                    impute_finished=True,
                    maximum_iterations=self.max_target_sequence_length)
        # 5. Predicting decoder
        # 与training共享参数
        with tf.variable_scope("decode", reuse=True):
            # 创建一个常量tensor并复制为batch_size的大小
            start_tokens = tf.tile(tf.constant([self.target_letter_to_int['<GO>']], dtype=tf.int32), [self.batch_size],
                                   name='start_tokens')
            predicting_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(decoder_embeddings,
                                                                         start_tokens,
                                                                         self.target_letter_to_int['<EOS>'])
            predicting_decoder = tf.contrib.seq2seq.BasicDecoder(cell,
                                                                 predicting_helper,
                                                                 encoder_state,
                                                                 output_layer)
            predicting_decoder_output, final_predicting_state, final_predicting_sequence_lengths = \
                tf.contrib.seq2seq.dynamic_decode(predicting_decoder,
                                                  impute_finished=True,
                                                  maximum_iterations=self.max_target_sequence_length)

        return training_decoder_output, predicting_decoder_output

    def seq2seq_model(self, rnn_size, num_layers):
        # 获取encoder的状态输出
        _, encoder_state = self.get_encoder_layer(rnn_size,num_layers)

        # 预处理后的decoder输入
        decoder_input = self.process_decoder_input(self.targets, self.target_letter_to_int)

        # 将状态向量与输入传递给decoder
        training_decoder_output, predicting_decoder_output = self.decoding_layer(
                                                                            num_layers,
                                                                            rnn_size,
                                                                            encoder_state,
                                                                            decoder_input)

        return training_decoder_output, predicting_decoder_output

