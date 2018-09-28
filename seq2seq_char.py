#!/usr/bin/python
# -*- coding: UTF-8 -*-

import numpy as np
import time
import tensorflow as tf
from model import Seq2Seq, config


def extract_character_vocab(data):
    '''
    构造映射表
    '''
    special_words = ['<PAD>', '<UNK>', '<GO>',  '<EOS>']

    set_words = list(set([character for line in data.split('\n') for character in line]))
    # 这里要把四个特殊字符添加进词典
    int_to_vocab = {idx: word for idx, word in enumerate(special_words + set_words)}
    vocab_to_int = {word: idx for idx, word in int_to_vocab.items()}

    return int_to_vocab, vocab_to_int


def pad_sentence_batch(sentence_batch, pad_int):
    '''
    对batch中的序列进行补全，保证batch中的每行都有相同的sequence_length

    参数：
    - sentence batch
    - pad_int: <PAD>对应索引号
    '''
    max_sentence = max([len(sentence) for sentence in sentence_batch])
    return [sentence + [pad_int] * (max_sentence - len(sentence)) for sentence in sentence_batch]


def get_batches(targets, sources, batch_size, source_pad_int, target_pad_int):
    '''
    定义生成器，用来获取batch
    '''
    for batch_i in range(0, len(sources) // batch_size):
        start_i = batch_i * batch_size
        sources_batch = sources[start_i:start_i + batch_size]
        targets_batch = targets[start_i:start_i + batch_size]
        # 补全序列
        pad_sources_batch = np.array(pad_sentence_batch(sources_batch, source_pad_int))
        pad_targets_batch = np.array(pad_sentence_batch(targets_batch, target_pad_int))

        # 记录每条记录的长度
        targets_lengths = []
        for target in targets_batch:
            targets_lengths.append(len(target))

        source_lengths = []
        for source in sources_batch:
            source_lengths.append(len(source))

        yield pad_targets_batch, pad_sources_batch, targets_lengths, source_lengths


def train(config, model, source_int, target_int):
    batch_size = config.batch_size

    # 将数据集分割为train和validation
    train_source = source_int[batch_size:]
    train_target = target_int[batch_size:]
    display_step = 50  # 每隔50轮输出loss
    valid_source = source_int[:batch_size]
    valid_target = target_int[:batch_size]
    (valid_targets_batch, valid_sources_batch, valid_targets_lengths,
        valid_sources_lengths) = next(
                    get_batches(valid_target, valid_source, batch_size,
                    source_letter_to_int['<PAD>'],
                    target_letter_to_int['<PAD>']))

    checkpoint = "./trained_model.ckpt"
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(1, config.epochs + 1):
            for batch_i, (targets_batch, sources_batch, targets_lengths, sources_lengths) in enumerate(
                    get_batches(train_target, train_source, batch_size,
                                source_letter_to_int['<PAD>'],
                                target_letter_to_int['<PAD>'])):
                feed_dict = {model.inputs: sources_batch,
                             model.targets: targets_batch,
                             model.learning_rate: config.learning_rate,
                             model.target_sequence_length: targets_lengths,
                             model.source_sequence_length: sources_lengths}
                _, loss = sess.run(
                    [model.train_op, model.cost],
                    feed_dict= feed_dict)

                if batch_i % display_step == 0:
                    # 计算validation loss
                    feed_dict = {model.inputs: valid_sources_batch,
                                 model.targets: valid_targets_batch,
                                 model.learning_rate: config.learning_rate,
                                 model.target_sequence_length: valid_targets_lengths,
                                 model.source_sequence_length: valid_sources_lengths}
                    validation_loss = sess.run(
                        [model.cost],
                        feed_dict = feed_dict)

                    print('Epoch {:>3}/{} Batch {:>4}/{} - Training Loss: {:>6.3f}  - Validation loss: {:>6.3f}'
                          .format(epoch_i,
                                  config.epochs,
                                  batch_i,
                                  len(train_source) // batch_size,
                                  loss,
                                  validation_loss[0]))

        # 保存模型
        saver = tf.train.Saver()
        saver.save(sess, checkpoint)
        print('Model Trained and Saved')


def source_to_seq(text):
    '''
    对源数据进行转换
    '''
    sequence_length = 7
    return [source_letter_to_int.get(word, source_letter_to_int['<UNK>']) for word in text] + [source_letter_to_int['<PAD>']]*(sequence_length-len(text))


def test(config):
    batch_size = config.batch_size
    # 输入一个单词
    input_word = 'common'
    text = source_to_seq(input_word)

    checkpoint = "./trained_model.ckpt"

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # 加载模型
        loader = tf.train.import_meta_graph(checkpoint + '.meta')
        loader.restore(sess, checkpoint)

        input_data = loaded_graph.get_tensor_by_name('inputs:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')

        answer_logits = sess.run(logits, {input_data: [text] * batch_size,
                                          target_sequence_length: [len(input_word)] * batch_size,
                                          source_sequence_length: [len(input_word)] * batch_size})[0]

    pad = source_letter_to_int["<PAD>"]

    print('原始输入:', input_word)

    print('\nSource')
    print('  Word 编号:    {}'.format([i for i in text]))
    print('  Input Words: {}'.format(
        " ".join([source_int_to_letter[i] for i in text])))

    print('\nTarget')
    print('  Word 编号:       {}'.format([i for i in answer_logits if i != pad]))
    print('  Response Words: {}'.format(
        " ".join([target_int_to_letter[i] for i in answer_logits if i != pad])))


if __name__ == '__main__':
    with open(r'data/letters_source.txt', 'r', encoding='utf-8') as f:
        source_data = f.read()

    with open(r'data/letters_target.txt', 'r', encoding='utf-8') as f:
        target_data = f.read()

    print(source_data.split('\n')[:10])
    print(target_data.split('\n')[:10])

    # 构造映射表
    source_int_to_letter, source_letter_to_int = extract_character_vocab(source_data)
    target_int_to_letter, target_letter_to_int = extract_character_vocab(target_data)

    # 对字母进行转换
    source_int = [[source_letter_to_int.get(letter, source_letter_to_int['<UNK>'])
                   for letter in line] for line in source_data.split('\n')]
    target_int = [[target_letter_to_int.get(letter, target_letter_to_int['<UNK>'])
                   for letter in line] + [target_letter_to_int['<EOS>']] for line in target_data.split('\n')]

    config = config()
    model = Seq2Seq(config, target_letter_to_int, source_letter_to_int)
    train(config, model, source_int, target_int)


