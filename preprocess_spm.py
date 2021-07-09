#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#


"""
Example: python data/vocab.txt data/train.txt
vocab.txt: 1stline=word, 2ndline=count
"""

import os
import torch
import sys
import numpy as np

from xlm.logger import create_logger
from xlm.data.dictionary import Dictionary
import sentencepiece as spm
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def index_data(path, bin_path, tokenizer: PreTrainedTokenizerBase):
    """
    Index sentences with a sentencepiece model.
    """
    # if bin_path is not None and os.path.isfile(bin_path):
    #     print("Loading data from %s ..." % bin_path)
    #     data = torch.load(bin_path)
    #     # assert spm == data['spm']
    #     return data

    positions = []
    sentences = []
    unk_words = {}

    # index sentences
    f = open(path, 'r', encoding='utf-8')
    for i, line in enumerate(f):
        if i % 1000000 == 0 and i > 0:
            print(i)
        indexed_s = tokenizer.encode(line.rstrip())[1:-1]  # tokenizer automatically add BOS & EOS
        # s = line.rstrip().split()
        # skip empty sentences
        if len(indexed_s) == 0:
            print("Empty sentence in line %i." % i)

        # index sentence words
        count_unk = 0
        indexed = []
        for word_id in indexed_s:
            # if we find a special word which is not an unknown word, skip the sentence
            if word_id in tokenizer.all_special_ids:
                logger.warning('Found unexpected special word "%s" (%i)!!' % (tokenizer.convert_ids_to_tokens(word_id),
                                                                              word_id))
                continue
            assert word_id >= 0
            indexed.append(word_id)
            if word_id == tokenizer.unk_token_id:
                unk_words[tokenizer.unk_token] = unk_words.get(tokenizer.unk_token, 0) + 1
                count_unk += 1
        # add sentence
        positions.append([len(sentences), len(sentences) + len(indexed)])
        sentences.extend(indexed)
        sentences.append(tokenizer.eos_token_id)  # EOS index
    f.close()

    # tensorize data
    positions = np.int64(positions)
    if len(tokenizer) < 1 << 16:
        sentences = np.uint16(sentences)
    elif len(tokenizer) < 1 << 31:
        sentences = np.int32(sentences)
    else:
        raise Exception("Dictionary is too big.")
    assert sentences.min() >= 0
    data = {
        'tokenizer': tokenizer.name_or_path,
        'positions': positions,
        'sentences': sentences,
        'unk_words': unk_words,
    }
    if bin_path is not None:
        print("Saving the data to %s ..." % bin_path)
        torch.save(data, bin_path, pickle_protocol=4)

    return data


if __name__ == '__main__':

    logger = create_logger(None, 0)

    model_path = sys.argv[1]
    txt_path = sys.argv[2]
    bin_path = sys.argv[2] + '.pth'
    # assert os.path.isfile(model_path)
    assert os.path.isfile(txt_path)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    logger.info("")

    data = index_data(txt_path, bin_path, tokenizer)
    logger.info("%i words (%i unique) in %i sentences." % (
        len(data['sentences']) - len(data['positions']),
        len(tokenizer),
        len(data['positions'])
    ))

    if len(data['unk_words']) > 0:
        logger.info("%i unknown words (%i unique), covering %.2f%% of the data." % (
            sum(data['unk_words'].values()),
            len(data['unk_words']),
            sum(data['unk_words'].values()) * 100. / (len(data['sentences']) - len(data['positions']))
        ))
        if len(data['unk_words']) < 30:
            for w, c in sorted(data['unk_words'].items(), key=lambda x: x[1])[::-1]:
                logger.info("%s: %i" % (w, c))
