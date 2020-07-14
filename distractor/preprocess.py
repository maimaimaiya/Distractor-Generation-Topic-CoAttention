#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
    Pre-process Data / features files and build vocabulary
"""

import argparse
import glob
import sys
import gc
import os
import codecs
import torch
from onmt.utils.logging import init_logger, logger

import onmt.inputters as inputters
import onmt.opts as opts


def parse_args():
    """ Parsing arguments """
    parser = argparse.ArgumentParser(
        description='preprocess.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
   # parser.add_argument("")
   #  group.add_argument('-train_dir', required=True, default='data/race_train.json',
   #                     help="Path to the training data")
   #  group.add_argument('-valid_dir', required=True, default='data/race_dev.json',
   #                     help="Path to the validation data")
   #  group.add_argument('-data_type', choices=["text"], help="""text""")
   #  group.add_argument('-save_data', required=True, default='data/processed',
   #                     help="Output file for the prepared data")

    opts.add_md_help_argument(parser)
    opts.preprocess_opts(parser)
    #print (parser.parse_args())
    opt = parser.parse_args()
    torch.manual_seed(opt.seed)

    return opt


def build_save_dataset(corpus_type, fields, opt):
    """ Building and saving the dataset """
    assert corpus_type in ['train', 'valid']

    if corpus_type == 'train':
        corpus = opt.train_dir
    else:
        corpus = opt.valid_dir

    dataset = inputters.build_dataset(
        fields,
        data_path=corpus,
        data_type=opt.data_type,
        total_token_length=opt.total_token_length,
        src_seq_length=opt.src_seq_length,
        src_sent_length=opt.src_sent_length,
        seq_length_trunc=opt.seq_length_trunc)

    # We save fields in vocab.pt seperately, so make it empty.
    dataset.fields = []

    pt_file = "{:s}.{:s}.pt".format(opt.save_data, corpus_type)
    logger.info(" * saving %s dataset to %s." % (corpus_type, pt_file))
    torch.save(dataset, pt_file)

    return pt_file


def build_save_vocab(train_dataset, data_type, fields, opt):
    """ Building and saving the vocab """
    fields = inputters.build_vocab(train_dataset, data_type, fields,
                                   opt.share_vocab,
                                   opt.src_vocab_size,
                                   opt.src_words_min_frequency,
                                   opt.tgt_vocab_size,
                                   opt.tgt_words_min_frequency)

    # Can't save fields, so remove/reconstruct at training time.
    vocab_file = opt.save_data + '.vocab.pt'
    logger.info("saving vocabulary {}".format(vocab_file))
    torch.save(inputters.save_fields_to_vocab(fields), vocab_file)


def main():
    opt = parse_args()

    init_logger(opt.log_file)
    logger.info("Extracting features...")

    logger.info("Building `Fields` object...")
    fields = inputters.get_fields(opt.data_type)

    logger.info("Building & saving training data...")
    train_dataset_files = build_save_dataset('train', fields, opt)

    logger.info("Building & saving validation data...")
    build_save_dataset('valid', fields, opt)

    logger.info("Building & saving vocabulary...")
    # train_dataset_files = 'data/processed.train.pt'
    build_save_vocab(train_dataset_files, opt.data_type, fields, opt)


if __name__ == "__main__":

    main()
