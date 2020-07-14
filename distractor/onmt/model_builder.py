"""
This file is for models creation, which consults options
and creates each encoder and decoder accordingly.
"""

import torch
import torch.nn as nn
from torch.nn.init import xavier_uniform_

import onmt.inputters as inputters

from onmt.distractor.embeddings import Embeddings
from onmt.distractor.encoder import DistractorEncoder
from onmt.distractor.decoder import HierDecoder
from onmt.distractor.model import DGModel
# from onmt.distractor.attention import CQAttention
from onmt.utils.misc import use_gpu
from onmt.utils.logging import logger
import numpy as np

def build_embeddings(opt, word_dict, feature_dicts, for_encoder=True):
    """
    Build an Embeddings instance.
    Args:
        opt: the option in current environment.
        word_dict(Vocab): words dictionary.
        feature_dicts([Vocab], optional): a list of feature dictionary.
        for_encoder(bool): build Embeddings for encoder or decoder?
    """
    if for_encoder:
        embedding_dim = opt.src_word_vec_size
    else:
        embedding_dim = opt.tgt_word_vec_size

    word_padding_idx = word_dict.stoi[inputters.PAD_WORD]
    # x = word_dict[0]
    num_word_embeddings = len(word_dict)

    feats_padding_idx = [feat_dict.stoi[inputters.PAD_WORD]
                         for feat_dict in feature_dicts]
    num_feat_embeddings = [len(feat_dict) for feat_dict in
                           feature_dicts]

    return Embeddings(word_vec_size=embedding_dim,
                      position_encoding=opt.position_encoding,
                      feat_merge=opt.feat_merge,
                      feat_vec_exponent=opt.feat_vec_exponent,
                      feat_vec_size=opt.feat_vec_size,
                      dropout=opt.dropout,
                      word_padding_idx=word_padding_idx,
                      feat_padding_idx=feats_padding_idx,
                      word_vocab_size=num_word_embeddings,
                      feat_vocab_sizes=num_feat_embeddings,
                      sparse=opt.optim == "sparseadam")


def load_test_model(opt, dummy_opt, model_path=None):
    if model_path is None:
        model_path = opt.models[0]
    checkpoint = torch.load(model_path,
                            map_location=lambda storage, loc: storage)#map_location='cpu')
    fields = inputters.load_fields_from_vocab(
        checkpoint['vocab'], data_type=opt.data_type)

    model_opt = checkpoint['opt']
    # model_opt.gpuid=-1
    for arg in dummy_opt:
        if arg not in model_opt:
            model_opt.__dict__[arg] = dummy_opt[arg]
    model = build_base_model(model_opt, fields, use_gpu(model_opt), checkpoint)
    # if torch.cuda.device_count() > 1:
    #     model = nn.DataParallel(model)
    model.eval()
    model.generator.eval()
    return fields, model, model_opt

class Btm():
    def __init__(self):
        dir = "/data/home/shuaipengju/distractor_code/data/btm_model/"
        model_dir = dir  # 模型存放的文件夹
        K = 30  # 主题个数
        voca_pt = dir+"topic.vocab.pt"  # 词汇-id对应表路径
        #test_corpus = argvs[4]  # 测试集路径
        # voca = read_voca(voca_pt)#以字典形式存储词汇id
        self.w2id = self.getW2id(voca_pt)
        # test11(w2id)
        W = len(self.w2id)  # 词汇个数
        pz_pt = model_dir + 'k%d.pz' % K  # 主题概率的存储路径
        pz = self.read_pz(pz_pt)
        zw_pt = model_dir + 'k%d.pw_z' % K  # 主题词汇概率分布的存储路径
        k = 0
        self.topics = []
        for l in open(zw_pt):
            app1 = {}  # 以字典形式存储主题下词汇与其对应的概率值
            vs = [float(v) for v in l.split()]
            wvs = zip(range(len(vs)), vs)
            wvs = sorted(wvs, key=lambda d: d[1], reverse=True)
            for w, v in wvs:
                app1[w] = v
            self.topics.append((pz[k], app1))  # 存储到列表：主题-词汇-概率
            # print(len(topics))
            k += 1


    def read_pz(self,pt):
        return [float(p) for p in open(pt).readline().split()]

    def get_embedding(self,word_dict):
        prob_list = []
        # for id in w2id.values():
        for word in word_dict.itos:
            topic_id = self.w2id[word]
            temp = []
            for i in range(len(self.topics)):  # 计算p(b)
                prob_topic = self.topics[i][0]
                prob_w1 = self.topics[i][1][topic_id]
                p = prob_topic * prob_w1
                temp.append(p)
            # temp = []
            # temp.append(fine_prob)
            prob_list.append(temp)
        # return fine_prob
        prob_np = np.array(prob_list, dtype='float32')
        prob_tenr = torch.from_numpy(prob_np)
        # self.prob_soft = torch.nn.functional.softmax(self.prob_list,dim=1)
        prob_soft = torch.nn.functional.softmax(prob_tenr, dim=0)
        prob_emb = nn.Embedding.from_pretrained(prob_soft)  # .to(self.device)
        return prob_emb

    def getW2id(self,pt):

        # f = open(pt, 'r', encoding='utf-8')
        #     # for line in f.readlines():
        #     #     dict_data = json.loads(line)
        #     #     break
        # src_vocab_size = 50000
        vocabs = torch.load(pt)
        # merged_vocab = merge_vocabs(
        #     [vocabs[0][1], vocabs[3][1],
        #      vocabs[2][1], vocabs[1][1]],
        #     vocab_size=src_vocab_size)
        vocabs_dic = {}
        for vocab in vocabs:
            vocabs_dic[vocab[0]] = vocab[1]

        return vocabs_dic['marged'].stoi

    def get_topic_prob(self,word):
        max_prob = 0
        min_prob = 1
        for i in range(len(self.topics)):  # 计算p(b)
            prob_topic = self.topics[i][0]
            prob_w1 = self.topics[i][1][word]
            p = prob_topic * prob_w1
            if p > max_prob:
                max_prob = p
            if p < min_prob:
                min_prob = p
        fine_prob = max_prob - min_prob
        return fine_prob

def build_base_model(model_opt, fields, gpu, checkpoint=None):
    """
    Args:
        model_opt: the option loaded from checkpoint.
        fields: `Field` objects for the model.
        gpu(bool): whether to use gpu.
        checkpoint: the model gnerated by train phase, or a resumed snapshot
                    model from a stopped training.
    Returns:
        the MemModel.
    """

    # Build Topic
    Topic = Btm()

    # Build encoder.
    src_dict = fields["src"].vocab
    feature_dicts = inputters.collect_feature_vocabs(fields, 'src')
    src_embeddings = build_embeddings(model_opt, src_dict, feature_dicts)
    enc_topic_embeddings = Topic.get_embedding(src_dict)

    encoder = DistractorEncoder(enc_topic_embeddings,model_opt.rnn_type,
                                model_opt.word_encoder_type,
                                model_opt.sent_encoder_type,
                                model_opt.question_init_type,
                                model_opt.word_encoder_layers,
                                model_opt.sent_encoder_layers,
                                model_opt.question_init_layers,
                                model_opt.rnn_size, src_dict["<PAD>"],model_opt.dropout,
                                src_embeddings, model_opt.lambda_question,
                                model_opt.lambda_answer)



    # Build decoder.
    tgt_dict = fields["tgt"].vocab
    dec_topic_embeddings = Topic.get_embedding(tgt_dict)
    feature_dicts = inputters.collect_feature_vocabs(fields, 'tgt')
    tgt_embeddings = build_embeddings(model_opt, tgt_dict,
                                      feature_dicts, for_encoder=False)

    # Share the embedding matrix - preprocess with share_vocab required.
    if model_opt.share_embeddings:
        # src/tgt vocab should be the same if `-share_vocab` is specified.
        if src_dict != tgt_dict:
            raise AssertionError('The `-share_vocab` should be set during '
                                 'preprocess if you use share_embeddings!')
        tgt_embeddings.word_lut.weight = src_embeddings.word_lut.weight

    bidirectional_encoder = True if model_opt.question_init_type == 'brnn' else False
    decoder = HierDecoder(dec_topic_embeddings,model_opt.rnn_type, bidirectional_encoder,
                          model_opt.dec_layers, model_opt.rnn_size,
                          model_opt.global_attention,
                          model_opt.dropout,
                          tgt_embeddings)

    # Build NMTModel(= encoder + decoder).
    device = torch.device("cuda" if gpu else "cpu")
    print ('device: ',device)
    model = DGModel(encoder, decoder)

    # Build Generator.
    gen_func = nn.LogSoftmax(dim=-1)
    generator = nn.Sequential(
        nn.Linear(model_opt.dec_rnn_size, len(fields["tgt"].vocab)),
        gen_func
    )


    # Load the model states from checkpoint or initialize them.
    if checkpoint is not None:
        model.load_state_dict(checkpoint['model'], strict=False)
        generator.load_state_dict(checkpoint['generator'], strict=False)
    else:
        if model_opt.param_init != 0.0:
            for p in model.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
            for p in generator.parameters():
                p.data.uniform_(-model_opt.param_init, model_opt.param_init)
        if model_opt.param_init_glorot:
            for p in model.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)
            for p in generator.parameters():
                if p.dim() > 1:
                    xavier_uniform_(p)

        if hasattr(model.encoder, 'embeddings'):
            model.encoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_enc, model_opt.fix_word_vecs_enc)
        if hasattr(model.decoder, 'embeddings'):
            model.decoder.embeddings.load_pretrained_vectors(
                model_opt.pre_word_vecs_dec, model_opt.fix_word_vecs_dec)

    # Add generator to model (this registers it as parameter of model).
    model.generator = generator
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)#, device_ids=[0,1]
        model = model.module
    model.to(device)

    return model


def build_model(model_opt, opt, fields, checkpoint):
    """ Build the Model """
    logger.info('Building model...')
    model = build_base_model(model_opt, fields,
                             use_gpu(opt), checkpoint)
    logger.info(model)
    return model
