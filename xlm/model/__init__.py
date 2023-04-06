# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import os
import torch

from .pretrain import load_embeddings
from .transformer import DECODER_ONLY_PARAMS, TransformerModel  # , TRANSFORMER_LAYER_PARAMS
from .memory import HashingMemory
from transformers import AutoModelForMaskedLM


logger = getLogger()


def check_model_params(params):
    """
    Check models parameters.
    """
    # masked language modeling task parameters
    assert params.bptt >= 1
    assert 0 <= params.word_pred < 1
    assert 0 <= params.sample_alpha < 1
    s = params.word_mask_keep_rand.split(',')
    assert len(s) == 3
    s = [float(x) for x in s]
    assert all([0 <= x <= 1 for x in s]) and sum(s) == 1
    params.word_mask = s[0]
    params.word_keep = s[1]
    params.word_rand = s[2]

    # input sentence noise for DAE
    if len(params.ae_steps) == 0:
        assert params.word_shuffle == 0
        assert params.word_dropout == 0
        assert params.word_blank == 0
    else:
        assert params.word_shuffle == 0 or params.word_shuffle > 1
        assert 0 <= params.word_dropout < 1
        assert 0 <= params.word_blank < 1

    # model dimensions
    assert params.emb_dim % params.n_heads == 0

    # share input and output embeddings
    # Leo's comment: either we tie the in and out embeddings or we use Adaptive SoftMax
    assert params.share_inout_emb is False or params.asm is False

    # adaptive softmax
    if params.asm:
        assert params.asm_div_value > 1
        s = params.asm_cutoffs.split(',')
        assert all([x.isdigit() for x in s])
        params.asm_cutoffs = [int(x) for x in s]
        assert params.max_vocab == -1 or params.asm_cutoffs[-1] < params.max_vocab

    # memory
    if params.use_memory:
        HashingMemory.check_params(params)
        s_enc = [x for x in params.mem_enc_positions.split(',') if x != '']
        s_dec = [x for x in params.mem_dec_positions.split(',') if x != '']
        assert len(s_enc) == len(set(s_enc))
        assert len(s_dec) == len(set(s_dec))
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_enc)
        assert all(x.isdigit() or x[-1] == '+' and x[:-1].isdigit() for x in s_dec)
        params.mem_enc_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_enc]
        params.mem_dec_positions = [(int(x[:-1]), 'after') if x[-1] == '+' else (int(x), 'in') for x in s_dec]
        assert len(params.mem_enc_positions) + len(params.mem_dec_positions) > 0
        assert len(params.mem_enc_positions) == 0 or 0 <= min([x[0] for x in params.mem_enc_positions]) <= max([x[0] for x in params.mem_enc_positions]) <= params.n_layers - 1
        assert len(params.mem_dec_positions) == 0 or 0 <= min([x[0] for x in params.mem_dec_positions]) <= max([x[0] for x in params.mem_dec_positions]) <= params.n_layers - 1

    # reload pretrained word embeddings
    if params.reload_emb != '':
        assert os.path.isfile(params.reload_emb)

    # reload a pretrained model
    if params.reload_model != '':
        if params.encoder_only:
            assert os.path.isfile(params.reload_model)
        else:
            s = params.reload_model.split(',')
            assert len(s) == 2
            assert all([x == '' or os.path.isfile(x) for x in s])


def set_pretrain_emb(model, dico, word2id, embeddings):
    """
    Pretrain word embeddings.
    """
    n_found = 0
    with torch.no_grad():
        for i in range(len(dico)):
            idx = word2id.get(dico[i], None)
            if idx is None:
                continue
            n_found += 1
            model.embeddings.weight[i] = embeddings[idx].cuda()
            model.pred_layer.proj.weight[i] = embeddings[idx].cuda()
    logger.info("Pretrained %i/%i words (%.3f%%)."
                % (n_found, len(dico), 100. * n_found / len(dico)))


def copy_parameter_from_hg(hg_model, xlm_model, params):
    def count_param(module):
        return sum([p.numel() for p in module.parameters() if p.requires_grad])
    # self.config
    from copy import deepcopy
    assert 'xlm' in hg_model.name_or_path
    xlm_total_parameters = count_param(xlm_model)
    hg_total_parameters = count_param(hg_model)
    parameter_copied_so_far = 0
    # word embedding
    embedder = hg_model.roberta
    assert xlm_model.embeddings.weight.numel() == embedder.embeddings.word_embeddings.weight.numel()
    xlm_model.embeddings.weight = embedder.embeddings.word_embeddings.weight
    parameter_copied_so_far += embedder.embeddings.word_embeddings.weight.numel()
    # position embeddings
    # TOOD: check if there's special embeddings for special symbols? the magic number of 514 rather than 512
    assert xlm_model.position_embeddings.weight.numel() == embedder.embeddings.position_embeddings.weight.numel()
    if params.sinusoidal_embeddings:
        xlm_model.position_embeddings.weight.detach_()
        xlm_model.position_embeddings.weight.requires_grad = False
        parameter_copied_so_far -= embedder.embeddings.position_embeddings.weight.numel()
    xlm_model.position_embeddings.weight = embedder.embeddings.position_embeddings.weight
    parameter_copied_so_far += embedder.embeddings.position_embeddings.weight.numel()
    # embedding layernorm
    assert type(embedder.embeddings.LayerNorm) == type(xlm_model.layer_norm_emb)
    assert embedder.embeddings.LayerNorm.weight.numel() == embedder.embeddings.LayerNorm.weight.numel()
    xlm_model.layer_norm_emb.weight = embedder.embeddings.LayerNorm.weight
    parameter_copied_so_far += xlm_model.layer_norm_emb.weight.numel()
    if embedder.embeddings.LayerNorm.bias is not None:
        xlm_model.layer_norm_emb.bias = embedder.embeddings.LayerNorm.bias
        parameter_copied_so_far += xlm_model.layer_norm_emb.bias.numel()
    xlm_model.layer_norm_emb.eps = embedder.embeddings.LayerNorm.eps

    for i in range(xlm_model.n_layers):
        # copy attention
        # Encoder: start of layer
        layer = embedder.encoder.layer[i]

        # self attention
        self_attn = layer.attention.self
        assert xlm_model.attentions[i].q_lin.weight.numel() == self_attn.query.weight.numel()
        assert xlm_model.attentions[i].q_lin.bias.numel() == self_attn.query.bias.numel()
        xlm_model.attentions[i].q_lin.weight = self_attn.query.weight
        xlm_model.attentions[i].q_lin.bias = self_attn.query.bias
        parameter_copied_so_far += self_attn.query.weight.numel() + self_attn.query.bias.numel()

        assert xlm_model.attentions[i].k_lin.weight.numel() == self_attn.key.weight.numel()
        assert xlm_model.attentions[i].k_lin.bias.numel() == self_attn.key.bias.numel()
        xlm_model.attentions[i].k_lin.weight = self_attn.key.weight
        xlm_model.attentions[i].k_lin.bias = self_attn.key.bias
        parameter_copied_so_far += self_attn.key.weight.numel() + self_attn.key.bias.numel()

        assert xlm_model.attentions[i].v_lin.weight.numel() == self_attn.value.weight.numel()
        assert xlm_model.attentions[i].v_lin.bias.numel() == self_attn.value.bias.numel()
        xlm_model.attentions[i].v_lin.weight = self_attn.value.weight
        xlm_model.attentions[i].v_lin.bias = self_attn.value.bias
        parameter_copied_so_far += self_attn.value.weight.numel() + self_attn.value.bias.numel()

        # MLP and Layernorm after the attention
        self_output = layer.attention.output
        assert xlm_model.attentions[i].out_lin.weight.numel() == self_output.dense.weight.numel()
        assert xlm_model.attentions[i].out_lin.bias.numel() == self_output.dense.bias.numel()
        xlm_model.attentions[i].out_lin.weight = self_output.dense.weight
        xlm_model.attentions[i].out_lin.bias = self_output.dense.bias
        parameter_copied_so_far += self_output.dense.weight.numel() + self_output.dense.bias.numel()

        assert xlm_model.layer_norm1[i].weight.numel() == self_output.LayerNorm.weight.numel()
        assert xlm_model.layer_norm1[i].bias.numel() == self_output.LayerNorm.bias.numel()
        xlm_model.layer_norm1[i].weight = self_output.LayerNorm.weight
        xlm_model.layer_norm1[i].bias = self_output.LayerNorm.bias
        parameter_copied_so_far += self_output.LayerNorm.weight.numel() + self_output.LayerNorm.bias.numel()
        xlm_model.layer_norm1[i].eps = self_output.LayerNorm.eps

        # so far, we did not consider decoder
        assert layer.is_decoder == False and xlm_model.is_decoder == False
        # Then, we have MLP and Layernorm
        assert xlm_model.ffns[i].lin1.weight.numel() == layer.intermediate.dense.weight.numel()
        assert xlm_model.ffns[i].lin1.bias.numel() == layer.intermediate.dense.bias.numel()
        xlm_model.ffns[i].lin1.weight = layer.intermediate.dense.weight
        xlm_model.ffns[i].lin1.bias = layer.intermediate.dense.bias
        parameter_copied_so_far += layer.intermediate.dense.weight.numel() + layer.intermediate.dense.bias.numel()
        assert xlm_model.ffns[i].lin2.weight.numel() == layer.output.dense.weight.numel()
        assert xlm_model.ffns[i].lin2.bias.numel() == layer.output.dense.bias.numel()
        xlm_model.ffns[i].lin2.weight = layer.output.dense.weight
        xlm_model.ffns[i].lin2.bias = layer.output.dense.bias
        parameter_copied_so_far += layer.output.dense.weight.numel() + layer.output.dense.bias.numel()

        assert xlm_model.layer_norm2[i].weight.numel() == layer.output.LayerNorm.weight.numel()
        assert xlm_model.layer_norm2[i].bias.numel() == layer.output.LayerNorm.bias.numel()
        xlm_model.layer_norm2[i].weight = layer.output.LayerNorm.weight
        xlm_model.layer_norm2[i].bias = layer.output.LayerNorm.bias
        parameter_copied_so_far += layer.output.LayerNorm.weight.numel() + layer.output.LayerNorm.bias.numel()
        xlm_model.layer_norm2[i].eps = layer.output.LayerNorm.eps

    if xlm_model.with_output and params.asm is False:
        # if adaptive softmax is used, we don't copy.
        # input embeddings and output embeddings are tied, by hg default
        xlm_model.pred_layer.proj.weight = hg_model.lm_head.decoder.weight
        xlm_model.pred_layer.proj.bias = hg_model.lm_head.decoder.bias
        parameter_copied_so_far += hg_model.lm_head.decoder.bias.numel()
    # assert parameter_copied_so_far == xlm_total_parameters


def build_model(params, dico):
    """
    Build model.
    """
    if params.use_hg:
        # so far, only encoder_only model is supported
        hg_model = AutoModelForMaskedLM.from_pretrained(params.model_name_or_path)
        # build
        params.layer_norm_eps = hg_model.config.layer_norm_eps
        xlm_model = TransformerModel(params, dico, is_encoder=True, with_output=True)
        copy_parameter_from_hg(hg_model, xlm_model, params)

        logger.info("Model: {}".format(xlm_model))
        logger.info(
            "Number of parameters (model): %i" % sum([p.numel() for p in xlm_model.parameters() if p.requires_grad]))
        return xlm_model if params.use_cpu else xlm_model.cuda()
    if params.encoder_only:
        # build
        model = TransformerModel(params, dico, is_encoder=True, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(model, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            logger.info("Reloading model from %s ..." % params.reload_model)
            reloaded = torch.load(params.reload_model, map_location=lambda storage, loc: storage.cuda(params.local_rank))['model']
            if all([k.startswith('module.') for k in reloaded.keys()]):
                reloaded = {k[len('module.'):]: v for k, v in reloaded.items()}

            # # HACK to reload models with less layers
            # for i in range(12, 24):
            #     for k in TRANSFORMER_LAYER_PARAMS:
            #         k = k % i
            #         if k in model.state_dict() and k not in reloaded:
            #             logger.warning("Parameter %s not found. Ignoring ..." % k)
            #             reloaded[k] = model.state_dict()[k]

            model.load_state_dict(reloaded)

        logger.info("Model: {}".format(model))
        logger.info("Number of parameters (model): %i" % sum([p.numel() for p in model.parameters() if p.requires_grad]))

        return model.cuda()

    else:
        # build
        encoder = TransformerModel(params, dico, is_encoder=True, with_output=True)  # TODO: only output when necessary - len(params.clm_steps + params.mlm_steps) > 0
        decoder = TransformerModel(params, dico, is_encoder=False, with_output=True)

        # reload pretrained word embeddings
        if params.reload_emb != '':
            word2id, embeddings = load_embeddings(params.reload_emb, params)
            set_pretrain_emb(encoder, dico, word2id, embeddings)
            set_pretrain_emb(decoder, dico, word2id, embeddings)

        # reload a pretrained model
        if params.reload_model != '':
            enc_path, dec_path = params.reload_model.split(',')
            assert not (enc_path == '' and dec_path == '')

            # reload encoder
            if enc_path != '':
                logger.info("Reloading encoder from %s ..." % enc_path)
                enc_reload = torch.load(enc_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                enc_reload = enc_reload['model' if 'model' in enc_reload else 'encoder']
                if all([k.startswith('module.') for k in enc_reload.keys()]):
                    enc_reload = {k[len('module.'):]: v for k, v in enc_reload.items()}
                encoder.load_state_dict(enc_reload)

            # reload decoder
            if dec_path != '':
                logger.info("Reloading decoder from %s ..." % dec_path)
                dec_reload = torch.load(dec_path, map_location=lambda storage, loc: storage.cuda(params.local_rank))
                dec_reload = dec_reload['model' if 'model' in dec_reload else 'decoder']
                if all([k.startswith('module.') for k in dec_reload.keys()]):
                    dec_reload = {k[len('module.'):]: v for k, v in dec_reload.items()}
                for i in range(params.n_layers):
                    for name in DECODER_ONLY_PARAMS:
                        if name % i not in dec_reload:
                            logger.warning("Parameter %s not found." % (name % i))
                            dec_reload[name % i] = decoder.state_dict()[name % i]
                decoder.load_state_dict(dec_reload)

        logger.debug("Encoder: {}".format(encoder))
        logger.debug("Decoder: {}".format(decoder))
        logger.info("Number of parameters (encoder): %i" % sum([p.numel() for p in encoder.parameters() if p.requires_grad]))
        logger.info("Number of parameters (decoder): %i" % sum([p.numel() for p in decoder.parameters() if p.requires_grad]))

        return encoder.cuda(), decoder.cuda()
