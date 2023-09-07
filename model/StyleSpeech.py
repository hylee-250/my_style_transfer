import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist

from .modules import (
    MelStyleEncoder,
    PhonemeEncoder,
    MelDecoder, 
    VarianceAdaptor,
    Glow,
)
from utils.tools import get_mask_from_lengths
import pdb

class StyleSpeech(nn.Module):
    """ StyleSpeech """

    def __init__(self, preprocess_config, model_config):
        super(StyleSpeech, self).__init__()
        self.model_config = model_config

        self.mel_style_encoder = MelStyleEncoder(preprocess_config, model_config)
        self.phoneme_encoder = PhonemeEncoder(model_config)
        self.variance_adaptor = VarianceAdaptor(preprocess_config, model_config)
        self.mel_decoder = MelDecoder(model_config)
        self.phoneme_linear = nn.Linear(
            model_config["transformer"]["encoder_hidden"],
            model_config["transformer"]["encoder_hidden"],
        )
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )

        with open(os.path.join(preprocess_config["path"]["preprocessed_path"],
                               "speakers.json"),
            "r",
        ) as f:
            n_speaker = len(json.load(f))
        
        # build post flow
        self.noise_sclale = model_config['flow_postnet']['noise_scale']
        # mel_bin: 80
        cond_hs = 80
        if model_config['flow_postnet']['use_txt_cond'] == True:
            # cond_hs = 80 + 256 = 336
            cond_hs = cond_hs + model_config['flow_postnet']['hidden_size']

        # cond_hs = 336 + 128 = 464
        # 672 / 2 = 336
        cond_hs = cond_hs + model_config['flow_postnet']['style_dim']
        self.post_flow = Glow(
            80, model_config['flow_postnet']['post_glow_hidden'], model_config['flow_postnet']['post_glow_kernel_size'], 1,
            model_config['flow_postnet']['post_glow_n_blocks'], model_config['flow_postnet']['post_glow_n_block_layers'],
            n_split=4, n_sqz=2,
            gin_channels=cond_hs,
            share_cond_layers=model_config['flow_postnet']['post_share_cond_layers'],
            share_wn_layers=model_config['flow_postnet']['share_wn_layers'],
            sigmoid_scale=model_config['flow_postnet']['sigmoid_scale']
        )
        self.prior_dist = dist.Normal(0, 1,validate_args=False)
        

    def G(self,style_vector,texts,src_masks,mel_masks,max_mel_len,
            p_targets=None,e_targets=None,d_targets=None,p_control=1.0,
            e_control=1.0,d_control=1.0,
        ):
        output = self.phoneme_encoder(texts, style_vector, src_masks)
        output = self.phoneme_linear(output)

        (output,p_predictions,e_predictions,log_d_predictions,
            d_rounded,mel_lens,mel_masks,
        ) = self.variance_adaptor(
            output,src_masks,mel_masks,max_mel_len,
            p_targets,e_targets,d_targets,
            p_control,e_control,d_control,
        )
        decoder_inp = output
        mel_out, mel_masks = self.mel_decoder(output, style_vector, mel_masks)
        mel_out = self.mel_linear(mel_out)

        return (
            mel_out,
            p_predictions,
            e_predictions,
            log_d_predictions,
            d_rounded,
            mel_lens,
            mel_masks,
            decoder_inp,
        )
        
        ################### postflow ############################################
        # is_training = self.training
        # train, inference 시 코드가 달라짐
        
        # tgt_nonpadding: target non-padding
        # target mel에서 padding되지 않은 값들
        # ret['x_mask'] = tgt_nonpadding
        # 이미 이전 단계에서 src_mask, mel_masks가 생성되어서 사용됨 return 할 필요 없음
        # ret['spk_embed'] = spk_embed
        # ret['emo_embed'] = emo_embed
        # ret['ref_prosody'] = prosody_utter_mel + prosody_ph_mel + prosody_word_mel
        # spk_embed, emo_embed, local style encoder가 쓰이지 않기에 return 하지 않음
        
        # self.run_post_glow(ref_mels, infer)
        #self.run_post_glow(ref_mels, infer, is_training, ret)
        # return ret
        ########################################################################

    def forward(self,_,texts,src_lens,max_src_len,mels,mel_lens,max_mel_len,
        p_targets=None,e_targets=None,d_targets=None,
        p_control=1.0,e_control=1.0,d_control=1.0,infer=False,
        ):
        src_masks = get_mask_from_lengths(src_lens, max_src_len)
        mel_masks = get_mask_from_lengths(mel_lens, max_mel_len)

        style_vector = self.mel_style_encoder(mels, mel_masks)

        (mel_out,p_predictions,e_predictions,log_d_predictions,
                d_rounded,mel_lens,mel_masks,decoder_inp,
        ) = self.G(style_vector,texts,src_masks,mel_masks,max_mel_len,
                p_targets,e_targets,d_targets,p_control,e_control,d_control,
        )
        if not infer:
            # mels: [B,n_feat,mel_bin]
            # mel_masks: [B,n_feat]
            # mels = mels.transpose(1,2)
            mel_masks = mel_masks.unsqueeze(2)
            # mel_masks: [B,n_feat,1]
            # mel_masks = mel_masks.transpose(1,2)
            
            z_pf,ldj_pf, postflow = self.run_post_glow(mels,infer,mel_out,decoder_inp,style_vector,mel_masks)
            
            return (
                mel_out,p_predictions,e_predictions,log_d_predictions,
                d_rounded,src_masks,mel_masks,src_lens,mel_lens,
                z_pf,ldj_pf, postflow,
            )
        else:
            mel_out = self.run_post_glow(mels,mel_out,decoder_inp,style_vector,mel_masks,infer=True)
            return (
                mel_out,p_predictions,e_predictions,log_d_predictions,
                d_rounded,src_masks,mel_masks,src_lens,mel_lens,
            )
        # return (
        #     mel_out,p_predictions,e_predictions,log_d_predictions,
        #     d_rounded,src_masks,mel_masks,src_lens,mel_lens,
        # )
        
        
    def run_post_glow(self, mels, infer,mel_out, decoder_inp,style_vector,mel_masks):
        # mels: [B,n_feats,mel_bin]
        # mel_masks: [B,n_feats,1]
        # mel_out: [B,n_feats,mel_bin]
        x_recon = mel_out.transpose(1, 2)
        g = x_recon
        # g(x_recon): [B,mel_bin,n_feats]
        B, _, n_feats = g.shape
        if self.model_config['flow_postnet']['use_txt_cond'] == True:
            # decoder_inp: [B,n_feats,hidden_dim]
            # decoder_inp.transpose(1,2): [B,hidden_dim,n_feats]
            g = torch.cat([g, decoder_inp.transpose(1, 2)], 1)
            # g: [B,mel_bin+hidden_dim,n_feats] = [B,80+256=336,n_feats]
        
        g_style_vector = style_vector.repeat(1,n_feats,1).transpose(1,2)
        # g_style_vector: [B,style_dim,n_feats]
        
        g = torch.cat([g, g_style_vector], dim=1)
        # g = g + g_style_vector : [B,mel_bin+hidden_dim+style_dim,n_feats] = [B,336+128=464,n_feats]
        
        prior_dist = self.prior_dist
        if not infer:
            mel_masks = mel_masks.transpose(1, 2)
            y_lengths = mel_masks.sum(-1)
            g = g.detach()
            mels = mels.transpose(1, 2)
            # mel_masks: [B,1,n_feats]
            # mels: [B,mel_bin,n_feats]
            
            # self.post_flow: Glow()
            # z_postflow: mels, ldj: log determinant total
            z_postflow, ldj = self.post_flow(mels, mel_masks, g=g)
            # z_postflow: [B,mel_bin,n_feats], ldj: [B]
            ldj = ldj / y_lengths / 80
            z_pf, ldj_pf = z_postflow, ldj
            postflow = -prior_dist.log_prob(z_postflow).mean() - ldj.mean()
            return z_pf,ldj_pf, postflow
        else:
            mel_masks = torch.ones_like(x_recon[:, :1, :])
            z_post = prior_dist.sample(x_recon.shape).to(g.device) * self.noise_scale
            x_recon_, _ = self.post_flow(z_post, mel_masks, g, reverse=True)
            x_recon = x_recon_
            mel_out = x_recon.transpose(1, 2)
            return mel_out