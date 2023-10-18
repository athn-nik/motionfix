import torch
import torch.nn as nn
from torch import  nn
from src.model.utils.timestep_embed import TimestepEmbedding, Timesteps
from src.model.utils.positional_encoding import PositionalEncoding
from src.model.utils.transf_utils import SkipTransformerEncoder, TransformerEncoderLayer
from src.model.utils.all_positional_encodings import build_position_encoding
from src.data.tools.tensors import lengths_to_mask


class MldDenoiser(nn.Module):

    def __init__(self,
                 ablation = True,
                 nfeats: int = 263,
                 condition: str = "text",
                 latent_dim: list = [1, 256],
                 ff_size: int = 1024,
                 num_layers: int = 9,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 normalize_before: bool = False,
                 activation: str = "gelu",
                 flip_sin_to_cos: bool = True,
                 position_embedding: str = "learned",
                 arch: str = "trans_enc",
                 freq_shift: int = 0,
                 guidance_scale: float = 7.5,
                 guidance_uncondp: float = 0.1,
                 text_encoded_dim: int = 768,
                 use_deltas: bool = False,
                 **kwargs) -> None:

        super().__init__()
        self.use_deltas = use_deltas
        self.latent_dim = latent_dim
        self.text_encoded_dim = text_encoded_dim
        self.condition = condition
        self.abl_plus = False
        self.ablation_skip_connection = False
        self.diffusion_only = True
        self.arch = arch
        self.pe_type = position_embedding

        # if self.diffusion_only:
            # assert self.arch == "trans_enc", "only implement encoder for diffusion-only"
        self.pose_embd = nn.Linear(nfeats, self.latent_dim)
        self.pose_proj = nn.Linear(self.latent_dim, nfeats)
        self.first_pose_proj = nn.Linear(self.latent_dim, nfeats)


        # emb proj
        if self.condition in ["text", "text_uncond"]:
            # text condition
            # project time from text_encoded_dim to latent_dim
            self.time_proj = Timesteps(text_encoded_dim, flip_sin_to_cos,
                                       freq_shift)
            self.time_embedding = TimestepEmbedding(text_encoded_dim,
                                                    self.latent_dim)
            # project time+text to latent_dim
            if text_encoded_dim != self.latent_dim:
                # todo 10.24 debug why relu
                self.emb_proj = nn.Sequential(
                    nn.ReLU(), nn.Linear(text_encoded_dim, self.latent_dim))
        else:
            raise TypeError(f"condition type {self.condition} not supported")

        if self.pe_type == "actor":
            self.query_pos = PositionalEncoding(self.latent_dim, dropout)
            self.mem_pos = PositionalEncoding(self.latent_dim, dropout)
        elif self.pe_type == "mld":
            self.query_pos = build_position_encoding(
                self.latent_dim, position_embedding='learned')
            self.mem_pos = build_position_encoding(
                self.latent_dim, position_embedding='learned')
        else:
            raise ValueError("Not Support PE type")


        if self.ablation_skip_connection:
            # use DETR transformer
            encoder_layer = TransformerEncoderLayer(
                self.latent_dim,
                num_heads,
                ff_size,
                dropout,
                activation,
                normalize_before,
            )
            encoder_norm = nn.LayerNorm(self.latent_dim)
            self.encoder = SkipTransformerEncoder(encoder_layer,
                                                  num_layers, encoder_norm)
        else:
            # use torch transformer
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=self.latent_dim,
                nhead=num_heads,
                dim_feedforward=ff_size,
                dropout=dropout,
                activation=activation)
            self.encoder = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=num_layers)

    def forward(self,
                noised_motion,
                timestep,
                encoder_hidden_states,
                lengths=None,
                **kwargs):
        # 0.  dimension matching
        # noised_motion [latent_dim[0], batch_size, latent_dim] <= [batch_size, latent_dim[0], latent_dim[1]]
        bs = noised_motion.shape[0]
        noised_motion = noised_motion.permute(1, 0, 2)
        # 0. check lengths for no vae (diffusion only)
        if lengths not in [None, []]:
            if self.use_deltas:
                mask = lengths_to_mask([x+1 for x in lengths], noised_motion.device)
            else:
                mask = lengths_to_mask([x for x in lengths], noised_motion.device)

        # 1. time_embedding
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timestep.expand(noised_motion.shape[1]).clone()
        time_emb = self.time_proj(timesteps)
        time_emb = time_emb.to(dtype=noised_motion.dtype)
        # [1, bs, latent_dim] <= [bs, latent_dim]
        time_emb = self.time_embedding(time_emb).unsqueeze(0)

        # 2. condition + time embedding
        if self.condition in ["text", "text_uncond"]:
            # text_emb [seq_len, batch_size, text_encoded_dim] <= [batch_size, seq_len, text_encoded_dim]
            encoder_hidden_states = encoder_hidden_states.permute(1, 0, 2)
            text_emb = encoder_hidden_states  # [num_words, bs, latent_dim]
            # textembedding projection
            if self.text_encoded_dim != self.latent_dim:
                # [1 or 2, bs, latent_dim] <= [1 or 2, bs, text_encoded_dim]
                text_emb_latent = self.emb_proj(text_emb)
            else:
                text_emb_latent = text_emb
            if self.abl_plus:
                emb_latent = time_emb + text_emb_latent
            else:
                emb_latent = torch.cat((time_emb, text_emb_latent), 0)
        else:
            raise TypeError(f"condition type {self.condition} not supported")
        # 4. transformer
        if self.arch == "trans_enc":
            # if self.diffusion_only:
            proj_noised_motion = self.pose_embd(noised_motion)
            xseq = torch.cat((emb_latent, proj_noised_motion), axis=0)
            # else:
            # xseq = torch.cat((sample, emb_latent), axis=0)

            # if self.ablation_skip_connection:
            #     xseq = self.query_pos(xseq)
            #     tokens = self.encoder(xseq)
            # else:
            #     # adding the timestep embed
            #     # [seqlen+1, bs, d]
            #     # todo change to query_pos_decoder
            xseq = self.query_pos(xseq)
            token_mask = torch.ones((bs, 
                                     text_emb_latent.shape[0] + time_emb.shape[0]),
                                     dtype=bool, device=xseq.device)
            aug_mask = torch.cat((token_mask, mask), 1)

            tokens = self.encoder(xseq,
                                  src_key_padding_mask=~aug_mask)
                
            # if self.diffusion_only:
            denoised_motion_proj = tokens[emb_latent.shape[0]:]
            if self.use_deltas:

                denoised_first_pose = self.first_pose_proj(denoised_motion_proj[:1])            
                denoised_motion_only = self.pose_proj(denoised_motion_proj[1:])
                denoised_motion_only[~mask.T[1:]] = 0
                denoised_motion = torch.zeros_like(noised_motion)
                denoised_motion[1:] = denoised_motion_only
                denoised_motion[0] = denoised_first_pose
            else:
                denoised_motion = self.pose_proj(denoised_motion_proj)
                denoised_motion[~mask.T] = 0
            # zero for padded area
            # else:
            #     sample = tokens[:sample.shape[0]]

        else:
            raise TypeError("{self.arch} is not supoorted")

        # 5. [batch_size, latent_dim[0], latent_dim[1]] <= [latent_dim[0], batch_size, latent_dim[1]]
        denoised_motion = denoised_motion.permute(1, 0, 2)

        return denoised_motion
