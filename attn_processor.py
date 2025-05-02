import math
import torch
import diffusers
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from typing import Optional
from diffusers.models.attention_processor import SlicedAttnProcessor, AttnProcessor, Attention


class SelfGuidanceSlicedAttnProcessor_1_5(SlicedAttnProcessor):
    def __init__(self, slice_size: int = 4, module_name: str = None, greenlist: list = None):
        super().__init__(slice_size)
        self.module_name = module_name
        self.greenlist = greenlist

    def __call__(
            self,
            attn: torch.nn.Module,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim # 3

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            height = width = None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) # sd1.5 - [2, 1024, 640] | sd2.1 - [2, 2304, 640]
        query_len = len(query)
        # print("query: ", query.shape, "query len:", query_len)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query) # sd2.1 - [20, 2304, 64]
        # print("query add heads: ", query.shape)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) # [2, 77, 640]
        # print("key: ", key.shape)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key) # [20, 77, 64]
        # print("key add heads: ", key.shape)
        value = attn.head_to_batch_dim(value)

        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads),
            device=query.device,
            dtype=query.dtype,
        )

        all_attention_probs = []  # To collect attention slices

        for i in range((batch_size_attention - 1) // self.slice_size + 1):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx] # sd2.1 - [5, 2304, 64]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
            attn_slice1 = attn_slice
            # print("attn_slice: ", attn_slice.shape) # sd1.5 - [4, 1024, 77] | sd2.1 - [5, 2304, 77]

            # sd1.5
            attn_slice = attn_slice.reshape(query_len, -1, *attn_slice.shape[1:])

            # print("attn_slice: ", attn_slice.shape) # sd1.5 - [2, 2, 1024, 77] | sd2.1 - [5, 1, 2304, 77]
            attn_slice = attn_slice.mean(dim=1)
            # print("attn_slice: ", attn_slice.shape) # sd1.5 - [2, 1024, 77] | sd2.1 - [5, 2304, 77]
            h = w = math.isqrt(attn_slice.shape[1])
            attn_slice = attn_slice.reshape(len(attn_slice), h, w, -1)
            # print("attn_slice: ", attn_slice.shape)
            _SG_RES = 64
            if _SG_RES != attn_slice.shape[2]:
                attn_slice = TF.resize(attn_slice.permute(0, 3, 1, 2), _SG_RES, antialias=True).permute(0, 2, 3, 1)
            # print("attn_slice: ", attn_slice.shape) # sd1.5 - [2, 64, 64, 77] | sd2.1 - [5, 64, 64, 77]

            # Store attention scores for self-guidance
            all_attention_probs.append(attn_slice)

            # TODO: problem is that value is 3d and attn_slice is 4d -> bmm error
            attn_slice1 = torch.bmm(attn_slice1, value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice1

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        # Store all attention probabilities in _aux attribute
        try:
            # TODO: store all_attention_probs in _aux attribute NOT list of 4 tensors
            # attn._aux['attn'] = all_attention_probs
            all_attention_probs = torch.stack(all_attention_probs).mean(0)  # [4,2,64,64,77]->[2,64,64,77]
            attn._aux['attn'] = all_attention_probs
        except AttributeError:
            # TODO: this is not the same as in try block
            attn._aux = {'attn': all_attention_probs}

        return hidden_states

class SelfGuidanceSlicedAttnProcessor_2_1(SlicedAttnProcessor):
    def __init__(self, slice_size: int = 4, module_name: str = None, greenlist: list = None):
        super().__init__(slice_size)
        self.module_name = module_name
        self.greenlist = greenlist

    def __call__(
            self,
            attn: torch.nn.Module,
            hidden_states: torch.Tensor,
            encoder_hidden_states: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = hidden_states

        input_ndim = hidden_states.ndim # 3

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)
        else:
            height = width = None

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states) # sd1.5 - [2, 1024, 640] | sd2.1 - [2, 2304, 640]
        # print("query: ", query.shape, "query len:", query_len)
        dim = query.shape[-1]
        query = attn.head_to_batch_dim(query) # sd2.1 - [20, 2304, 64]
        # print("query add heads: ", query.shape)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states) # [2, 77, 640]
        # print("key: ", key.shape)
        value = attn.to_v(encoder_hidden_states)
        key = attn.head_to_batch_dim(key) # [20, 77, 64]
        # print("key add heads: ", key.shape)
        value = attn.head_to_batch_dim(value)

        batch_size_attention, query_tokens, _ = query.shape
        hidden_states = torch.zeros(
            (batch_size_attention, query_tokens, dim // attn.heads),
            device=query.device,
            dtype=query.dtype,
        )

        all_attention_probs = []  # To collect attention slices

        print(self.module_name)

        for i in range((batch_size_attention - 1) // self.slice_size + 1):
            start_idx = i * self.slice_size
            end_idx = (i + 1) * self.slice_size

            query_slice = query[start_idx:end_idx] # sd2.1 - [5, 2304, 64]
            key_slice = key[start_idx:end_idx]
            attn_mask_slice = attention_mask[start_idx:end_idx] if attention_mask is not None else None

            attn_slice = attn.get_attention_scores(query_slice, key_slice, attn_mask_slice)
            attn_slice1 = attn_slice
            # print("attn_slice: ", attn_slice.shape) # sd2.1 - [5, 2304, 77]

            if self.module_name in self.greenlist:
                print("inside slice loop")
                h = w = math.isqrt(attn_slice.shape[1])
                attn_slice = attn_slice.reshape(len(attn_slice), h, w, -1)
                print("attn_slice: ", attn_slice.shape) # sd2.1 - [5, 48, 48, 77]
                _SG_RES = 64
                if _SG_RES != attn_slice.shape[2]:
                    attn_slice = TF.resize(attn_slice.permute(0, 3, 1, 2), _SG_RES, antialias=True).permute(0, 2, 3, 1)
                # print("attn_slice: ", attn_slice.shape) # sd1.5 - [2, 64, 64, 77] | sd2.1 - [5, 64, 64, 77]

                # Store attention scores for self-guidance
                all_attention_probs.append(attn_slice)

            # TODO: problem is that value is 3d and attn_slice is 4d -> bmm error
            attn_slice1 = torch.bmm(attn_slice1, value[start_idx:end_idx])
            hidden_states[start_idx:end_idx] = attn_slice1

        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear projection
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        if self.module_name in self.greenlist:
            print("store attn values")
            # Store all attention probabilities in _aux attribute
            try:
                # TODO: store all_attention_probs in _aux attribute NOT list of 4 tensors
                # attn._aux['attn'] = all_attention_probs
                all_attention_probs = torch.stack(all_attention_probs).mean(0)  # [4,2,64,64,77]->[2,64,64,77]
                attn._aux['attn'] = all_attention_probs
            except AttributeError:
                # TODO: this is not the same as in try block
                attn._aux = {'attn': all_attention_probs}

        return hidden_states
    
class SelfGuidanceAttnProcessor:
    r"""
    Default processor for performing attention-related computations.
    """

    def __call__(
            self,
            attn: diffusers.models.attention_processor.Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        ### SELF GUIDANCE
        scores_ = attention_probs.reshape(len(query), -1, *attention_probs.shape[1:]).mean(1)
        h = w = math.isqrt(scores_.shape[1])
        scores_ = scores_.reshape(len(scores_), h, w, -1)
        _SG_RES = 64
        if _SG_RES != scores_.shape[2]:
            scores_ = TF.resize(scores_.permute(0, 3, 1, 2), _SG_RES, antialias=True).permute(0, 2, 3, 1)
        try:
            save_aux = False
            if not save_aux:
                len_ = len(attn._aux['attn'])
                del attn._aux['attn']
                attn._aux['attn'] = [None] * len_ + [scores_]
            else:
                attn._aux['attn'][-1] = attn._aux['attn'][-1].cpu()
                attn._aux['attn'].append(scores_)
        except:
            del attn._aux['attn']
            attn._aux = {'attn': [scores_]}
        ### END SELF GUIDANCE

        hidden_states = torch.bmm(attention_probs, value)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states


class SelfGuidanceAttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, save_aux):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError("AttnProcessor2_0 requires PyTorch 2.0, to use it, please upgrade PyTorch to 2.0.")
        self.save_aux = save_aux

    def __call__(
            self,
            attn: diffusers.models.attention_processor.Attention,
            hidden_states,
            encoder_hidden_states=None,
            attention_mask=None,
            temb=None,
    ):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        ### SELF GUIDANCE
        query_ = attn.head_to_batch_dim(query)
        key_ = attn.head_to_batch_dim(key)
        scores_ = attn.get_attention_scores(query_, key_, attention_mask)
        scores_ = scores_.reshape(len(query), -1, *scores_.shape[1:])
        scores_ = scores_.mean(1)
        h = w = math.isqrt(scores_.shape[1])
        scores_ = scores_.reshape(len(scores_), h, w, -1)
        _SG_RES = 64
        if _SG_RES != scores_.shape[2]:
            scores_ = TF.resize(scores_.permute(0, 3, 1, 2), _SG_RES, antialias=True).permute(0, 2, 3, 1)
        try:
            # print("AttnProcessor call", self.save_aux) # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            if not self.save_aux: # save_aux = False
                len_ = len(attn._aux['attn'])
                del attn._aux['attn']
                attn._aux['attn'] = [None] * len_ + [scores_]
            else: # save_aux = True
                attn._aux['attn'][-1] = attn._aux['attn'][-1].cpu()
                attn._aux['attn'].append(scores_)
        except:
            try:
                del attn._aux['attn']
            except:
                pass
            attn._aux = {'attn': [scores_]} # THIS IS IMPORTANT
        ### END SELF GUIDANCE

        inner_dim = key.shape[-1]
        head_dim = inner_dim // attn.heads

        query = query.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        key = key.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, attn.heads, head_dim).transpose(1, 2)

        # the output of sdp = (batch, num_heads, seq_len, head_dim)
        # TODO: add support for attn.scale when we move to Torch 2.1
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, attn_mask=attention_mask, dropout_p=0.0, is_causal=False
        )

        hidden_states = hidden_states.transpose(1, 2).reshape(batch_size, -1, attn.heads * head_dim)
        hidden_states = hidden_states.to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states