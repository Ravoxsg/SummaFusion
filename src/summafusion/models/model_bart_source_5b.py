import copy
import math
import random
import warnings
from typing import Optional, Tuple
from numpy.random import  uniform

import numpy as np 
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.file_utils import (
    add_code_sample_docstrings,
    add_end_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from transformers.modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    Seq2SeqLMOutput,
    Seq2SeqModelOutput,
    Seq2SeqQuestionAnsweringModelOutput,
    Seq2SeqSequenceClassifierOutput,
)
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import logging

from transformers import BartConfig
from transformers.activations import ACT2FN
from transformers.models.bart import BartPretrainedModel
from transformers.models.bart.modeling_bart import BartAttention, BartLearnedPositionalEmbedding, _expand_mask, shift_tokens_right, _make_causal_mask



class BartEncoderLayerSource5b(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_head_mask: torch.Tensor,
        output_attentions: bool = False,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape *(seq_len, batch, embed_dim)*
            attention_mask (`torch.FloatTensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                *(encoder_attention_heads,)*.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states
        hidden_states, attn_weights, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (attn_weights,)

        return outputs

class BartDecoderLayerSource5b(nn.Module):
    def __init__(self, config: BartConfig, args):
        super().__init__()
        self.args = args
        self.embed_dim = config.d_model

        self.self_attn = BartAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = BartAttention(
            self.embed_dim,
            config.decoder_attention_heads,
            dropout=config.attention_dropout,
            is_decoder=True,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

        self.cross_attn_weights = []
        self.cross_attn_weights.append([])
        for _ in range(self.args.n_candidates_to_use):
            self.cross_attn_weights.append([])

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        layer_head_mask: Optional[torch.Tensor] = None,
        cross_attn_layer_head_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = True,
    ):
        """
        Args:
            hidden_states (`torch.FloatTensor`): input to the layer of shape *(batch, seq_len, embed_dim)*
            attention_mask (`torch.FloatTensor`): attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            encoder_hidden_states (`torch.FloatTensor`):
                cross attention input to the layer of shape *(batch, seq_len, embed_dim)*
            encoder_attention_mask (`torch.FloatTensor`): encoder attention mask of size
                *(batch, 1, tgt_len, src_len)* where padding elements are indicated by very large negative values.
            layer_head_mask (`torch.FloatTensor`): mask for attention heads in a given layer of size
                *(encoder_attention_heads,)*.
            cross_attn_layer_head_mask (`torch.FloatTensor`): mask for cross-attention heads in a given layer of
                size *(decoder_attention_heads,)*.
            past_key_value (`Tuple(torch.FloatTensor)`): cached past key and value projection states
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
        """
        residual = hidden_states

        # Self Attention
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        # add present self-attn cache to positions 1,2 of present_key_value tuple
        hidden_states, self_attn_weights, present_key_value = self.self_attn(
            hidden_states=hidden_states,
            past_key_value=self_attn_past_key_value,
            attention_mask=attention_mask,
            layer_head_mask=layer_head_mask,
            output_attentions=output_attentions,
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        cross_attn_present_key_value = None
        cross_attn_weights = None
        if encoder_hidden_states is not None:
            residual = hidden_states

            # cross_attn cached key/values tuple is at positions 3,4 of present_key_value tuple
            cross_attn_past_key_value = past_key_value[-2:] if past_key_value is not None else None
            hidden_states, cross_attn_weights, cross_attn_present_key_value = self.encoder_attn(
                hidden_states=hidden_states,
                key_value_states=encoder_hidden_states,
                attention_mask=encoder_attention_mask,
                layer_head_mask=cross_attn_layer_head_mask,
                past_key_value=cross_attn_past_key_value,
                output_attentions=True,
            )
            if self.args.evaluate_cross_attn_weights:
                weights = cross_attn_weights[:,:,0,:].detach().cpu().numpy()
                weights = weights.reshape((weights.shape[0] * weights.shape[1], weights.shape[2]))
                # source
                src_weights = np.mean(np.sum(weights[:, :self.args.max_source_length], axis = 1), 0)
                self.cross_attn_weights[0].append(src_weights)
                # candidates
                for k in range(self.args.n_candidates_to_use):
                    low = self.args.max_source_length + k * self.args.max_candidate_length
                    high = self.args.max_source_length + (k + 1) * self.args.max_candidate_length
                    cand_weights = np.mean(np.sum(weights[:, low:high], axis = 1), 0)
                    self.cross_attn_weights[k+1].append(cand_weights)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
            hidden_states = residual + hidden_states
            hidden_states = self.encoder_attn_layer_norm(hidden_states)

            # add cross-attn to positions 3,4 of present_key_value tuple
            present_key_value = present_key_value + cross_attn_present_key_value

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states, p=self.activation_dropout, training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = (hidden_states,)

        if output_attentions:
            outputs += (self_attn_weights, cross_attn_weights)

        if use_cache:
            outputs += (present_key_value,)

        return outputs

class BartEncoderSource5b(BartPretrainedModel):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, args, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(embed_dim) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim, self.padding_idx)

        self.args = args

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            embed_dim,
        )
        self.layers = nn.ModuleList([BartEncoderLayerSource5b(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

        if args.classify_candidates:
            out_size = args.hidden_size
            if args.use_source_for_cls:
                out_size += args.hidden_size
            self.fc1 = nn.Linear(out_size, args.cls_hidden_size)
            self.fc2 = nn.Linear(args.cls_hidden_size, len(args.scoring_methods))
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        candidates = True,
        source_encodings = None,
        **model_kwargs
    ):
        r"""
        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`BartTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
            head_mask (`torch.Tensor` of shape `(encoder_layers, encoder_attention_heads)`, *optional*):
                Mask to nullify selected heads of the attention modules. Mask values selected in `[0, 1]`:

                - 1 indicates the head is **not masked**,
                - 0 indicates the head is **masked**.

            inputs_embeds (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`, *optional*):
                Optionally, instead of passing `input_ids` you can choose to directly pass an embedded representation.
                This is useful if you want more control over how to convert `input_ids` indices into associated vectors
                than the model's internal embedding lookup matrix.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            output_hidden_states (`bool`, *optional*):
                Whether or not to return the hidden states of all layers. See `hidden_states` under returned tensors
                for more detail.
            return_dict (`bool`, *optional*):
                Whether or not to return a [`~file_utils.ModelOutput`] instead of a plain tuple.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        if self.args.joint_pos_embeds:
            embed_pos = self.embed_positions(input_shape)

        cls_outputs = None
        if candidates == True:
            #candidate_size = int(inputs_embeds.shape[1] / self.args.n_candidates_to_use)
            candidate_size = self.args.max_candidate_length
            encoder_states = () if output_hidden_states else None
            all_attentions = None
            all_hidden_states = []
            cls_outputs = []

            for j in range(self.args.n_candidates_to_use):

                start = j * candidate_size
                end = (j+1) * candidate_size

                if self.args.joint_pos_embeds:
                    embed_pos_j = embed_pos[start:end, :]
                else:
                    embed_pos_j = self.embed_positions((inputs_embeds.shape[0], candidate_size))
                hidden_states = inputs_embeds[:, start:end, :] + embed_pos_j
                hidden_states = self.layernorm_embedding(hidden_states)
                hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

                # expand attention_mask
                if attention_mask is not None:
                    # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                    attention_mask_j = _expand_mask(attention_mask[:, start:end], inputs_embeds.dtype)

                # check if head_mask has a correct number of layers specified if desired
                if head_mask is not None:
                    if head_mask.size()[0] != (len(self.layers)):
                        raise ValueError(
                            f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                        )

                for idx, encoder_layer in enumerate(self.layers):
                    if output_hidden_states:
                        encoder_states = encoder_states + (hidden_states,)
                    # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                    dropout_probability = random.uniform(0, 1)
                    if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                        layer_outputs = (None, None)
                    else:
                        if self.gradient_checkpointing and self.training:

                            def create_custom_forward(module):
                                def custom_forward(*inputs):
                                    return module(*inputs, output_attentions)

                                return custom_forward

                            layer_outputs = torch.utils.checkpoint.checkpoint(
                                create_custom_forward(encoder_layer),
                                hidden_states,
                                attention_mask_j,
                                (head_mask[idx] if head_mask is not None else None),
                            )
                        else:
                            layer_outputs = encoder_layer(
                                hidden_states,
                                attention_mask_j,
                                layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                                output_attentions=output_attentions,
                            )

                        hidden_states = layer_outputs[0]

                last_hidden_state = hidden_states[:, :, :]

                if self.args.classify_candidates:
                    if self.args.use_ss_for_cls:
                        input_cls = last_hidden_state[:, 0:1, :]
                        if self.args.use_source_for_cls:
                            input_cls = torch.cat((source_encodings[:, 0:1, :], input_cls), -1)
                        cls_preds = self.fc2(self.relu(self.fc1(input_cls)))
                        cls_outputs.append(cls_preds)
                    else:
                        start = (self.args.n_candidates_to_use + j) * candidate_size
                        end = (self.args.n_candidates_to_use + j + 1) * candidate_size
                        embed_pos_j = self.embed_positions((inputs_embeds.shape[0], candidate_size))
                        hidden_states = inputs_embeds[:, start:end, :] + embed_pos_j
                        hidden_states = self.layernorm_embedding(hidden_states)
                        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)
                        # expand attention_mask
                        if attention_mask is not None:
                            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                            attention_mask_j = _expand_mask(attention_mask[:, start:end], inputs_embeds.dtype)
                        # check if head_mask has a correct number of layers specified if desired
                        if head_mask is not None:
                            if head_mask.size()[0] != (len(self.layers)):
                                raise ValueError(
                                    f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                                )
                        for idx, encoder_layer in enumerate(self.layers):
                            if output_hidden_states:
                                encoder_states = encoder_states + (hidden_states,)
                            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                            dropout_probability = random.uniform(0, 1)
                            if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                                layer_outputs = (None, None)
                            else:
                                if self.gradient_checkpointing and self.training:

                                    def create_custom_forward(module):
                                        def custom_forward(*inputs):
                                            return module(*inputs, output_attentions)

                                        return custom_forward

                                    layer_outputs = torch.utils.checkpoint.checkpoint(
                                        create_custom_forward(encoder_layer),
                                        hidden_states,
                                        attention_mask_j,
                                        (head_mask[idx] if head_mask is not None else None),
                                    )
                                else:
                                    layer_outputs = encoder_layer(
                                        hidden_states,
                                        attention_mask_j,
                                        layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                                        output_attentions=output_attentions,
                                    )

                                hidden_states = layer_outputs[0]
                        last_hidden_state_cls = hidden_states[:, :, :]
                        input_cls = last_hidden_state_cls[:, 0:1, :]
                        if self.args.use_source_for_cls:
                            input_cls = torch.cat((source_encodings[:, 0:1, :], input_cls), -1)
                        cls_preds = self.fc2(self.relu(self.fc1(input_cls)))
                        cls_outputs.append(cls_preds)

                all_hidden_states.append(last_hidden_state)
            hidden_states = torch.cat(all_hidden_states, dim = 1)
            if self.args.classify_candidates:
                cls_outputs = torch.cat(cls_outputs, dim = 1)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            ), cls_outputs

        else:

            embed_pos = self.embed_positions(input_shape)
            hidden_states = inputs_embeds + embed_pos
            hidden_states = self.layernorm_embedding(hidden_states)
            hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

            # expand attention_mask
            if attention_mask is not None:
                # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
                attention_mask = _expand_mask(attention_mask, inputs_embeds.dtype)

            encoder_states = () if output_hidden_states else None
            all_attentions = () if output_attentions else None

            # check if head_mask has a correct number of layers specified if desired
            if head_mask is not None:
                if head_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        f"The head_mask should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

            for idx, encoder_layer in enumerate(self.layers):
                if output_hidden_states:
                    encoder_states = encoder_states + (hidden_states,)
                # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
                dropout_probability = random.uniform(0, 1)
                if self.training and (dropout_probability < self.layerdrop):  # skip the layer
                    layer_outputs = (None, None)
                else:
                    if self.gradient_checkpointing and self.training:

                        def create_custom_forward(module):
                            def custom_forward(*inputs):
                                return module(*inputs, output_attentions)

                            return custom_forward

                        layer_outputs = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(encoder_layer),
                            hidden_states,
                            attention_mask,
                            (head_mask[idx] if head_mask is not None else None),
                        )
                    else:
                        layer_outputs = encoder_layer(
                            hidden_states,
                            attention_mask,
                            layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                            output_attentions=output_attentions,
                        )

                    hidden_states = layer_outputs[0]

                if output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

            if output_hidden_states:
                encoder_states = encoder_states + (hidden_states,)

            if not return_dict:
                return tuple(v for v in [hidden_states, encoder_states, all_attentions] if v is not None)
            return BaseModelOutput(
                last_hidden_state=hidden_states, hidden_states=encoder_states, attentions=all_attentions
            ), cls_outputs

class BartDecoderSource5b(BartPretrainedModel):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig, args, embed_tokens: Optional[nn.Embedding] = None):
        super().__init__(config)
        self.args = args
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(config.d_model) if config.scale_embedding else 1.0

        if embed_tokens is not None:
            self.embed_tokens = embed_tokens
        else:
            self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model, self.padding_idx)

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList([BartDecoderLayerSource5b(config, args) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

        self.gradient_checkpointing = False
        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def _prepare_decoder_attention_mask(self, attention_mask, input_shape, inputs_embeds, past_key_values_length):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_causal_mask(
                input_shape, inputs_embeds.dtype, past_key_values_length=past_key_values_length
            ).to(self.device)

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def display_cross_attn_weights(self):
        for k in range(len(self.layers)):
            layer = self.layers[k]
            weights = layer.cross_attn_weights
            mean_weights = [100 * np.mean(x) for x in weights]
            print_weights = ["{:.2f}".format(x) for x in mean_weights]
            print("Layer {}, weights (%): {}".format(k, print_weights))

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        head_mask=None,
        cross_attn_head_mask=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # retrieve input_ids and inputs_embeds
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both decoder_input_ids and decoder_inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            input_ids = input_ids.view(-1, input_shape[-1])
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either decoder_input_ids or decoder_inputs_embeds")

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids) * self.embed_scale

        attention_mask = self._prepare_decoder_attention_mask(
            attention_mask, input_shape, inputs_embeds, past_key_values_length
        )

        # expand encoder attention mask
        if encoder_hidden_states is not None and encoder_attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            encoder_attention_mask = _expand_mask(encoder_attention_mask, inputs_embeds.dtype, tgt_len=input_shape[-1])

        # embed positions
        positions = self.embed_positions(input_shape, past_key_values_length)

        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)

        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout, training=self.training)

        # decoder layers
        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None
        all_cross_attentions = () if (output_attentions and encoder_hidden_states is not None) else None
        next_decoder_cache = () if use_cache else None

        # check if head_mask/cross_attn_head_mask has a correct number of layers specified if desired
        for attn_mask, mask_name in zip([head_mask, cross_attn_head_mask], ["head_mask", "cross_attn_head_mask"]):
            if attn_mask is not None:
                if attn_mask.size()[0] != (len(self.layers)):
                    raise ValueError(
                        "The `{mask_name}` should be specified for {len(self.layers)} layers, but it is for {head_mask.size()[0]}."
                    )

        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue

            past_key_value = past_key_values[idx] if past_key_values is not None else None

            if self.gradient_checkpointing and self.training:

                if use_cache:
                    logger.warning(
                        "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
                    )
                    use_cache = False

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        # None for past_key_value
                        return module(*inputs, output_attentions, use_cache)

                    return custom_forward

                layer_outputs = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(decoder_layer),
                    hidden_states,
                    attention_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    head_mask[idx] if head_mask is not None else None,
                    cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None,
                    None,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_attention_mask,
                    layer_head_mask=(head_mask[idx] if head_mask is not None else None),
                    cross_attn_layer_head_mask=(
                        cross_attn_head_mask[idx] if cross_attn_head_mask is not None else None
                    ),
                    past_key_value=past_key_value,
                    output_attentions=output_attentions,
                    use_cache=use_cache,
                )
            hidden_states = layer_outputs[0]

            if use_cache:
                next_decoder_cache += (layer_outputs[3 if output_attentions else 1],)

            if output_attentions:
                all_self_attns += (layer_outputs[1],)

                if encoder_hidden_states is not None:
                    all_cross_attentions += (layer_outputs[2],)

        # add hidden states from the last decoder layer
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        next_cache = next_decoder_cache if use_cache else None
        if not return_dict:
            return tuple(
                v
                for v in [hidden_states, next_cache, all_hidden_states, all_self_attns, all_cross_attentions]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
            cross_attentions=all_cross_attentions,
        )

class BartModelSource5b(BartPretrainedModel):
    def __init__(self, config: BartConfig, args):
        super().__init__(config)

        self.args = args

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoderSource5b(config, args, self.shared)
        self.decoder = BartDecoderSource5b(config, args, self.shared)

        # Initialize weights and apply final processing
        self.post_init()

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def get_encoder(self):
        return self.encoder

    def get_source_encoder(self):
        return self.encoder

    def get_decoder(self):
        return self.decoder

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        source_out=None,
        source_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        if source_out == None:
            thresh = self.args.max_source_length
            source_ids = input_ids[:, :thresh]
            source_mask = attention_mask[:, :thresh]
            input_ids = input_ids[:, thresh:]
            attention_mask = attention_mask[:, thresh:]
            # source encoding
            source_encoder_outputs, _ = self.encoder(
                input_ids=source_ids,
                attention_mask=source_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                candidates = False,
                source_encodings = None
            )
            source_out = source_encoder_outputs["last_hidden_state"]

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            if input_ids is None:
                raise ValueError(
                    "If no `decoder_input_ids` or `decoder_inputs_embeds` are "
                    "passed, `input_ids` cannot be `None`. Please pass either "
                    "`input_ids` or `decoder_input_ids` or `decoder_inputs_embeds`."
                )

            decoder_input_ids = shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        cls_outputs = None
        if encoder_outputs is None:
            encoder_outputs, cls_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                candidates = True,
                source_encodings = source_out
            )

            to_cat = []
            if self.args.use_source == True:
                to_cat.append(source_out[:, :, :])
            if self.args.use_candidates == True:
                to_cat.append(encoder_outputs[0])
            encoder_output = torch.cat(to_cat, dim = 1)
            encoder_outputs["last_hidden_state"] = encoder_output
            to_cat = []
            if self.args.use_source == True:
                to_cat.append(source_mask)
            if self.args.use_candidates == True:
                to_cat.append(attention_mask[:, :(encoder_outputs[0].shape[1] - source_mask.shape[1])])
            attention_mask = torch.cat(to_cat, dim = 1)

        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )

        # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_outputs[0],
            encoder_attention_mask=attention_mask,
            head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return decoder_outputs + encoder_outputs

        return Seq2SeqModelOutput(
            last_hidden_state=decoder_outputs.last_hidden_state,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        ), cls_outputs

######################################################################################################

class BartForConditionalGenerationSource5b(BartPretrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_missing = [r"final_logits_bias", r"lm_head\.weight"]

    def __init__(self, config: BartConfig, args):
        super().__init__(config)
        self.config = config
        self.args = args
        self.model = BartModelSource5b(config, args)
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_encoder(self):
        return self.model.get_encoder()

    def get_source_encoder(self):
        return self.model.get_source_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        source_out=None,
        source_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )
        
        outputs, cls_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            source_out=source_out,
            source_mask=source_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            cross_attn_head_mask=cross_attn_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias
        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        ), cls_outputs

    def prepare_inputs_for_generation(
        self,
        decoder_input_ids,
        source_out=None,
        source_mask=None,
        past=None,
        attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "source_out": source_out,
            "source_mask": source_mask,
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "decoder_head_mask": decoder_head_mask,
            "cross_attn_head_mask": cross_attn_head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past
