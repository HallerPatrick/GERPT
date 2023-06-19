import copy
from typing import Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import math
from transformers import PreTrainedModel
from transformers.modeling_outputs import (BaseModelOutputWithPast,
                                           CausalLMOutputWithPast,
                                           SequenceClassifierOutputWithPast)

from src.loss import CrossEntropyLossSoft
from src.models.ngme import NGramsEmbedding, soft_n_hot
from .configuration_transformer import CharFormerConfig


def clones(module, N):
    """Produce N identical layers."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    """Construct a layernorm module (See citation for details)."""
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


def attention(query, key, value, mask=None, dropout=None):
    """Compute 'Scaled Dot Product Attention'"""
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, max_positions, dropout=0.1):
        """Take in model size and number of heads."""
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

        self.register_buffer(
            "bias",
            torch.tril(
                torch.ones((max_positions, max_positions), dtype=torch.bool)
            ).view(1, 1, max_positions, max_positions),
        )

    def forward(self, query, key, value, mask=None):
        """Implements Figure 2"""
        # if mask is not None:
        #     # Same mask applied to all h heads.
        #     mask = mask.unsqueeze(1)

        nbatches, query_length, _ =  query.size()
        key_length = key.size(-2)

        causal_mask = self.bias[
            :, :, key_length - query_length : key_length, :key_length
        ]

        # causal mask is only for one dimension, apply to whole batch
        causal_mask = causal_mask.repeat(nbatches, 1, 1, 1)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=causal_mask,
                                 dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


def subsequent_mask(size):
    """Mask out subsequent positions."""
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


class PositionwiseFeedForward(nn.Module):
    """Implements FFN equation."""

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class Generator(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)


class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Implement the PE function."""

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)


class AddPositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_sequence_length):
        super(AddPositionalEncoding, self).__init__()
        self.hidden_size = hidden_size
        self.max_sequence_length = max_sequence_length
        self.positional_encoding = nn.Parameter(torch.empty(max_sequence_length, hidden_size))
        nn.init.normal_(self.positional_encoding)

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.positional_encoding[:seq_len]

class EncoderLayer(nn.Module):
    """Encoder is made up of self-attn and feed forward (defined below)"""

    def __init__(self, size, self_attn, feed_forward, dropout,
                 intermediate_layer_predictions=True, generator=None, max_sequence_len=512, force_prediction=False):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.add_positional_encoding = AddPositionalEncoding(size, max_sequence_len)
        self.norm = self.sublayer[0].norm

        self.size = size
        self.intermediate_layer_predictions = intermediate_layer_predictions
        self.force_prediction = force_prediction
        if intermediate_layer_predictions and self.training:
            self.classifier = copy.deepcopy(generator)

    def forward(self, x, mask):
        x = self.add_positional_encoding(x)
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, self.feed_forward)
        if self.force_prediction or (self.intermediate_layer_predictions and self.training):
            return x, self.classifier(self.norm(x))
        else:
            return x, None


class Encoder(nn.Module):
    """Core encoder is a stack of N layers"""

    def __init__(self, layer, n_layers, intermediate_layer_predictions=True):
        super(Encoder, self).__init__()
        self.layers = clones(layer, n_layers)
        # enforce a prediction for the last layer
        self.layers[-1].force_prediction = True
        self.norm = LayerNorm(layer.size)
        self.intermediate_layer_predictions = intermediate_layer_predictions

    def forward(self, input_embds, mask):
        """Pass the input (and mask) through each layer in turn.
        """
        intermediate_predictions = []
        hidden_states = input_embds

        for layer in self.layers:
            hidden_states, prediction = layer(hidden_states, mask)
            # hidden_states = output[0]
            intermediate_predictions.append(prediction)

        hidden_states = self.norm(hidden_states)
        return BaseModelOutputWithPast(last_hidden_state=hidden_states), intermediate_predictions


class MultiLayerCrossEntropy(nn.Module):
    def __init__(self, vocab_size, *args, **kwargs):
        super(MultiLayerCrossEntropy, self).__init__()
        self.cross_entropy = nn.CrossEntropyLoss(*args, **kwargs)
        self.vocab_size = vocab_size

    def forward(self, layer_outputs, target):
        total_loss = torch.zeros(1, dtype=layer_outputs[-1].dtype, device=layer_outputs[-1].device)
        n_layers_with_loss = 0
        for layer_output in layer_outputs:
            if layer_output is not None:
                # if True:
                if self.training:
                    loss = self.cross_entropy(layer_output.view(-1, self.vocab_size).contiguous(), target.view(-1))
                else:
                    # in evaluation consider only the last prediction
                    loss = self.cross_entropy(layer_output[:, -1, :].contiguous(), target)
                total_loss += loss
                n_layers_with_loss += 1

        average_loss_of_all_layers = total_loss / n_layers_with_loss
        final_layer_loss = loss
        return average_loss_of_all_layers, final_layer_loss

class CharFormerPreTrainedModel(PreTrainedModel):
    config_class = CharFormerConfig
    base_model_prefix = "char_former"

class NextCharTransformer(CharFormerPreTrainedModel):
    """
    A standard next-character prediction model. Base for this and many
    other models.
    """
    def __init__(self, config: CharFormerConfig):
        super().__init__(config)

        attn = MultiHeadedAttention(config.n_heads, config.hidden_size, config.max_positional_embeddings, config.dropout)
        ff = PositionwiseFeedForward(config.hidden_size, config.inner_linear, config.dropout)

        generator = Generator(config.hidden_size, config.vocab_size)
        self.encoder = Encoder(EncoderLayer(config.hidden_size, copy.deepcopy(attn), copy.deepcopy(ff),
                                            config.dropout, config.intermediate_layer_predictions, generator,
                                            config.max_positional_embeddings),
                               config.n_layers, config.intermediate_layer_predictions)

        self.embed = Embeddings(config.hidden_size, config.vocab_size)
        

        # use weight sharing
        if config.tied:
            self.generator.proj.weight = self.src_embed.lut.weight

        # This was important from their code.
        # Initialize parameters with Glorot / fan_avg.
        # for p in self.parameters():
        #     if p.dim() > 1:
        #         nn.init.xavier_uniform_(p)

        self.vocab_size = config.vocab_size
        self.intermediate_layer_predictions = config.intermediate_layer_predictions
        self.n_layers = config.n_layers

    def forward(self, input_ids, attention_mask):
        """Take in and process masked src and target sequences."""

        batch_size = input_ids.size(0)
        if attention_mask is not None:
            assert batch_size > 0, "batch_size has to be defined and > 0"
            attention_mask = attention_mask.view(batch_size, -1)
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=self.dtype)  # fp16 compatibility
            attention_mask = (1.0 - attention_mask) * torch.finfo(self.dtype).min

        src_emb = self.embed(input_ids)
        # breakpoint()
        emb, intermediate_predictions = self.encoder(src_emb, attention_mask)
        return emb, intermediate_predictions

    def update(self, training_percent):
        """Stop using losses from intermediate layer as function of time in training.
           See section 2.1 - Intermediate Layer Losses
        """
        for i, layer in enumerate(self.encoder.layers[:-1]):
            if training_percent > (i // (2 * self.n_layers)):
                layer.intermediate_layer_predictions = False

def shift_inplace(tensor):
    shifted_labels = []
    for i in range(tensor.size(1)):
        shifted = tensor[:, i, (i + 1) :].squeeze(0)

        if i == 0:
            shifted_labels.append(shifted)
            continue

        seq_size = tensor.size(2) - 1

        missing_idxs = torch.arange(seq_size - (i), seq_size).to(tensor.device)

        # Shifted labels[0] -> [batch, seq]
        if shifted.dim == 1:
            shifted = torch.concat(
                (shifted, shifted_labels[0].index_select(1, missing_idxs)), dim=0
            )
        else:
            try:
                shifted = torch.concat(
                    (shifted, shifted_labels[0].index_select(1, missing_idxs)), dim=1
                )
            except IndexError as e:
                print(f"shifting failed: {e}")

        shifted_labels.append(shifted)

    return torch.stack(shifted_labels, dim=1)

class NextCharTransformerForCausalLM(CharFormerPreTrainedModel):

    def __init__(self, config: CharFormerConfig) -> None:
        super().__init__(config)
        self.char_former = NextCharTransformer(config)
        # self.embed_out = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fct = MultiLayerCrossEntropy(config.vocab_size)


    # def get_output_embeddings(self):
    #     return self.embed_out

    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.char_former.tokenizer = tokenizer
        # print("Post-Setting tokenizer and calculating token weight tensor...")
        # print("This may take a while...")
        # print("Move this logic somewhere else")
        # self.loss_fct = CrossEntropyLossSoft(
        #     weight=self.tokenizer.create_weight_tensor()
        # )
        self.loss_fct = MultiLayerCrossEntropy(self.config.vocab_size, ignore_index=self.tokenizer.unk_token_id)


    # def set_output_embeddings(self, new_embeddings):
    #     self.embed_out = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        head_mask: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs, intermediate_predictions = self.char_former(
            input_ids,
            attention_mask,
            # head_mask=head_mask,
            # inputs_embeds=inputs_embeds,
            # past_key_values=past_key_values,
            # use_cache=use_cache,
            # output_attentions=output_attentions,
            # output_hidden_states=output_hidden_states,
            # return_dict=return_dict,
        )

        # Intermediate predictions are already the logits
        # It is a list of predictions of every encoder layer
        # Outputs are the hidden states: [batch, seq, hidden]

        lm_loss = None
        if labels is not None:
            shift_logits = []
            for logits in intermediate_predictions:
                if logits is not None:
                    shift_logits.append(logits[:, :-1, :].contiguous())
                else:
                    shift_logits.append(None)

            labels = labels[:, 1:].contiguous()
            average_loss_of_all_layers, final_layer_loss = self.loss_fct(shift_logits, labels)
            lm_loss = average_loss_of_all_layers[0]

        lm_logits = intermediate_predictions[-1]

        # self.char_former.update()

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

        hidden_states = outputs[0]
        lm_logits = self.embed_out(hidden_states)

        lm_loss = None
        if labels is not None:
            # we are doing next-token prediction; shift prediction scores and input ids by one
            if self.config.use_ngme:
                shift_logits = lm_logits[:, :-1, :].contiguous()

                labels = shift_inplace(labels)

                labels = (
                    soft_n_hot(
                        labels.permute(1, 2, 0).contiguous(),
                        self.config.vocab_size,
                        strategy="exp",
                    )
                    .permute(1, 0, 2)
                    .contiguous()
                )

                lm_loss = self.loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    labels.view(-1, labels.size(-1)),
                )
            else:
                shift_logits = lm_logits[:, :-1, :].contiguous()
                loss_fct = torch.nn.CrossEntropyLoss()
                labels = labels[:, 1:].contiguous()
                lm_loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)), labels.view(-1)
                )

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((lm_loss,) + output) if lm_loss is not None else output

        return CausalLMOutputWithPast(
            loss=lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def sample(
        self,
        input_ids,
        tokenizer=None,
        max_length=20,
        temperature=0.7,
        token_divider=None,
        **kwargs,
    ):
        return super().sample(input_ids, **kwargs)

        input_ids = input_ids[0]

        if isinstance(input_ids, str):
            generated_output = input_ids
            input_ids = tokenizer.encode(input_ids, return_tensors="pt").to(self.device)
        else:
            generated_output = str(
                tokenizer.decode(input_ids)
            )  # .decode_ngram_sequence(1)

        generated_output_with_divider = None

        if token_divider:
            generated_output_with_divider = generated_output


        self.eval()
        self.force_prediction = True

        with torch.no_grad():
            for _ in range(max_length):
                iteration = {}

                iteration["input"] = str(tokenizer.decode(input_ids))

                # One batch
                input_ids = input_ids.unsqueeze(0)

                # Reached max length
                if input_ids.size(-1) > self.config.max_positional_embeddings:
                    print("Reached max length")
                    break
    
                output = self.forward(input_ids, return_dict=True)
                # breakpoint()
                next_token_logits = output.logits[0, -1, :]

                next_token_probs = next_token_logits.squeeze().div(temperature)

                next_token_probs = torch.softmax(next_token_logits, dim=-1)
                target_idx = torch.multinomial(next_token_probs, 1)[0]

                # # Get ngram word
                try:
                    word = tokenizer.get_item_for_index(target_idx.item())
                except:
                    word = tokenizer.convert_ids_to_tokens(target_idx.item())

                # # Append to generated sequence
                generated_output = generated_output + word

                if token_divider:
                    generated_output_with_divider = (
                        generated_output_with_divider + token_divider + word
                    )

                # Use last 200 chars as sequence for new input
                input_ids = (
                    (tokenizer(generated_output, return_tensors="pt")["input_ids"])
                    .to(self.device)
                    .squeeze(0)
                )
        self.force_prediction = False

        if token_divider and generated_output_with_divider:
            return generated_output_with_divider

        return generated_output
