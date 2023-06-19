from argparse import Namespace
from pathlib import Path

from src.models.rnn import RNNModel
# from src.models.transformer.transformer import TransformerLightningModule
from src.utils import count_parameters


def load_model(dictionary, args: Namespace, print_params: bool = True):
    if "lstm" in args.model:
        if args.continue_from and Path(args.continue_from).exists():
            print("LOADING FROM CHECKPOINT")
            model = RNNModel.load_from_checkpoint(args.continue_from, lr=args.lr)
        else:
            model = RNNModel(
                dictionary,
                args.nlayers,
                args.ngram,
                args.hidden_size,
                None,
                args.embedding_size,
                args.lr,
                weighted_loss=args.weighted_loss,
                weighted_labels=args.weighted_labels,
                strategy=args.weight_strat,
                generate=args.generate,
                temperature=args.temperature,
                chars_to_gen=args.chars,
                is_forward_lm=args.is_forward,
                cell_type=args.model,
                packed=args.packed,
                loss_type=args.loss_type,
            )
    else:
        # Adjust args
        args.ntoken = len(dictionary)
        args.weight_tensor = dictionary.create_weight_tensor(
            args.weighted_loss
        ).tolist()

        args.ngram_indexes = dictionary.ngram_indexes
        # args.pad_token_id = dictionary.word2idx["<pad>"]

        # model = TransformerLightningModule(
        #     TransformerConfig.from_args(args), dictionary=dictionary
        # )

    if print_params:
        # Print Parameters
        if hasattr(model, "rnn"):
            print(count_parameters(model.rnn))
        else:
            print(count_parameters(model))

    return model


from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoModelForSequenceClassification, AutoTokenizer)

from src.models.transformer.configuration_transformer import GPTNGMEConfig
from src.models.transformer.modelling_transformer import (
    GPTNGMEForCausalLM, GPTNGMEForSequenceClassification, GPTNGMEModel)
from src.models.transformer.tokenization_transformer import GPTNGMETokenizer

AutoConfig.register("gpt_ngme", GPTNGMEConfig)
AutoTokenizer.register("gpt_ngme", GPTNGMETokenizer)
AutoModel.register(GPTNGMEConfig, GPTNGMEModel)
AutoModelForCausalLM.register(GPTNGMEConfig, GPTNGMEForCausalLM)
AutoModelForSequenceClassification.register(
    GPTNGMEConfig, GPTNGMEForSequenceClassification
)


from src.models.char_former.configuration_transformer import CharFormerConfig
from src.models.char_former.modelling_transformer import (
        NextCharTransformerForCausalLM, NextCharTransformer)
from src.models.transformer.tokenization_transformer import GPTNGMETokenizer

AutoConfig.register("char_former", CharFormerConfig)
# AutoTokenizer.register("gpt_ngme", GPTNGMETokenizer)
AutoModel.register(CharFormerConfig, NextCharTransformer)
AutoModelForCausalLM.register(CharFormerConfig, NextCharTransformerForCausalLM)


from src.models.rwkv.configuration_rwkv import NGMERwkvConfig
from src.models.rwkv.modeling_rwkv import RwkvModel, RwkvForCausalLM

AutoConfig.register("ngme_rwkv", NGMERwkvConfig)
# AutoTokenizer.register("gpt_ngme", GPTNGMETokenizer)
AutoModel.register(NGMERwkvConfig, RwkvModel)
AutoModelForCausalLM.register(NGMERwkvConfig, RwkvForCausalLM)

# Export load_model and all HF transformer files
__all__ = [
    "load_model",
    "GPTNGMEConfig",
    "GPTNGMEModel",
    "GPTNGMETokenizer",
    "GPTNGMEForCausalLM",

    "CharFormerConfig",
    "NextCharTransformer",
    "NextCharTransformerForCausalLM",

    "RwkvConfig",
    "RwkvModel",
    "RwkvForCausalLM"
]
