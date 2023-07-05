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

from src.models.simple.configuration_gpt_neox import SimpleGPTConfig
from src.models.simple.modeling_gpt_neox import (
    SimpleGPTForCausalLM, SimpleGPTModel, SimpleGPTForSequenceClassification
)
from src.models.transformer.tokenization_transformer import NGMETokenizer

AutoConfig.register("simple_gpt", SimpleGPTConfig)
AutoModel.register(SimpleGPTConfig, SimpleGPTModel)
AutoModelForCausalLM.register(SimpleGPTConfig, SimpleGPTForCausalLM)
AutoTokenizer.register("simple_gpt", NGMETokenizer)
AutoModelForSequenceClassification.register(
    SimpleGPTConfig, SimpleGPTForSequenceClassification
)


from src.models.char_former.configuration_transformer import CharFormerConfig
from src.models.char_former.modelling_transformer import (
        NextCharTransformerForCausalLM, NextCharTransformer)
from src.models.transformer.tokenization_transformer import GPTNGMETokenizer

AutoConfig.register("char_former", CharFormerConfig)
# AutoTokenizer.register("gpt_ngme", GPTNGMETokenizer)
AutoModel.register(CharFormerConfig, NextCharTransformer)
AutoModelForCausalLM.register(CharFormerConfig, NextCharTransformerForCausalLM)



from src.models.opt.modeling_opt import NGMEOPTForCausalLM, NGMEOPTModel
from src.models.opt.configuration_opt import NGMEOPTConfig

AutoConfig.register("ngme_opt", NGMEOPTConfig)
AutoModel.register(NGMEOPTConfig, NGMEOPTModel)
AutoModelForCausalLM.register(NGMEOPTConfig, NGMEOPTForCausalLM)

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

]
