from typing import Optional, List, Tuple, Union

import streamlit as st

import torch
import torch.nn.functional as F

from src.models import RNNModel
from src.models import llama

USE_LLAMA = False
MODEL_PATH = "./checkpoints/model.pt.last.ckpt"
LLAMA_MODEL_PATH = ""


def load_model(model_path):
    model = RNNModel.load_from_checkpoint(model_path)
    model.eval()
    return model

def load_llama_model():
    model_path = LLAMA_MODEL_PATH + "/7B"
    model_tokenizer = LLAMA_MODEL_PATH + "/tokenizer.model"

    return llama.load_model(model_path, model_tokenizer, 1, 1)


def generate(model, temp: float, seed: Optional[str], chars: int) -> Union[str, List[str]]:
    """Generation tasks for a given seed text"""

    if isinstance(model, llama.LLaMA):
        return llama.generate_from_model(model, seed if seed else "", temp)

    generated_output = seed if seed else ""

    inp = (
        model.dictionary.tokenize_line(
            list(generated_output),
            id_type=torch.int64,
            return_tensor="pt",
        )["source"]
        .unsqueeze(dim=2)
        .to(model.device)
    )

    generated_tokens = []

    with torch.no_grad():
        model.eval()

        for i in range(chars):
            hidden = model.init_hidden(1)
            output, hidden = model(inp, hidden)

            output = output[-1]

            if temp == 0.0:
                output = F.softmax(output, dim=0).cpu()
                # Just get highest confidence
                ngram_idx = torch.argmax(output)
                # Get ngram word
                ngram_order = model.dictionary.get_ngram_order(ngram_idx.item())
                token = model.dictionary.get_item_for_index(ngram_idx.item())
            else:
                output = F.log_softmax(output, dim=0)

                # Remove all UNK tokens for ngram > 2
                # if model.ngrams > 2:
                #     output = torch.index_select(output, 0, token_idxs).cpu()

                word_weights = output.squeeze().div(temp).exp().cpu()

                # multinomial over all tokens
                ngram_idx = torch.multinomial(word_weights, 1)[0]

                ngram_order = model.dictionary.get_ngram_order(ngram_idx.item())

                token = model.dictionary.get_item_for_index(ngram_idx.item())
            
            generated_tokens.append((token, ngram_order))

            # Append to generated sequence
            generated_output = generated_output + token

            # Use last 200 chars as sequence for new input
            inp = (
                model.dictionary.tokenize_line(
                    list(generated_output[-200:]),
                    id_type=torch.int64,
                    return_tensor="pt",
                )["source"]
                .unsqueeze(dim=2)
                .to(model.device)
            )
    return generated_tokens

ngram_to_color = {
    4: "#CD5C5C",
    3: "#F08080",
    2: "#FA8072",
    1: "#E9967A"
}

def color_text(text: str, ngram: int):
    color = ngram_to_color[ngram]
    return f'<span class="big-font" style="background-color:{color}">{text}</span>'

def format_generated_tokens(tokens: List[Tuple[str, int]]):
    return "".join([color_text(text, color) for text, color in tokens])

def generate_text(temperature: float, seed_text: str, num_tokens: int):
    tokens = generate(st.session_state.model, temperature, seed_text, num_tokens)
    if isinstance(tokens, list):
        st.session_state["generated_tokens"] = format_generated_tokens(tokens)
    else:
        st.session_state["generated_tokens"] = tokens

def main():
    st.set_page_config(page_title="Generate with NGME", page_icon=":)")
    st.write(
        """<style>
        [data-testid="column"] {
            width: calc(50% - 1rem);
            flex: 1 1 calc(50% - 1rem);
            min-width: calc(50% - 1rem);
        }
        </style>""",
        unsafe_allow_html=True,
    )

    st.title("Generate text with NGME")
    
    if "model" not in st.session_state:
        with st.spinner("Loading model..."):
            if USE_LLAMA:
                st.session_state["model"] = load_llama_model()
            else:
                st.session_state["model"] = load_model(MODEL_PATH)

    st.success("Model loaded successfully!")

    # settings_col, gen_col = st.columns([1, 3])

    with st.sidebar:
        temperature_slider = st.slider(
            "Temperature",
            0.0, 1.0, 0.7
        )
        no_tokens = st.slider(
            "No. of tokens to generate",
            10, 1000, 100
        )

    seed_text = st.text_input(
        label="Text to start from",
        placeholder="Hello, I am "
    )

    st.button("Generate", on_click=generate_text, args=(temperature_slider, seed_text, no_tokens))

    st.markdown("---")
    st.markdown("""
        <style>
        .big-font {
            font-size:30px !important;
        }
        </style>
        """, unsafe_allow_html=True)
    st.markdown(st.session_state.generated_tokens if "generated_tokens" in st.session_state else "", unsafe_allow_html=True)


if __name__ == '__main__':
    main()
