import unicodedata
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from util.logit_lens import LogitLens
from baselines.prompt import PromptedModel

def generate_interactive(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    top_k: int = 5,
    max_out_len: int = 200,
    compare_against: Optional[AutoModelForCausalLM] = None,
    use_logit_lens: bool = False,
    layer_module_tmp: str = "transformer.h.{}",
    ln_f_module: str = "transformer.ln_f",
    lm_head_module: str = "lm_head",
):
    """
    Puts generation in a loop. Allows users to repeatedly provide inputs
    with which text is generated.
    """

    if use_logit_lens:
        llens_gen = LogitLens(
            model,
            tok,
            layer_module_tmp,
            ln_f_module,
            lm_head_module,
            disabled=not use_logit_lens,
        )
        if compare_against:
            llens_vanilla = LogitLens(
                compare_against,
                tok,
                layer_module_tmp,
                ln_f_module,
                lm_head_module,
                disabled=not use_logit_lens,
            )

    while True:
        prompt = input("Enter a prompt: ").strip(" \r\t\n")

        print(
            f"Argument Model: "
            f"{generate_fast(model, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
        )
        if compare_against:
            print(
                f"Baseline Model: "
                f"{generate_fast(compare_against, tok, [prompt], n_gen_per_prompt=1, top_k=top_k, max_out_len=max_out_len)}"
            )

        if use_logit_lens:
            inp_prompt = tok([prompt], padding=True, return_tensors="pt").to(
                next(model.parameters()).device
            )

            with llens_gen:
                model(**inp_prompt)
            print("\n--- Argument Model Logit Lens ---")
            llens_gen.pprint()

            if compare_against:
                with llens_vanilla:
                    compare_against(**inp_prompt)
                print("--- Baseline Model Logit Lens ---")
                llens_vanilla.pprint()

        print()


def generate_fast(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    n_gen_per_prompt: int = 1,
    top_k: int = 5,
    max_out_len: int = 200,
):
    """
    Fast, parallelized auto-regressive text generation with top-k sampling.
    Our custom implementation.
    """

    # Unroll prompts and tokenize
    inp = [prompt for prompt in prompts for _ in range(n_gen_per_prompt)]
    inp_tok = tok(inp, padding=True, return_tensors="pt").to(
        next(model.parameters()).device
    )
    input_ids, attention_mask = inp_tok["input_ids"], inp_tok["attention_mask"]
    batch_size = input_ids.size(0)

    # Setup storage of fast generation with attention caches.
    # `cur_context` is used to define the range of inputs that are not yet
    # stored in `past_key_values`. At each step, we are generating the
    # next token for the index at `cur_context.stop + 1`.
    past_key_values, cur_context = None, slice(0, attention_mask.sum(1).min().item())

    with torch.no_grad():
        while input_ids.size(1) < max_out_len:  # while not exceeding max output length
            model_out = model(
                input_ids=input_ids[:, cur_context],
                attention_mask=attention_mask[:, cur_context],
                past_key_values=past_key_values,
                use_cache=True,
            )
            logits, past_key_values = model_out.logits, model_out.past_key_values
            softmax_out = torch.nn.functional.softmax(logits[:, -1, :], dim=1)

            # Top-k sampling
            tk = torch.topk(softmax_out, top_k, dim=1).indices
            softmax_out_top_k = torch.gather(softmax_out, 1, tk)
            softmax_out_top_k = softmax_out_top_k / softmax_out_top_k.sum(1)[:, None]
            new_tok_indices = torch.multinomial(softmax_out_top_k, 1)
            new_toks = torch.gather(tk, 1, new_tok_indices)

            # If we're currently generating the continuation for the last token in `input_ids`,
            # create a new index so we can insert the new token
            if cur_context.stop == input_ids.size(1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_zeros(batch_size, 1)], dim=1
                )
                input_ids = torch.cat(
                    [
                        input_ids,
                        input_ids.new_ones(batch_size, 1) * tok.pad_token_id,
                    ],
                    dim=1,
                )

            last_non_masked = attention_mask.sum(1) - 1
            for i in range(batch_size):
                new_idx = last_non_masked[i] + 1
                if last_non_masked[i].item() + 1 != cur_context.stop:
                    continue

                # Stop generating if we've already maxed out for this prompt
                if new_idx < max_out_len:
                    input_ids[i][new_idx] = new_toks[i]
                    attention_mask[i][new_idx] = 1

            cur_context = slice(cur_context.stop, cur_context.stop + 1)

    txt = [tok.decode(x) for x in input_ids.detach().cpu().numpy().tolist()]
    txt = [
        unicodedata.normalize("NFKD", x)
        .replace("\n\n", " ")
        .replace("<|endoftext|>", "")
        for x in txt
    ]

    return txt


def generate_fast2(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    top_k: int = 5,
    max_out_len: int = 200
):
    """
    Fast, parallel autoregressive text generation using Top-K sampling.

    Args:
        model (AutoModelForCausalLM): The language model for generation.
        tok (AutoTokenizer): The tokenizer corresponding to the model.
        prompts (List[str]): A list of prompts to generate continuations for.
        top_k (int, optional): The k value for Top-K sampling. Defaults to 5.
        max_out_len (int, optional): The maximum length of the generated text. Defaults to 200.

    Returns:
        List[str]: A list of generated texts, containing only the continuation part.
    """
    # If the model is an instance of PromptedModel, get its prompt prefix
    if isinstance(model, PromptedModel):
        prompt_prefix = model.prompt
    else:
        prompt_prefix = ""

    # Set the tokenizer's padding side to left to ensure correct handling during generation
    tok.padding_side = 'left'
    tokens = tok(prompts, return_tensors="pt", padding=True).to("cuda") # Note here
    tok.padding_side = 'right'  # Restore padding side

    # Ensure pad_token_id is set, otherwise use eos_token_id for padding
    pad_token_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # Configure generation parameters, enable sampling and set Top-K
    generation_kwargs = {
        "max_length": max_out_len,
        "do_sample": True,             # Enable sampling
        "top_k": top_k,                # Top-K sampling
        "num_return_sequences": 1,     # Generate one sequence per prompt
        "pad_token_id": pad_token_id
    }

    # Generate output
    output = model.generate(**tokens, **generation_kwargs)

    # Decode the generated sequences
    txt = []
    for i, seq in enumerate(output):
        # Get the corresponding prompt
        prompt = prompts[i]

        # Decode the entire sequence
        decoded_seq = tok.decode(seq, skip_special_tokens=True)

        # Combine the model's prompt prefix (if any)
        if prompt_prefix:
            full_prompt = prompt_prefix + prompt
        else:
            full_prompt = prompt

        # Ensure the generated sequence starts with the prompt, and remove the prompt part
        if decoded_seq.startswith(full_prompt):
            continuation = decoded_seq[len(full_prompt):].lstrip(' ').lstrip('.')
        else:
            # If the generated sequence does not start with the prompt, use the entire generated content
            continuation = decoded_seq

        # Normalize text, remove redundant tokens and newlines
        continuation = unicodedata.normalize("NFKD", continuation).replace("\n\n", " ").replace("<|endoftext|>", "")
        txt.append(continuation)

    return txt
