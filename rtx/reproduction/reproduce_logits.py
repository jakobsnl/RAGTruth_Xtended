import argparse
import numpy as np
import os
import random
import torch

import config

from tqdm import tqdm
from utils import MODEL_MAP, load_json, get_tokenizer, get_model, get_device


def set_hf_cache():
    """
    Set the cache directories for Hugging Face datasets and model weights.
    """
    os.environ[
        'HF_DATASETS_CACHE'] = config.hf_cache_dir  # Set location for datasets cache
    os.environ[
        'HF_HOME'] = config.hf_cache_dir  # Set location for model weights cache


def set_random_seeds(seed: int):
    """
    Set random seeds for reproducibility.
    """
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def assign_variables(entry: dict) -> None:
    """
    Assign dictionary key-value pairs as global variables.
    """
    for key, value in entry.items():
        globals()[key] = value


def save_output(sample_output, logits_dir: str, response_range: tuple,
                add_hidden_states: bool) -> None:
    """
    Save logits and hidden states of a model's output.
    """

    # Load & save logits
    start_idx, end_idx = response_range
    logits_list = [
        sample_output.logits[0][idx] for idx in range(start_idx, end_idx + 1)
    ]  # Consider the added, first <s> token
    logits = torch.stack(logits_list)
    torch.save(logits, os.path.join(logits_dir, 'logits.pt'))

    # Load & save hidden states
    if add_hidden_states:
        # Setup for logits hidden states
        hidden_states_dir = os.path.join(logits_dir, 'hidden_states')
        os.makedirs(hidden_states_dir, exist_ok=True)

        for layer_idx, hidden_state in enumerate(sample_output.hidden_states):
            hidden_states_list = [
                hidden_state[0][idx] for idx in range(start_idx, end_idx)
            ]
            hidden_states = torch.stack(hidden_states_list)
            hidden_state_path = os.path.join(hidden_states_dir,
                                             f'{layer_idx}.pt')
            torch.save(hidden_states, hidden_state_path)


def process_samples(model_data_path: str,\
                    model,\
                    tokenizer,\
                    output_dir: str,\
                    device: str,\
                    add_hidden_states: bool) -> None:
    """
    Process dataset samples and compute model outputs.
    """
    # Load model data
    print(f'Loading model data from {model_data_path}')
    model_data = load_json(model_data_path)

    for sample in tqdm(model_data, total=len(model_data), position=0, leave=True, \
                        desc='Processing Samples', unit='samples'):
        # Setup
        assign_variables(sample)  #'prompt' & 'response'
        logits_dir = os.path.join(output_dir, 'outputs', str(sample['id']))
        os.makedirs(logits_dir, exist_ok=True)

        if not os.path.exists(os.path.join(logits_dir, 'logits.pt')):
            input_tokens = torch.tensor(
                tokenizer(prompt, truncation=False,
                          padding=False)['input_ids'])
            output_tokens = torch.tensor(
                tokenizer(response, truncation=False,
                          padding=False)['input_ids'])
            combined_tokens = torch.cat((input_tokens, output_tokens),
                                        dim=-1).unsqueeze(0).to(device)

            # Specify response start and end to only save relevant logits/ hidden states
            # indices - 1 as output contains logits for index i in response at index i - 1
            response_start_idx = input_tokens.size(0) - 1
            response_end_idx = response_start_idx + output_tokens.size(0)

            # Generate model output and save it
            with torch.no_grad():
                output = model(combined_tokens,
                               output_hidden_states=add_hidden_states)
                save_output(output, logits_dir,
                            (response_start_idx, response_end_idx),
                            add_hidden_states)


def reproduce_model_logits(model_name: str, input_dir: str, save_dir: str,
                           device: str, add_hidden_states: bool):
    """
    reproduce a model for logits and hidden states for all dataset samples.
    """
    # Get Model and Tokenizer
    print(f'Loading model: {model_name}')
    model = get_model(model_name, config.hf_cache_dir)
    tokenizer = get_tokenizer(model_name, config.hf_cache_dir)

    # Process dataset
    output_dir = os.path.join(save_dir, model_name)
    model_data_path = os.path.join(input_dir, model_name, f'{model_name}.json')
    process_samples(model_data_path, model, tokenizer, output_dir, device,
                    add_hidden_states)
    print(
        f'Processing complete. Logits and hidden states saved to {output_dir}')


def reproduce_logits(input_dir: str, seed: int, save_dir: str,
                     add_hidden_states: bool):
    """
    Entry point for reproducing logits and hidden states.
    """
    # Setup
    set_hf_cache()
    set_random_seeds(seed)
    device = get_device()

    # reproduce model logits
    for model_name in MODEL_MAP.keys():
        reproduce_model_logits(model_name, input_dir, save_dir, device,
                               add_hidden_states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        default='dataset/rta/',
        help=
        "Directory that contains the models respective folder, e.g. 'dataset/rta/'"
    )
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='dataset/rta/',
        help='Directory in which the outputs shall be saved as .pt files')
    parser.add_argument('--add_hidden_states',
                        action='store_true',
                        default=False)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    reproduce_logits(**vars(args))
