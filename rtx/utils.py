import json
import torch
import torch.nn.functional as F
import numpy as np

from config import MIN_K_RATIOS
from transformers import AutoModelForCausalLM, PreTrainedModel, AutoTokenizer, LlamaTokenizer, \
    LlamaForCausalLM, PreTrainedTokenizer
from torchmetrics.functional.classification import binary_auroc

# Global dict that maps all model names to repective tokenizer and model objects
MODEL_MAP = {
    "mistral-7B-instruct": ("mistralai/Mistral-7B-Instruct-v0.2",
                            AutoModelForCausalLM, AutoTokenizer),
    "llama-2-7b-chat":
    ("meta-llama/Llama-2-7b-chat-hf", LlamaForCausalLM, LlamaTokenizer),
    "llama-2-13b-chat":
    ("meta-llama/Llama-2-13b-chat-hf", LlamaForCausalLM, LlamaTokenizer),
    "llama-2-70b-chat":
    ("meta-llama/Llama-2-70b-chat-hf", LlamaForCausalLM, LlamaTokenizer)
}

# Mapping dictionary that maps metrics to functions that can be applied to a metrics
UNCERTAINTY_SCORE_FUNCTION_MAP = {
    "sampled_logit": lambda logits, **kwargs: \
        get_sampled_logits(logits=logits, **kwargs),
    "sampled_probability": lambda logits, axis=-1, **kwargs: \
        get_sampled_logits(logits=torch.softmax(logits, dim=-1), **kwargs),
    "perplexity": lambda logits, axis=-1: \
        torch.exp(- (torch.softmax(logits, dim=axis) *\
            torch.log_softmax(logits, dim=axis)).sum(dim=axis)),
    "variance": lambda logits, axis=-1: \
        torch.var(logits, dim=axis),
    "mean": lambda logits, axis=-1:\
        torch.mean(logits, dim=axis),
    "euclidean": lambda logits, axis=-1:\
        torch.norm(logits, p=2, dim=axis),
    "entropy": lambda logits, axis=-1: \
        -(torch.softmax(logits, dim=axis) * torch.log_softmax(logits, dim=axis)).sum(dim=axis)
}

# Mapping dictionary that maps hallucination sequence scope to indices
SEQUENCE_SCOPE_INDEX_MAP = {'all': -1, 'first': 0, 'second': 1, 'third+': 2}


def get_device() -> str:
    """
    Get available device
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using device: mps (Apple Silicon)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(
            f"Using device: gpu ({torch.cuda.get_device_name(torch.cuda.current_device())})"
        )
    else:
        device = torch.device("cpu")
        print("Using device: cpu")

    return device


def load_json(file_path: str, jsonl: bool = False) -> dict | list[dict]:
    """
    Load given JSON file as dict
    """
    with open(file_path, 'r') as file:
        if jsonl:
            data = [json.loads(line) for line in file]
        else:
            data = json.load(file)

    return data


def save_json(data: dict, filename: str):
    """
    Save given dict as a JSON file
    """
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def load_logits(logits_path: str) -> torch.Tensor:
    """
    Load logits from pytorch file
    """
    # Load logits, exlude last token: logits[-1] is the /s> token,
    # which is not decoded from the response string
    return torch.load(logits_path,
                      weights_only=True,
                      map_location=torch.device('cpu'))[:-1]


def get_tokenizer(model_name: str,
                  cache_dir: str = None) -> PreTrainedTokenizer:
    """
    Load Tokenizer for given model
    """
    # Get tokenizer class
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model {model_name} is not supported.")
    model_path, _, tokenizer_class = MODEL_MAP[model_name]

    # Load tokenizer
    tokenizer = tokenizer_class.from_pretrained(model_path,
                                                cache_dir=cache_dir)
    return tokenizer


def get_model(model_name: str, cache_dir: str = None) -> PreTrainedModel:
    """
    Return the appropriate model for the given model name.
    """
    # Get model class
    if model_name not in MODEL_MAP:
        raise ValueError(f"Model {model_name} is not supported.")
    model_path, model_class, _ = MODEL_MAP[model_name]

    # Load model
    model = model_class.from_pretrained(model_path,
                                        torch_dtype=torch.float16,
                                        cache_dir=cache_dir,
                                        device_map="auto")

    return model


def get_score_keys(function_map: dict) -> list[str]:
    """
    Return a list of score keys from METRIC_FUNCTION_MAP.
    """
    return list(function_map.keys())


def get_sampled_logits(logits: torch.Tensor,
                       response_token_ids: torch.Tensor) -> torch.tensor:
    """
    Return the logit value of each sampled token for a given response sequence
    """
    sampled_logits = logits.gather(dim=-1, index=response_token_ids)
    return sampled_logits.squeeze()


def apply_score_function(logits: torch.tensor, score_key: str,
                         **kwargs) -> torch.tensor:
    """
    Apply a score function to a mogits based on the score key.
    """
    if score_key in UNCERTAINTY_SCORE_FUNCTION_MAP:
        function = UNCERTAINTY_SCORE_FUNCTION_MAP[score_key]
    else:
        raise ValueError(f"Score {score_key} is not supported.")

    return function(logits, **kwargs)


def get_ground_truth_list(entry: dict) -> list[int]:
    """
        Load ground truth data from entry dict and return a list of hallucination labels.
        The first hallucination is marked with 1, and subsequent hallucinations are marked with 2.
        """
    # Get binary hallucination list from annotated range
    total_tokens = entry['ground_truth']['total_tokens']
    annotated_ranges = entry['ground_truth']['annotation']
    ground_truth = [0] * total_tokens  # Initialize with zeros

    for start, end in annotated_ranges:
        for j in range(start, end):
            ground_truth[j] = 1

    return ground_truth


def get_mink(logits: list[torch.Tensor],
             response_token_ids: list[int]) -> dict:
    """
    Return the Min-K value for given logits (probability & entropy). 
    """
    if len(response_token_ids) > 10:
        # Ensure logits is a tensor
        logits = torch.stack(logits)
        # Adjust response_token_ids to match the logits shape
        response_token_ids = torch.stack(response_token_ids).unsqueeze(-1)

        # Calculate probabilities, log probabilities
        probabilities = F.softmax(logits, dim=-1)
        log_probabilities = F.log_softmax(logits, dim=-1)

        # Calculate entropies
        entropies = -(probabilities * log_probabilities).sum(dim=-1)

        # Gather sampled log probabilities
        sampled_log_probabilities = log_probabilities.gather(
            dim=-1, index=response_token_ids).squeeze(-1)

        # Init scores dict
        scores = {
            'mink': {
                f'{int(ratio*100)}': None
                for ratio in MIN_K_RATIOS
            },
            'mink-e': {
                f'{int(ratio*100)}': None
                for ratio in MIN_K_RATIOS
            }
        }

        # Calculate mink & mink-e scores
        for ratio in MIN_K_RATIOS:
            assert len(sampled_log_probabilities) == len(entropies) == len(
                response_token_ids), 'Length mismatch for mink inputs'
            k_length = int(len(sampled_log_probabilities) * ratio)

            mink_topk = np.sort(sampled_log_probabilities.cpu())[:k_length]
            scores['mink'][f'{int(ratio*100)}'] = np.mean(mink_topk).item()

            mink_e_topk = np.sort(entropies.cpu())[:k_length]
            scores['mink-e'][f'{int(ratio*100)}'] = np.mean(mink_e_topk).item()

        return scores
    return None


def min_max_normalize(all_values: torch.Tensor) -> list:
    """
    Normalize scores to a range between 0 and 1.
    """
    min_val, max_val = all_values.min(), all_values.max()
    return torch.tensor([(value - min_val) / (max_val - min_val)
                         for value in all_values])


def get_auroc(hallucination_logits: list,
              non_hallucination_logits: list[int]) -> float:
    """
    Compute AUROC score for given logits and binary hallucination labels.
    """
    if len(hallucination_logits) == 0 or len(non_hallucination_logits) == 0:
        return None
    else:
        logits = min_max_normalize(
            torch.tensor([*hallucination_logits, *non_hallucination_logits]))
        labels = [1] * len(hallucination_logits) + [0] * len(
            non_hallucination_logits)
        if len(hallucination_logits) > 0 and len(non_hallucination_logits) > 0:
            intermediate_auroc = float(
                binary_auroc(logits, torch.tensor(labels, dtype=torch.long)))
            auroc = round(intermediate_auroc, 3)
        else:
            auroc = None

    return auroc


def filter_for_integers(input_list: list) -> list:
    """
    Filters the input list to include only elements that can be converted to integers
    """
    filtered_list = []
    for element in input_list:
        try:
            int(element)
            filtered_list.append(int(element))
        except ValueError:
            continue
    return sorted(filtered_list)
