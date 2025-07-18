import argparse
import os
import config

from reproduction.reproduce_logits import reproduce_logits
from utils import MODEL_MAP, load_json, save_json, get_tokenizer


def create_source_info_mapping(source_data: list[dict]) -> dict[str, dict]:
    """
    Create mapping dict that makes dataset entries available via source_id as key
    """
    return {entry['source_id']: entry for entry in source_data}


def generate_ground_truth(entry: dict, tokenizer) -> dict:
    """
    Generate ground truth labels for hallucinations in the response using token matching.
    """
    # Get response encoding
    response = entry['response']
    response_token_ids = tokenizer.encode(response)

    # Init ground truth array with response length
    ground_truth = [0] * len(response_token_ids)
    annotated_ranges = []

    # Create binary ground truth information - fill array
    for label in entry['labels']:
        # Get halluscination tokens
        start, end = label['start'], label['end']
        label_text = response[start:end]
        label_token_ids = tokenizer.encode(label_text)[1:]
        assert label_text == label[
            'text'], 'The given token range and label text do not match.'

        # find annotated ranges and set haluscination markers in ground truth array
        for i in range(len(response_token_ids) - len(label_token_ids) + 1):
            if response_token_ids[i:i +
                                  len(label_token_ids)] == label_token_ids:
                for j in range(i, i + len(label_token_ids)):
                    ground_truth[j] = 1
                annotated_ranges.append((i, i + len(label_token_ids)))
                break

    return {
        'total_tokens': len(response_token_ids),
        'annotation': annotated_ranges
    }


def create_combined_entry(response_entry: dict, source_info: dict) -> dict:
    """
    Create a combined entry from response and source data.
    """
    return {
        'id': response_entry['id'],
        'source_id': response_entry['source_id'],
        'temperature': response_entry['temperature'],
        'labels': response_entry['labels'],
        'split': response_entry['split'],
        'quality': response_entry['quality'],
        'response': response_entry['response'],
        'source': source_info.get('source'),
        'source_info': source_info.get('source_info'),
        'task_type': source_info.get('task_type'),
        'prompt': source_info.get('prompt')
    }


def process_responses(response_data: list[dict], source_info_dict: dict[str,
                                                                        dict],
                      models_to_include: list[str]) -> dict[str, list[dict]]:
    """
    Process and restructure RAGTruth responses and combine with according RAGTruth source info.
    """
    # Init structure to store new dataset
    model_dict = {}
    for model in models_to_include:
        model_dict[model] = []

    # Restructure RAGTruth data
    for entry in response_data:
        model = entry.get('model')
        source_id = entry.get('source_id')

        # Filter unused models
        if model not in models_to_include:
            continue

        # Combine response and source_info and add to model data
        source_info = source_info_dict.get(source_id, {})
        combined_entry = create_combined_entry(entry, source_info)
        model_dict[model].append(combined_entry)

    return model_dict


def save_model_data(model_dict: dict[str, list[dict]], save_dir: str):
    """
    Save model data dict model-wise as JSON file
    """
    for model, model_data in model_dict.items():
        model_dir = os.path.join(save_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        model_file = os.path.join(model_dir, f"{model}.json")
        save_json(model_data, model_file)
        print(f"Saved data for model {model} to {model_file}")


def create_dataset(input_dir: str, save_dir: str, seed: int, add_logits: bool,
                   add_hidden_states: bool):
    """
    Restructure intial RAGTruth dataset from response and source_info data to model-wise files
    with added ground truth
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Load files
    response_file = os.path.join(input_dir, 'response.jsonl')
    source_file = os.path.join(input_dir, 'source_info.jsonl')
    response_data = load_json(response_file, jsonl=True)
    source_data = load_json(source_file, jsonl=True)

    # Restructure RAGTruth data to desired format
    source_info_dict = create_source_info_mapping(source_data)
    model_dict = process_responses(response_data, source_info_dict,
                                   MODEL_MAP.keys())

    # Add needed attributes about ground truth hallusciantion labels
    for model_name in model_dict.keys():
        # Create model directory
        os.makedirs(os.path.join(save_dir, model_name, 'outputs'),
                    exist_ok=True)

        # Load tokenizer
        tokenizer = get_tokenizer(model_name, config.hf_cache_dir)
        for i, entry in enumerate(model_dict[model_name]):
            # Add ground truth information
            ground_truth = generate_ground_truth(entry, tokenizer)
            model_dict[model_name][i]['ground_truth'] = ground_truth

    # Save model data per model
    save_model_data(model_dict, save_dir)

    if add_logits:
        # Addtionally reproduce each models responses and save corresponsing  logits
        # (+ optional hidden states)
        reproduce_logits(save_dir, seed, save_dir, add_hidden_states)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        default='dataset/RAGTruth',
        help="Directory that contains the RAGTruth data, e.g. 'dataset/RAGTruth"
    )
    parser.add_argument(
        '--save_dir',
        type=str,
        required=True,
        default='dataset/rta',
        help='Directory in which the model outputs shall be saved.')
    parser.add_argument('--seed', type=int, required=False, default=1)
    parser.add_argument('--add_logits', required=False, action='store_true')
    parser.add_argument('--add_hidden_states',
                        action='store_true',
                        default=False)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    create_dataset(**vars(args))
