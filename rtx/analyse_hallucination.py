import argparse
import os

from analysis.ModelHandler import ModelHandler
from utils import MODEL_MAP, load_json, save_json


def save_results(model_results: dict,
                 output_dir: str,
                 model_name: str,
                 sequence_scope,
                 sample_ids: list[int] = None) -> None:
    """
    Save the model results to independent json files in the output directory.
    """
    model_output_dir = os.path.join(output_dir, model_name, sequence_scope)
    os.makedirs(model_output_dir, exist_ok=True)

    # Save each scores AUROCs in a separate json file to the output directory
    for score_key in list(model_results[model_name].keys()):
        if score_key == 'mink':
            score_dict = {model_name: model_results[model_name][score_key]}
        else:
            score_dict = {
                model_name: model_results[model_name][score_key],
                'id': model_results['id'][score_key]
            }
        score_json_path = os.path.join(model_output_dir, f'{score_key}.json')
        save_json(score_dict, score_json_path)
        if sample_ids is not None:
            print(
                f'Saved {score_key} AUROCs for {model_name} sample ids {sample_ids} to {score_json_path}'
            )
        else:
            print(
                f'Saved {score_key} AUROCs for {model_name} to {score_json_path}'
            )

    print('--- Process finished ---\n')


def analyse_model_halucination(output_dir: str,
                               dataset_dir: str,
                               model_name: str,
                               sequence_scope: str,
                               sample_size_threshold: int,
                               sample_ids: list[str] = None) -> None:
    """
    Carry out AUROC and Min-K computation for a models dataset
    """
    model_data_dir = os.path.join(dataset_dir, model_name)
    model_data = load_json(os.path.join(model_data_dir, f'{model_name}.json'))

    if sample_ids:
        # Filter the model data based on sample_ids
        model_data = [
            entry for entry in model_data if entry['id'] in sample_ids
        ]
    model_handler = ModelHandler(model_name, model_data, model_data_dir,
                                 sequence_scope, sample_size_threshold)
    model_results = model_handler.to_dict()
    save_results(model_results, output_dir, model_name, sequence_scope,
                 sample_ids)


def analyse_hallucination(output_dir: str,
                          dataset_dir: str,
                          model_name: str,
                          sequence_scopes: list,
                          sample_size_threshold: int,
                          sample_ids: list[str] = None) -> None:
    """
    Carry out AUROC and Min-K computation for the entire rta dataset
    """
    models_to_process = MODEL_MAP.keys() if model_name is None else list(
        model_name)
    for sequence_scope in sequence_scopes:
        for model_to_process in models_to_process:
            print(
                f'Model: {model_to_process}\nSequence Scope: {sequence_scope}')
            analyse_model_halucination(output_dir, dataset_dir,
                                       model_to_process, sequence_scope,
                                       sample_size_threshold, sample_ids)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Verify and visualize saved logits.')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=True,
        help=
        'Path to the output directory for saving the visualizations. E.g.: "results"'
    )
    parser.add_argument('--dataset_dir',
                        type=str,
                        required=True,
                        help='Path to the model data file. E.g. "dataset/rta"')
    parser.add_argument(
        '--model_name',
        type=str,
        required=False,
        nargs='+',
        default=None,
        help=
        'Name of the model to be analysed. If not provided, all models will be analysed.'
    )
    parser.add_argument(
        '--sequence_scopes',
        type=str,
        choices=['all', 'first', 'second', 'third+'],
        default=['all'],
        required=False,
        nargs='+',
        help=
        'Specify one or more hallucination sequence scope \
        (e.g., --sequence_scopes first second third+). Defaults to "all".'
    )
    parser.add_argument(
        '--sample_size_threshold',
        type=int,
        default=100,
        help='Minimum number of samples required to compute global AUROC.')
    parser.add_argument(
        '--sample_ids',
        type=str,
        required=False,
        nargs='+',
        help=
        'To be analysed IDs within the entries of the modeldata JSON file. Mainly debugging purposes'
    )
    args = parser.parse_args()
    if args.sample_ids and not args.model_name:
        parser.error("--model_name is required when --sample_ids is provided.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    analyse_hallucination(**vars(args))
