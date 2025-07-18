import argparse
import config
import os
import matplotlib.patches as patches
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch

from tqdm import tqdm

from utils import load_json, get_tokenizer, load_logits, get_sampled_logits, \
    UNCERTAINTY_SCORE_FUNCTION_MAP


def compute_mean(logits: torch.Tensor):
    """
    Compute mean across each tokens logits for a given logits_path.
    """
    # Return tokenwise mean
    return logits.mean(dim=-1)


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


def normalize_scores(scores: torch.Tensor) -> torch.Tensor:
    """
    Normalize scores to a range between 0 and 1 based on their minimum and maximum values.
    """
    # Skip the first two tokens (e.g., <s> and <pad>)
    min_val = scores[2:].min().item()
    max_val = scores[2:].max().item()
    normalized_scores = (scores - min_val) / (max_val - min_val)
    normalized_scores[:2] = 0
    return normalized_scores


def plot_text_with_highlighting_and_underlining(
        words: list,
        magnitudes: np.ndarray,
        is_hallucinations: torch.Tensor,
        model_name: str,
        sample_id: str,
        source: str,
        words_per_line: int = 15,
        max_magnitude: float = 0.6):
    """
    Plot given response with logit magnitudes as well as marked hallucinations.
    """
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans']
    })
    fig, ax = plt.subplots(figsize=(13, 8))

    for i, (word, magnitude,
            underline) in enumerate(zip(words, magnitudes,
                                        is_hallucinations)):  #-40 for 214
        color = cm.Greens(magnitude / max_magnitude)

        # Add background highlighting
        rect = patches.Rectangle(
            (i % words_per_line - 0.5, -0.5 - 1.2 * (i // words_per_line)),
            1,
            1,
            linewidth=0,
            edgecolor='none',
            facecolor=color)
        ax.add_patch(rect)

        # Add underlining if it's hallucinated
        if underline == 1:
            rect = patches.Rectangle(
                (i % words_per_line - 0.5, -0.5 - 1.2 * (i // words_per_line)),
                1,
                1,
                linewidth=2,
                edgecolor='red',
                facecolor="none",
                zorder=4)
            ax.add_patch(rect)

        # Add the text
        ax.text(i % words_per_line,
                -1.2 * (i // words_per_line),
                word.replace('\u2581', ''),
                color='black',
                ha='center',
                va='center',
                fontsize=18)

    ax.set_xlim(-0.5, words_per_line - 0.5)
    ax.set_ylim(-1.2 * (len(words) // words_per_line) - 0.5, 0.5)  #-40 for 214
    ax.axis('off')  # Turn off the axis

    # Add model_name and sample_id centrally at the top
    fig.text(0.5,
             0.92,
             f'{model_name} [{source} logit, id: {sample_id}]',
             ha='center',
             va='center',
             fontsize=12)


def verify_logits_visually(model_data: dict, model_data_dir: str,
                           model_name: str, output_dir: str | None):
    """
    Verify logits (both qualitatively and quantitatively) 
    for a given sample_id in the RAGTruth_analysis dataset
    """
    tokenizer = get_tokenizer(model_name, config.hf_cache_dir)
    print(f"Processing model: {model_name}")

    for entry in tqdm(model_data,
                      total=len(model_data),
                      desc="Processing Samples - visual check",
                      unit="sample"):
        halluscination_ground_truth = torch.tensor(
            get_ground_truth_list(entry), dtype=int).to('cpu')
        sample_id = entry['id']
        logits_path = os.path.join(model_data_dir, 'outputs', f'{sample_id}',
                                   'logits.pt')
        logits = load_logits(logits_path)

        # Get sampled token indices
        response_token_ids = tokenizer.encode(entry['response'],
                                              return_tensors='pt')

        # Get magnitudes for plotting
        scores_avg = compute_mean(logits).to('cpu')
        scores_sampled = get_sampled_logits(logits, response_token_ids)
        entropies = UNCERTAINTY_SCORE_FUNCTION_MAP['entropy'](logits)

        for source, scores in zip(['mean', 'sampled', 'entropy'],
                                  [scores_avg, scores_sampled, entropies]):

            # Tokenization of response
            token_ids = tokenizer.encode(entry['response'])
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            # Plotting
            plot_text_with_highlighting_and_underlining(
                words=tokens,
                magnitudes=normalize_scores(scores).numpy(),
                is_hallucinations=halluscination_ground_truth,
                model_name=model_name,
                sample_id=sample_id,
                source=source,
                words_per_line=6,
                max_magnitude=1.2)

            # Save the plot
            if output_dir:
                save_dir = os.path.join(output_dir, str(entry['id']))
                os.makedirs(save_dir, exist_ok=True)
                plt.savefig(os.path.join(save_dir, f"{source}.png"), dpi=300)


def visualise_hallucination(model_data_dir: str,
                            sample_ids: list[int] | None = None,
                            output_dir: str = None):
    """
    Visualise responses logits and annotated hallucination in the model data.
    """
    model_name, _ = os.path.splitext(
        os.path.basename(model_data_dir.rstrip("/")))
    model_output_dir = os.path.join(output_dir,
                                    model_name) if output_dir else None
    model_data = load_json(os.path.join(model_data_dir, f'{model_name}.json'))
    if sample_ids:
        # Filter the model data based on sample_ids
        model_data = [
            entry for entry in model_data if entry['id'] in sample_ids
        ]

    verify_logits_visually(model_data, model_data_dir, model_name,
                           model_output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualise token level logits.")
    parser.add_argument('--model_data_dir',
                        type=str,
                        required=False,
                        default='dataset/rta/llama-2-13b-chat',
                        help='Path to the model data file.')
    parser.add_argument(
        '--output_dir',
        type=str,
        required=False,
        default=None,
        help='Path to the output directory for saving the visualizations.')
    parser.add_argument(
        '--sample_ids',
        type=str,
        nargs='+',
        required=False,
        default=[64, 214, 730],
        help='To be verfied IDs within the entries of the modeldata JSON file.'
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    visualise_hallucination(**vars(args))
