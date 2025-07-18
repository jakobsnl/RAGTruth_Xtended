import config
import os
import torch

from copy import deepcopy
from tqdm import tqdm

from utils import UNCERTAINTY_SCORE_FUNCTION_MAP, SEQUENCE_SCOPE_INDEX_MAP, \
    load_logits, get_tokenizer, get_auroc, get_ground_truth_list, apply_score_function, get_score_keys, \
    filter_for_integers, get_mink
from .AurocAnalyser import SequenceAurocAnalyser, TokenAurocAnalyser
from .MinkAnalyser import MinkTokenAnalyser


class ModelHandler:
    """
    Class to handle the logit-based hallucination analysis across the dataset.
    
    Attributes:
        model_name (str): The name of the model.
        model_data (dict): The data associated with the model.
        model_data_dir (str): The directory where the model data is stored.
        sequence_scope (int): The index of the hallucination sequence scope. 
                              Wether to analyse all, first_sequence, second or third+ (all from third) 
                              hallucination and non hallucnation sequence token.
        
        sample_size_threshold (int): sample size threshold (size of logit collection, e.g. per index), 
                                     below which the AUROC is not computed.
    """

    def __init__(self, model_name: str, model_data: dict, model_data_dir: str,
                 sequence_scope: str, sample_size_threshold: int):
        self.model_name = model_name
        self.model_data = model_data
        self.model_data_dir = model_data_dir
        self.sequence_scope = SEQUENCE_SCOPE_INDEX_MAP[sequence_scope]
        self.auroc_score_keys = get_score_keys(UNCERTAINTY_SCORE_FUNCTION_MAP)
        self.mink_score_keys = ['logits', 'response_token_ids']
        self.sample_size_threshold = sample_size_threshold

        # Init global storage for different scores and parts of hallucinations
        self.scores = {}

        for score_key in self.auroc_score_keys:
            # Store different token categories for global analysis
            self.scores[score_key] = {
                'all': [],
                'first_sequence': [],
                'subsequent_sequences': [],
                'within_sequence_index': {},
                'non': []
            }

        # Init global storage of results
        self.results = {'auroc': {}, 'mink': {}}

        for score_key in self.auroc_score_keys:
            self.results['auroc'][score_key] = {
                'all': None,
                'first_sequence': None,
                'subsequent_sequences': None,
                'first_vs_subsequent_sequences': None,
                'auroc_within_sequence_index': {}
            }

        for score_key in self.mink_score_keys:
            self.scores[score_key] = {
                'hallucination': {
                    'within_sequence_index': {}
                },
                'pre_hallucination': {
                    'within_sequence_index': {}
                },
                'no_hallucination': {
                    'within_sequence_index': {}
                }
            }

        self.results['mink'] = {
            'hallucination': {},
            'pre_hallucination': {},
            'no_hallucination': {}
        }

        # Init output dict
        self.model_analysis = {self.model_name: {}, 'id': {}}

        for score_key in self.auroc_score_keys:
            self.model_analysis[self.model_name][score_key] = None
            self.model_analysis['id'][score_key] = {}

        self.model_analysis[self.model_name]['mink'] = None

        # Gather aurocs and mink for different types of hallucination
        self._analyse_model()

    def _analyse_model(self) -> None:
        """
        Gathers global AUROC assets and response wise AUROC for hallucinations across 
        all samples in the model data.
        """
        assert len(self.model_data
                   ) > 0, "No data to process. Go check the given dataset_dir"
        for entry in tqdm(self.model_data,
                          position=0,
                          leave=True,
                          total=len(self.model_data),
                          desc="Processing Samples",
                          unit="sample"):
            # Reproduce logits
            logits_path = os.path.join(self.model_data_dir,
                                       f"outputs/{entry['id']}", "logits.pt")
            logits = load_logits(logits_path)

            # Get ground truth labels
            hallucination_mask = torch.Tensor(get_ground_truth_list(entry))

            # Get sampled token indices
            tokenizer = get_tokenizer(self.model_name, config.hf_cache_dir)
            response_token_ids = tokenizer.encode(entry['response'],
                                                  return_tensors='pt')

            # Identify indices of hallucinations
            all_hallucination_ranges = entry['ground_truth']['annotation']
            hallucination_ranges = []

            # Check if this entry is relevant for the sequence scope
            if len(all_hallucination_ranges) - 1 >= self.sequence_scope:
                if self.sequence_scope == -1:
                    hallucination_ranges = all_hallucination_ranges
                elif self.sequence_scope < 2:
                    hallucination_ranges = [
                        all_hallucination_ranges[self.sequence_scope]
                    ]
                else:
                    hallucination_ranges = all_hallucination_ranges[
                        self.sequence_scope:]

            # Get global AUROC assets + entry wise auroc
            for score_key in self.auroc_score_keys:
                # Get sampled scores (out of all) for RAGTruth response
                if score_key in ['sampled_logit', 'sampled_probability']:
                    scores = apply_score_function(
                        logits=logits,
                        score_key=score_key,
                        response_token_ids=response_token_ids)
                else:
                    scores = apply_score_function(logits=logits,
                                                  score_key=score_key)
                self._gather_global_auroc_assets(score_key, entry, scores,
                                                 hallucination_mask,
                                                 hallucination_ranges)

            # Get Min-K input scores
            # Infer wether hallucination is containes in the entries response
            if len(all_hallucination_ranges) == 0:
                non_hallucination_category = 'no_hallucination'
                end_index = entry['ground_truth']['total_tokens']
            else:
                non_hallucination_category = 'pre_hallucination'
                if all_hallucination_ranges[0][0] > 0:
                    end_index = entry['ground_truth']['annotation'][0][0]

            # Obtain the relevant scores for hallucination and non-hallucination tokens
            self._gather_global_mink_assets('logits',
                                            non_hallucination_category, logits,
                                            hallucination_ranges, end_index)
            self._gather_global_mink_assets('response_token_ids',
                                            non_hallucination_category,
                                            response_token_ids[0],
                                            hallucination_ranges, end_index)

        # Compute model-wide metrics
        print('Global AUROC computation ...')
        self._get_aurocs()

        print('Global Min-K computation...')
        self._get_minks()

    def _gather_global_auroc_assets(self, score_key: str, entry: dict,
                                    scores: torch.Tensor,
                                    hallucination_mask: torch.Tensor,
                                    hallucination_ranges: list):
        """
        Gather all uncertainty scores for a given entries hallucinations in different subsets (Sequence, Token).
        """
        # Get non-hallucination scores & add to global list
        non_hallucination_token_scores = scores[hallucination_mask == 0]
        self.scores[score_key]['non'].extend(
            list(non_hallucination_token_scores))

        # Get all hallucination scores & add to global list
        all_hallucination_token_scores = []

        for start_index, end_index in hallucination_ranges:
            all_hallucination_token_scores.extend(
                scores[start_index:end_index])
        self.scores[score_key]['all'].extend(
            list(all_hallucination_token_scores))

        # Skip entries that do not contain hallucination
        if len(hallucination_ranges) > 0:
            # Sequence level analysis
            if self.sequence_scope == -1:  # All hallucinations are analysed
                sequence_analyser = SequenceAurocAnalyser(
                    scores, deepcopy(hallucination_ranges),
                    non_hallucination_token_scores,
                    all_hallucination_token_scores)
                sequence_analysis = sequence_analyser.to_dict()
                del sequence_analyser

                self.scores[score_key]['first_sequence'].extend(
                    list(sequence_analysis['first_sequence']))
                self.scores[score_key]['subsequent_sequences'].extend(
                    list(sequence_analysis['subsequent_sequences']))

            # Token level analysis
            token_analyser = TokenAurocAnalyser(
                scores, deepcopy(hallucination_ranges),
                non_hallucination_token_scores)
            token_analysis = token_analyser.to_dict()
            del token_analyser

            # Add ith hallucination tokens
            max_token_index = max(
                filter_for_integers(list(token_analysis.keys())))
            for index_key in range(max_token_index):
                if str(index_key) not in self.scores[score_key][
                        'within_sequence_index'].keys():
                    self.scores[score_key]['within_sequence_index'][str(
                        index_key)] = token_analysis[str(index_key)]
                else:
                    self.scores[score_key]['within_sequence_index'][str(
                        index_key)].extend(token_analysis[str(index_key)])

            entry_data = {
                'sequence':
                sequence_analysis['aurocs']
                if self.sequence_scope == -1 else None,
                'token':
                token_analysis['aurocs']
            }

            # Store entry specific auroc scores
            self.model_analysis['id'][score_key][entry['id']] = entry_data

            if self.sequence_scope == -1:
                del sequence_analysis
            del token_analysis

    def _gather_global_mink_assets(self, score_key: str,
                                   non_hallucination_category: str,
                                   scores: torch.Tensor,
                                   hallucination_ranges: list, end_index: int):
        """
        Gather all hallucination and non-hallucination scores for a given entry.
        Specifically, store all ith non hallucination scores for entries without hallucination
        & store all ith hallucination scores up to the first_sequence hallucination 
          for entries containing hallucination.
        """
        mink_analyser = MinkTokenAnalyser(non_hallucination_category,
                                          deepcopy(scores),
                                          hallucination_ranges, end_index)
        mink_analysis = mink_analyser.to_dict()
        del mink_analyser

        # Add ith hallucination tokens
        for key in mink_analysis.keys():
            token_indices = filter_for_integers(list(
                mink_analysis[key].keys()))
            if len(token_indices) > 0:
                max_token_index = max(token_indices)
                for index_key in range(max_token_index):
                    if str(index_key) not in self.scores[score_key][key][
                            'within_sequence_index'].keys():
                        self.scores[score_key][key]['within_sequence_index'][
                            str(index_key)] = mink_analysis[key][str(
                                index_key)]
                    else:
                        self.scores[score_key][key]['within_sequence_index'][
                            str(index_key)].extend(
                                mink_analysis[key][str(index_key)])

    def _get_aurocs(self) -> None:
        """
        Computes global binary AUROC for hallucination uncertainty scores at both sequence and token level.
        """
        for score_key in self.auroc_score_keys:
            # All Hallucination Tokens
            self.results['auroc'][score_key]['all'] = get_auroc(
                self.scores[score_key]['all'], self.scores[score_key]['non'])

            # Sequence level differentiation
            self.results['auroc'][score_key]['first_sequence'] = get_auroc(
                self.scores[score_key]['first_sequence'],
                self.scores[score_key]['non'])
            self.results['auroc'][score_key][
                'subsequent_sequences'] = get_auroc(
                    self.scores[score_key]['subsequent_sequences'],
                    self.scores[score_key]['non'])
            self.results['auroc'][score_key][
                'first_vs_subsequent_sequences'] = get_auroc(
                    self.scores[score_key]['first_sequence'],
                    self.scores[score_key]['subsequent_sequences'])

            for token_index in list(self.scores[score_key]
                                    ['within_sequence_index'].keys())[0:]:
                sample_size = len(self.scores[score_key]
                                  ['within_sequence_index'][token_index])
                if sample_size < self.sample_size_threshold:
                    break
                self.results['auroc'][score_key][
                    'auroc_within_sequence_index'][token_index] = {
                        'sample_size':
                        len(self.scores[score_key]['within_sequence_index']
                            [token_index]),
                        score_key: {
                            'vs_non':
                            get_auroc(
                                self.scores[score_key]['within_sequence_index']
                                [token_index], self.scores[score_key]['non'])
                        }
                    }

    def _get_minks(self) -> None:
        """
        Computes global Min-K scores per in-sequence hallucination token index as well as for pre-hallucination
        and no hallucination tokens.
        """
        for non_hallucination_category in [
                'hallucination', 'pre_hallucination', 'no_hallucination'
        ]:
            for token_index in list(
                    self.scores['logits'][non_hallucination_category]
                ['within_sequence_index'].keys()):
                assert self.scores['logits'][non_hallucination_category]['within_sequence_index'].keys() == \
                    self.scores['response_token_ids'][non_hallucination_category]['within_sequence_index'].keys(), \
                    "Token of {non_hallucination_category} response_token_ids and logits do not match"

                logits = self.scores['logits'][non_hallucination_category][
                    'within_sequence_index'][token_index]
                response_token_ids = self.scores['response_token_ids'][
                    non_hallucination_category]['within_sequence_index'][
                        token_index]
                mink = get_mink(logits, response_token_ids)
                if mink is not None:
                    self.results['mink'][non_hallucination_category][
                        token_index] = {
                            'sample_size':
                            len(self.scores['logits']
                                [non_hallucination_category]
                                ['within_sequence_index'][token_index]),
                            'ratios':
                            get_mink(logits, response_token_ids)
                        }

    def to_dict(self) -> dict:
        """
        Converts the analysis results to a dictionary.
        
        Returns:
            dict: Adds global aurocs for hallucination tokens and sequences and minks to the results dictionary.
        """
        auroc_error = False

        for score_key in list(self.model_analysis[self.model_name].keys()):
            if score_key == 'mink':
                try:
                    self.model_analysis[self.model_name][score_key] = {
                        'hallucination':
                        self.results[score_key]['hallucination'],
                        'pre_hallucination':
                        self.results[score_key]['pre_hallucination'],
                        'no_hallucination':
                        self.results[score_key]['no_hallucination'],
                    }
                except:
                    print('Not enough samples for mink analysis')
            else:
                try:
                    self.model_analysis[self.model_name][score_key] = {
                        'sequence': {
                            'all_vs_non':
                            self.results['auroc'][score_key]['all'],
                            'first_sequence_vs_non':
                            self.results['auroc'][score_key]['first_sequence'],
                            'subsequent_sequences_vs_non:':
                            self.results['auroc'][score_key]
                            ['subsequent_sequences'],
                            'first_sequence_vs_subsequent_sequences':
                            self.results['auroc'][score_key]
                            ['first_vs_subsequent_sequences']
                        },
                        'token': {}
                    }
                    for token_index in list(
                            self.results['auroc'][score_key]
                        ['auroc_within_sequence_index'].keys()):
                        self.model_analysis[self.model_name][score_key]['token'][token_index] = \
                            self.results['auroc'][score_key]['auroc_within_sequence_index'][token_index]
                except:
                    auroc_error = True

        if auroc_error:
            print('No hallucination sequences in the given sample id set.')

        return self.model_analysis
