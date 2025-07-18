import torch

from utils import get_auroc


class AurocAnalyser:
    """
    Base class to analyse a models outputs with respect to AUROC.
    
    Attributes:
        scores: The scores output from the model.
        hallucination_ranges: The ranges of tokens annotated as hallucinations.
        all_non_halucination_token_scores: The scores of tokens that are not hallucinations.
    """

    def __init__(self, scores: torch.Tensor,
                 hallucination_ranges: list[tuple[int]],
                 all_non_halucination_token_scores: torch.Tensor):
        self.scores = scores
        self.hallucination_ranges = hallucination_ranges

        # Non-halluciation
        self.all_non_halucination_token_scores = all_non_halucination_token_scores


class SequenceAurocAnalyser(AurocAnalyser):
    """
    NOT PART OF PAPER
    Analyses model outputs with respect to annotated hallucinations on sequence-level. 
    
    Example: 'El-on-Mu-sk-is-a-[rep]-[ti]-[lo]-[ide] and has a [spl]-[it]-[ton]-[gue]'
    [] marking hallucinations, this compares prediction scores of 
       1. [rep]-[ti]-[lo]-[ide] vs. [spl]-[it]-[ton]-[gue] 
       2. [rep]-[ti]-[lo]-[ide] vs. non-hallucination tokens
       3. [spl]-[it]-[ton]-[gue] vs. non-hallucination tokens
       + storing of the relevant tokens for dataset-wide, analogous AUROC.
    """

    def __init__(self, scores: torch.Tensor,
                 hallucination_ranges: list[tuple[int]],
                 all_non_halucination_token_scores: torch.Tensor,
                 all_hallucination_token_scores: torch.Tensor):
        super().__init__(scores, hallucination_ranges,
                         all_non_halucination_token_scores)

        self.all_hallucination_sequence_scores = all_hallucination_token_scores

        # Init Hallucination sequence-level
        self.first = []
        self.subsequent = []

        self.first_hallucination_sequence_prediction = None
        self.subsequent_hallucination_sequence_predictions = []
        self.first_vs_subsequent_hallucination_prediction = None
        self.all_hallucination_sequence_prediction = None

        # Analysis
        if len(self.hallucination_ranges) > 0:
            self._analyse_hallucination_sequences()

    def _analyse_hallucination_sequences(self) -> None:
        """
        Analyses hallucination on sequence-level and gathers local AUROC scores. 
        Details see class description.
        """
        assert len(self.hallucination_ranges
                   ) > 0, "The given entry does not contain hallucination."
        start, end = self.hallucination_ranges.pop(0)

        # Store first hallucination sequence
        self.first = self.scores[start:end]

        # Extract and analyse subsequent hallucination tokens
        for (start, end) in self.hallucination_ranges:
            # Store subsequent hallucination sequence
            subsequent_sequence_scores = self.scores[start:end]
            self.subsequent.extend(subsequent_sequence_scores)

            # Compute first hallucination sequence vs all subsequent hallucination prediction score
            self.first_vs_subsequent_hallucination_prediction = get_auroc(
                self.first, self.subsequent)

            # Compute subsequent hallucination sequence prediction score
            subsequent_hallucination_auroc = get_auroc(
                subsequent_sequence_scores,
                self.all_non_halucination_token_scores)
            self.subsequent_hallucination_sequence_predictions.append(
                subsequent_hallucination_auroc)

        # Compute first hallucination sequence vs non hallucination prediction score
        self.first_hallucination_sequence_prediction = get_auroc(
            self.first, self.all_non_halucination_token_scores)
        # Compute all hallucination sequences vs non hallucination prediction score
        self.all_hallucination_sequence_prediction = get_auroc(
            self.all_hallucination_sequence_scores,
            self.all_non_halucination_token_scores)

    def to_dict(self) -> dict:
        """
        Converts the analysis results to a dictionary.
        """
        aurocs = {
            'all_vs_non':
            self.all_hallucination_sequence_prediction,
            'first_vs_non':
            self.first_hallucination_sequence_prediction,
            'subsequent_vs_non':
            self.subsequent_hallucination_sequence_predictions,
            'first_vs_subsequent':
            self.first_vs_subsequent_hallucination_prediction
        }

        sequence_analysis = {
            'first_sequence': self.first,
            'subsequent_sequences': self.subsequent
        }

        sequence_analysis['aurocs'] = aurocs

        return sequence_analysis


class TokenAurocAnalyser(AurocAnalyser):
    """
    Analyses model outputs with respect to annotated hallucinations on token-level 
    and computes respective AUROC.
    
    Example: 'El-on-Mu-sk-is-a-[rep]-[ti]-[lo]-[ide] and has a [spl]-[it]-[ton]-[gue].'
    [] marking hallucinations, this compares prediction scores of 
       1. [rep] vs. [ti]-[lo]-[ide] 
       2. [spl] vs. [it]-[ton]-[gue] 
       3. [rep] vs. non-hallucination tokens
       4. [spl] vs. non-hallucination tokens
       5. [rep, spl] vs. non-hallucination tokens
       6. [ti, lo, ide, it, ton, gue] vs. non-hallucination tokens
       7. [rep, spl] vs. [ti, lo, ide, it, ton, gue]
       + storing of the relevant tokens for dataset-wide, analogous AUROC
    """

    def __init__(self, scores: torch.Tensor,
                 hallucination_ranges: list[tuple[int]],
                 all_non_halucination_token_scores: torch.Tensor):
        super().__init__(scores, hallucination_ranges,
                         all_non_halucination_token_scores)

        # Dict to store the single token scores on dfferent analysis levels
        self.hallucination_storage = {
            'global': {},
            'sequence': {},
            'response': {}
        }

        # Analysis
        if len(self.hallucination_ranges) > 0:
            self._analyse_hallucination_tokens()

    def _analyse_hallucination_tokens(self) -> None:
        """
        Analyses hallucination on token-level and gathers local AUROCs. 
        Details see class description.
        """
        assert len(self.hallucination_ranges) > 0, \
            "The given entry does not contain hallucination. No need for filtering"
        hallucination_token_response_dict = {}

        # Extract and analyse first and conditional hallucination tokens
        for start, end in self.hallucination_ranges:
            for hallucination_token_index in range(end - start):
                index_key = str(hallucination_token_index)
                hallucination_token_score = self.scores[
                    start + hallucination_token_index].item()

                if index_key not in self.hallucination_storage['global'].keys(
                ):
                    self.hallucination_storage['global'][index_key] = []

                # Store ith hallucination token for global analysis
                self.hallucination_storage['global'][index_key].extend(
                    [hallucination_token_score])

                if index_key not in hallucination_token_response_dict.keys():
                    hallucination_token_response_dict[index_key] = []

                # Store ith hallucination token for response level analysis
                hallucination_token_response_dict[index_key].extend(
                    [hallucination_token_score])

                # Compute AUROC of hallucination token vs. non-hallucination tokens
                self.hallucination_storage['sequence'][index_key] = {
                    'vs_non':
                    get_auroc([hallucination_token_score],
                              self.all_non_halucination_token_scores)
                }

        # Compute response-level AUROC of ith vs non hallucination scores
        for index_key in hallucination_token_response_dict.keys():
            self.hallucination_storage['response'][index_key] = {
                'vs_non':
                get_auroc(hallucination_token_response_dict[index_key],
                          self.all_non_halucination_token_scores)
            }

    def to_dict(self) -> dict:
        """
        Converts the analysis results to a dictionary.
        
        Returns:
            dict: A dictionary containing the scores (for dataset-wide analysis) 
                  and computed AUROC (for local analysis).
        """
        # AUROCs on response and local level (per response)
        aurocs = {'response-level': {}, 'sequence-level': {}}

        # Dict to collect tokens and AUROCs for model wide analysis
        token_analysis = {}

        # Add 'ith Token' AUROCs & token lists
        for index_key in self.hallucination_storage['global'].keys():
            # Add AUROCs on response and sequence level
            aurocs['response-level'][index_key] = self.hallucination_storage[
                'response'][index_key]
            aurocs['sequence-level'][index_key] = self.hallucination_storage[
                'sequence'][index_key]

            # Add token lists
            token_analysis[index_key] = self.hallucination_storage['global'][
                index_key]

        token_analysis['aurocs'] = aurocs

        return token_analysis
