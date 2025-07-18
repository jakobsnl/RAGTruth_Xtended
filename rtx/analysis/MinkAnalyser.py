import torch


class MinkTokenAnalyser:
    """
    Class to gather relevant models outputs serving Min-K computation with respect to 
    annotated hallucination, pre hallucination and no hallucination tokens.
    
    Attributes:
        non_hallucination_category: pre hallucination (only gather tokens up to 1st hallucination) 
                                    or no hallucination
        scores: All scores in a response: logits or response token ids.
        hallucination_ranges: The ranges of tokens annotated as hallucinations.
        end_index: Token index of last non hallucination token before
                   (optional) first hallucination appears.
    """

    def __init__(self, non_hallucination_category: str, scores: torch.Tensor,
                 hallucination_ranges: list[tuple[int]], end_index: int):
        self.non_hallucination_category = non_hallucination_category
        self.scores = scores
        self.hallucination_ranges = hallucination_ranges
        self.end_index = end_index
        self.storage = {'hallucination': {}, non_hallucination_category: {}}

        if len(hallucination_ranges) > 0:
            self._gather_hallucination_tokens()
        self._gather_non_hallucination_tokens()

    def _gather_hallucination_tokens(self):
        """
        Gather the responses hallucination tokens.
        """
        for start, end in self.hallucination_ranges:
            for hallucination_token_index in range(end - start):
                index_key = str(hallucination_token_index)
                hallucination_token_score = self.scores[
                    start + hallucination_token_index]

                if index_key not in self.storage['hallucination'].keys():
                    self.storage['hallucination'][str(index_key)] = [
                        hallucination_token_score
                    ]
                else:
                    self.storage['hallucination'][str(index_key)].extend(
                        [hallucination_token_score])

    def _gather_non_hallucination_tokens(self):
        """
        Gather the responses non hallucination tokens.
        """
        non_halucination_scores = self.scores[:self.end_index]

        for non_hallucination_token_index in range(self.end_index):
            index_key = str(non_hallucination_token_index)
            hallucination_token_score = non_halucination_scores[
                non_hallucination_token_index]

            if str(index_key) not in self.storage[
                    self.non_hallucination_category].keys():
                self.storage[self.non_hallucination_category][str(
                    index_key)] = [hallucination_token_score]
            else:
                self.storage[self.non_hallucination_category][str(
                    index_key)].extend([hallucination_token_score])

    def to_dict(self) -> dict:
        """
        Return the Min-K analysis as a dictionary.
        """
        return self.storage
