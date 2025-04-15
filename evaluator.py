from rouge_score import rouge_scorer

class Evaluator:
    def __init__(self):
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])

    def evaluate(self, references, predictions):
        scores = [self.scorer.score(ref, pred) for ref, pred in zip(references, predictions)]
        return scores
