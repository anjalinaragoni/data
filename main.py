from data.data_loader import DataLoader
from data.model import SummarizerModel
from data.pipeline import SummarizationPipeline
from data.evaluator import Evaluator

if __name__ == "__main__":
    loader = DataLoader()
    model = SummarizerModel("facebook/bart-base")
    pipeline = SummarizationPipeline(loader, model)
    evaluator = Evaluator()

    results = pipeline.run(10)
    refs = [r["reference_summary"] for r in results]
    preds = [r["generated_summary"] for r in results]

    scores = evaluator.evaluate(refs, preds)
    for i, r in enumerate(results):
        print(f"\nSample {i+1}:\nDialogue: {r['dialogue']}\nReference: {r['reference_summary']}\nGenerated: {r['generated_summary']}")
        print(f"ROUGE Scores: {scores[i]}")
