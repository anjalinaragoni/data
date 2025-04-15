class SummarizationPipeline:
    def __init__(self, data_loader, summarizer):
        self.data_loader = data_loader
        self.summarizer = summarizer

    def run(self, sample_count=5):
        dataset = self.data_loader.load_data()
        results = []
        for sample in dataset.select(range(sample_count)):
            input_text = sample["dialogue"]
            reference = sample["summary"]
            generated = self.summarizer.summarize(input_text)
            results.append({
                "dialogue": input_text,
                "reference_summary": reference,
                "generated_summary": generated
            })
        return results
