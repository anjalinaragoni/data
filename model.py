from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class SummarizerModel:
    def __init__(self, model_name="facebook/bart-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def summarize(self, text, max_input_len=1024, max_output_len=60):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=max_input_len)
        summary_ids = self.model.generate(inputs["input_ids"], max_length=max_output_len, num_beams=4)
        return self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
