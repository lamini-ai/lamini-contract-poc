from lamini.generation.generation_node import GenerationNode
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_pipeline import GenerationPipeline

class GenPipeline(GenerationPipeline):
    def __init__(self, model_name):
        super(GenPipeline, self).__init__()
        self.model_name = model_name
        self.generation_node = LlamaNode(model_name=self.model_name)

    def forward(self, x):
        x = self.generation_node(x, output_type={"answer":"str"})
        return x

class LlamaNode(GenerationNode):
    def __init__(self):
        super(AnswerGenerator, self).__init__(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
        )

    def preprocess(self, obj: PromptObject):
        obj.data["question"] = obj.prompt
        obj.prompt = self.make_prompt(obj)

    def make_prompt(self, obj: PromptObject):
        return (
            f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>{obj.data['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"""
        )

def simple_prompt_generator(
    df: pd.DataFrame, 
    input_col: str = "question", 
    output_col: str = "answer",
):
    for idx, row in df.iterrows():
        yield PromptObject(
            prompt = row[input_col],
            data = {
                "question": row[input_col],
                "expected_output": row[output_col],
            }
        )