from tqdm.notebook import tqdm

import pandas as pd

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
    def __init__(self, model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        super(LlamaNode, self).__init__(
            model_name=self.model_name
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

async def save_answers_to_csv(
    answers, 
    path = "model_responses_and_gold_responses.csv", 
    print_outputs=True
):
    formated_answers = []
    pbar = tqdm(desc="Saving answers", unit=" answers")
    async for answer in answers:
        answer = {
            "Question": answer.data["question"],
            "Gold Response": answer.data["expected_output"],
            "Model Response": answer.response["answer"],
        }
        if print_outputs:
            print("Question: " + answer["Question"])
            print("Gold Response:" + answer["Gold Response"])
            print("Model Response: " + answer["Model Response"])
        formated_answers.append(answer)
        pbar.update()

    formated_answers = pd.DataFrame(formated_answers)
    formated_answers.to_csv(path, index = False)
    return formated_answers
    