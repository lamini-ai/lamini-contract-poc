from tqdm.notebook import tqdm

import lamini
import jsonlines

from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.embedding_node import EmbeddingNode
from lamini.generation.generation_node import GenerationNode
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.index.lamini_index import LaminiIndex

from typing import AsyncIterator, Iterator, Union

class SummaryPipeline(GenerationPipeline):
    def __init__(self, question_system_prompt: str = ""):
        super(SummaryPipeline, self).__init__()

        self.summary_generator = SummaryGenerator()

    def forward(self, x):
        x = self.summary_generator(x, output_type={"summary":"str"})
        return x

class SummaryGenerator(GenerationNode):
    def __init__(self):
        super(SummaryGenerator, self).__init__(
            model_name="meta-llama/Meta-Llama-3.1-8B-Instruct"
        )

    def preprocess(self, obj: PromptObject):
        obj.prompt = self.make_prompt(obj)

    def make_prompt(self, obj: PromptObject):
        prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Consider the following:\n\n"
        prompt += obj.data["content"]
        prompt += "\n Summarize the content above in a singular sentence."
        prompt += "<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
        return prompt

async def save_summaries(answers, path, print_outputs=True):
    with jsonlines.open(path, "w") as writer:
        pbar = tqdm(desc="Saving answers", unit=" answers")
        async for answer in answers:
            answer = {
                "summary": answer.response["summary"],
                "content": answer.data["content"],
                "input": answer.response["summary"],
                "output": answer.data["content"]
            }
            if print_outputs:
                print("Summary: " + answer["summary"])
                print("Content:" + answer["content"])
            writer.write(answer)
            pbar.update()
