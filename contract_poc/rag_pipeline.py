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

class AnswerPipeline(GenerationPipeline):
    def __init__(self, model_name):
        super(AnswerPipeline, self).__init__()
        self.answer_node = AnswerNode()

    def forward(self, x):
        x = self.answer_node(x, output_type={"answer":"str"})
        return x

class RetrievalNode(GenerationNode):
    def __init__(self):
        super(AnswerNode, self).__init__(model_name="meta-llama/Llama-3.2-3B-Instruct")

    def preprocess(self, obj: PromptObject):
        obj.data["question"] = obj.prompt
        obj.prompt = self.make_prompt(obj)

    def make_prompt(self, obj: PromptObject):

        return (
            f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
            Consider the following 10-K form from {obj.data['company']}:
            
            {obj.data['page_content']}
            
            Use the content above to answer the following question:
            
            {obj.data['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        )

class RetrievalNode(GenerationNode):
    def __init__(self):
        super(AnswerNode, self).__init__(model_name="meta-llama/Llama-3.2-3B-Instruct")

    def preprocess(self, obj: PromptObject):
        obj.data["question"] = obj.prompt
        obj.prompt = self.make_prompt(obj)

    def make_prompt(self, obj: PromptObject):

        return (
            f"""<|begin_of_text|><|start_header_id|>user<|end_header_id|>
            Consider the following 10-K form from {obj.data['company']}:
            
            {obj.data['page_content']}
            
            Use the content above to answer the following question:
            
            {obj.data['question']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
            """
        )