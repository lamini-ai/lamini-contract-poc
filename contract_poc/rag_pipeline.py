from tqdm.notebook import tqdm

import lamini
import numpy as np
import jsonlines

from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.embedding_node import EmbeddingNode
from lamini.generation.generation_node import GenerationNode
from lamini.generation.base_prompt_object import PromptObject
from lamini.generation.generation_pipeline import GenerationPipeline
from lamini.index.lamini_index import LaminiIndex

from typing import AsyncIterator, Iterator, Union, AsyncGenerator, Optional

class RAGModelStage(GenerationNode):
    def __init__(
        self, 
        model_path: str, 
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        rag_query_size=2
    ):
        super().__init__(
            model_name=model_name
        )
        self.rag_query_size = rag_query_size
        self.index = LaminiIndex.load_index(model_path)

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        *args,
        **kwargs,
    ):
        prompt = self.add_template(prompt)

        results = super().generate(
            prompt,
            output_type={"answer":"str"},
            *args,
            **kwargs,
        )

        return results

    def index_query(self, prompt, n = 4):
        embed = self.index.get_embeddings(prompt)
        _, indices = self.index.index.search(embed, n)
        return [self.index.splits[i] for i in indices[0]]
    
    async def add_template(self, prompts: Union[Iterator[PromptObject], AsyncIterator[PromptObject]]) -> AsyncGenerator[PromptObject, None]:
        async for prompt in prompts:
            results = self.index_query(prompt.prompt, n=self.rag_query_size)

            new_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>Consider the following:\n\n"
            for result in results:
                new_prompt += result + "\n\n "
            new_prompt += prompt.prompt + "\n\n"
            new_prompt += "If the text does not contain the answer, then respond with 'I do not know'.<|eot_id|><|start_header_id|>assistant<|end_header_id|>"

            yield PromptObject(prompt=new_prompt, data=prompt.data)

class EmbeddingModelStage(EmbeddingNode):
    def __init__(self):
        super().__init__()

    def generate(
        self,
        prompt: Union[Iterator[PromptObject], AsyncIterator[PromptObject]],
        model_name: Optional[str] = None,
    ) -> Union[Iterator[PromptObject], AsyncIterator[PromptObject]]:
        prompt = self.get_query_prompt(prompt)

        return super().generate(
            prompt,
            model_name=model_name,
        )

    async def get_query_prompt(self, prompts: Union[Iterator[PromptObject], AsyncIterator[PromptObject]]) -> AsyncGenerator[PromptObject, None]:
        async for prompt in prompts:
            yield PromptObject(prompt=prompt.prompt, data=prompt.data)

class RAGPipeline(GenerationPipeline):
    def __init__(
        self, 
        rag_model_path: str, 
        model_name: str = "meta-llama/Meta-Llama-3.1-8B-Instruct",
        rag_query_size: int = 2
    ):
        super().__init__()
        self.model_stage = RAGModelStage(rag_model_path, model_name=model_name, rag_query_size=rag_query_size)

    def forward(self, x: Union[Iterator[PromptObject], AsyncIterator[PromptObject]]) -> Union[Iterator[PromptObject], AsyncIterator[PromptObject]]:
        x = self.model_stage(x)
        return x