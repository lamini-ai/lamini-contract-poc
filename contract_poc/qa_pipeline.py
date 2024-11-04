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

class QuestionAnswerPipeline(GenerationPipeline):
    def __init__(self, question_system_prompt: str = ""):
        super(QuestionAnswerPipeline, self).__init__()

        self.question_generator = QuestionGenerator(system_prompt=question_system_prompt)
        self.answer_generator = AnswerGenerator()

    def forward(self, x):
        x = self.question_generator(x, output_type={
            "question_1": "str",
            "question_2": "str",
            "question_3": "str",
        })
        x = self.answer_generator(x, output_type={"answer":"str"})
        return x

class QuestionGenerator(GenerationNode):
    def __init__(self, system_prompt = "Ask three separate questions around a fact involving any numbers within this text: "):
        self.system_prompt = system_prompt
        super(QuestionGenerator, self).__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct"
        )
        
    def preprocess(self, obj: PromptObject):
        obj.prompt = self.make_prompt(obj)

    def postprocess(self, obj: PromptObject):
        response = obj.response
        questions = [
            response["question_1"],
            response["question_2"],
            response["question_3"],
        ]
        for question in questions:
            q_data = obj.data.copy()
            q_data["question"] = question
            ans = PromptObject(prompt=question, data=q_data)
            yield ans


    def make_prompt(self, obj):
        prompt = (
            self.system_prompt
        )
        prompt += obj.data["content"]
        #print(prompt)

        return prompt

class AnswerGenerator(GenerationNode):
    def __init__(self):
        super(AnswerGenerator, self).__init__(
            model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_new_tokens=150
        )

    def preprocess(self, obj: PromptObject):
        obj.data["question"] = obj.prompt
        obj.prompt = self.make_prompt(obj)

    def make_prompt(self, obj: PromptObject):
        prompt = "Consider the following:\n\n"
        prompt += obj.data["content"]
        prompt += "\n Use the content above to answer the following question:\n"
        prompt += obj.prompt
        if obj.prompt[-1] != "?":
            prompt += "?"
        prompt += "\n\n"
        prompt += "If the text does not contain the answer, then respond with 'I do not know'."
        return prompt


async def load_qa_prompts(prompts):

    for idx, page in enumerate(prompts):
        yield PromptObject(
            prompt="", data={
                "content":page
            }
        )

async def save_answers(answers, path = "generated_q_a_from_pdf.jsonl", print_outputs=True):
    with jsonlines.open(path, "w") as writer:
        pbar = tqdm(desc="Saving answers", unit=" answers")
        async for answer in answers:
            answer = {
                "prompt": answer.prompt,
                "question": answer.data["question"],
                "answer": answer.response["answer"],
            }
            if print_outputs:
                print("Question: " + answer["question"])
                print("Prompt:" + answer["prompt"])
                print("Answer: " + answer["answer"])
            writer.write(answer)
            pbar.update()


async def run_pipeline():
    pdf_pages = load_qa_prompts(PDFLoader("city_of_turlock.pdf", PageChunker(), batch_size = 1))
    answers = QuestionAnswerPipeline().call(pdf_pages)
    await save_answers(answers)