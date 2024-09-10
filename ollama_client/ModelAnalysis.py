import time
from collections import defaultdict
from dataclasses import dataclass

import ollama

from ollama_client.llm_client import LanguageModelClient
from ollama_client.utils import create_random_string, print_stacktrace


@dataclass
class OutputResponse:
    result: str
    response_time: float

class ModelAnalysis:
    def __init__(self, base_models=None):
        self.base_models = base_models if base_models else []

    def add_model(self, base_model: str) -> None:
        """
        :param base_model:
        :return:
        """
        self.base_models.append(base_model)

    async def _get_default_responses(self, texts: list[str]=None, system_instr: str="") -> dict[str, list[OutputResponse]]:
        if not texts:
            raise ValueError("expected texts to be non-empty")
        responses = defaultdict(list)
        for base_model in self.base_models:
            current_model_file = f'''
            FROM {base_model}
            SYSTEM """{system_instr}"""
            '''
            model_name = create_random_string(16)
            ollama.create(model=model_name, modelfile=current_model_file)
            llm = LanguageModelClient(model_name)
            for text in texts:
                start_time = time.monotonic()
                response = await llm.chat(text)
                end_time = time.monotonic()
                runtime = end_time - start_time
                responses[base_model].append(OutputResponse(result=response, response_time=runtime))
                llm.reset_chat()
            ollama.delete(model_name)
        return responses

    async def run_text_generation_test(self, texts: list[str]=None):
        texts = texts if texts else ['hello']
        responses = await self._get_default_responses(texts)
        await self._print_responses('Text Generation', responses, texts)

    async def run_sentiment_analysis_test(self, texts: list[str]=None):
        texts = texts if texts else ['hello']
        schema = '{input_text: string, overall_sentiment_score: number, sentences: {sentence_text: string, sentiment_score: number}[]}'
        system_instr = f'''Your task is to perform sentimental analysis. For each response from the user:
        1. Rate the entire text from -1 to 1 where -1 is the most negative, 1 is the most positive, and 0 is neutral.
        2. Evaluate the sentiment scores for each sentence with scores above
        3. Return ONLY the output of your result as a JSON in the following schema: {schema}
        4. Expect your response to only contain the JSON response
        5. MUST NOT produce markdown embedding
        '''
        responses = await self._get_default_responses(texts, system_instr)
        await self._print_responses('Sentiment Analysis', responses, texts)

    async def _print_responses(self, test_type, responses, texts):
        for model, text_responses in responses.items():
            print('=' * 50)
            print(f'Test Type: {test_type}')
            print('-' * 50)
            print(f'Model: {model}')
            for text, responses, in zip(texts, text_responses):
                print(f'original text: {text}')
                print(f'response time: {responses.response_time} s')
                print(f'result: {responses.result}')
            print('=' * 50)

    async def run_summarization_test(self, texts: list[str]=None):
        texts = texts if texts else ['hello']
        schema = '{summary: string, topic: string}'
        system_instr = f'Respond each of the responses as in the following JSON schema: {schema}. The summary field should summarize the entire text concisely and topic should be less than five words. MUST NOT produce markdown embedding'
        responses = await self._get_default_responses(texts, system_instr)
        await self._print_responses('Summarization', responses, texts)