import ollama
from ollama import AsyncClient

class LanguageModelClient:
    def __init__(self, model: str):
        self.model = model
        self._async_client = AsyncClient()
        self.chat_history = []
        try:
            ollama.show(self.model)
        except ollama.ResponseError as e:
            print(f'Error: {e.error}')
            if e.status_code == 404:
                ollama.pull(self.model)


    def _format_message(self, role: str='user', content: str=''):
        return {'role': role, 'content': content}

    async def chat(self, message: str, output=False) -> str:
        latest_chat_history = self.chat_history[:]
        latest_chat_history.append(self._format_message(content=message))
        result = []
        async for part in await  self._async_client.chat(model=self.model, messages=latest_chat_history, stream=True):
            contents = part['message']['content']
            if output:
                print(contents, end='', flush=True)
            result.append(contents)
        self.chat_history = latest_chat_history
        llm_output = "".join(result)
        self.chat_history.append(self._format_message(role='assistant', content=llm_output))
        return llm_output

    def reset_chat(self):
        self.chat_history = []

        