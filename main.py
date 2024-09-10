from ollama_client.llm_client import LanguageModelClient
from asyncio import run as asyncrun

async def main():
    model = 'gemma2:2b'
    llm = LanguageModelClient(model)
    message = "Tell me a good joke"
    await llm.chat(message)
if __name__ == '__main__':
    asyncrun(main())