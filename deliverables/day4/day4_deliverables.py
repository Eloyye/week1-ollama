import asyncio

import ollama

from ollama_client.llm_client import LanguageModelClient


async def chat_template(modelfile, query):
    updated_model_name = 'medical-bot'
    ollama.create(updated_model_name, modelfile=modelfile)
    model = LanguageModelClient(updated_model_name)
    result = await model.chat(query)
    print(f'Q: {query}')
    print(f'A: {result}')
    ollama.delete(updated_model_name)

async def prompt_enhanced():
    modelfile = '''You are a doctor trying to diagnose the patient and recommend appropriate tests and next steps for the patients. 
        Respond in a professional manner.
        If a patient mentions a probable cause of illness, follow up with further questions for confirmation or give appropriate tests. 
        Otherwise if a patient just list their symptoms, try to make some clarifications by following up with more specific questions. Until you are certain
        proceed with a formal diagnosis of which disease it could be. Be specific in the type of test you would conduct and answer why.
        If the patient gives very little information do not name the disease. For any questions the patient gives, make sure to properly answer the question.'''
    query = """Given the following information: When the body has too much of the hormone cortisol, over time, that can lead to Cushing syndrome.
    Cortisol plays many roles throughout the body. It helps control blood pressure, lowers inflammation, and keeps the heart and blood vessels working properly. Cushing syndrome may develop when the body has too much cortisol due to taking glucocorticoid medicine. Or the body might make too much cortisol because of a tumor or another medical problem.
    Cushing syndrome can cause a wide variety of symptoms, depending on how much extra cortisol is in the body. Some common symptoms include a fatty hump between the shoulders, a rounded face, and pink or purple stretch marks on the skin. Cushing syndrome also can lead to high blood pressure, bone loss and, in some cases, type 2 diabetes.
    The best treatment for each individual depends on the cause of Cushing syndrome. If too much glucocorticoid medicine is the source of the problem, lowering the dosage may help. If a tumor is causing Cushing syndrome, surgery to remove it may be needed. In some cases, radiation therapy or medicine to control cortisol production might be options. 
    Answer the following question from patient: Doctor, I've been feeling really weak lately and my heart rate seems to have decreased. Do you have any idea what could be causing this?"""
    await chat_template(modelfile, query)
async def hyperparameter_tuned():
    modelfile = '''FROM gemma-2-2b-chatdoctor:latest
        SYSTEM """You are a doctor trying to diagnose the patient and recommend appropriate tests and next steps for the patients. 
        Respond in a professional manner.
        If a patient mentions a probable cause of illness, follow up with further questions for confirmation or give appropriate tests. 
        Otherwise if a patient just list their symptoms, try to make some clarifications by following up with more specific questions. Until you are certain
        proceed with a formal diagnosis of which disease it could be. Be specific in the type of test you would conduct and answer why.
        If the patient gives very little information do not name the disease. For any questions the patient gives, make sure to properly answer the question.
        """ 
        PARAMETER temperature 0.4
        PARAMETER mirostat_eta 0.2
        PARAMETER top_k 30
        PARAMETER top_p 0.7
        '''
    query = """Doctor, I've been experiencing Coryza lately, any idea what could be causing it?"""
    await chat_template(modelfile, query)
async def optimal_tuned():
    modelfile = '''FROM gemma-2-2b-chatdoctor:latest
        SYSTEM """You are a doctor trying to diagnose the patient and recommend appropriate tests and next steps for the patients. 
        Respond in a professional manner.
        Try to be more inquisitive and ask for more information about the patient's symptoms.
        Be specific in the type of test you would conduct and answer why.
        You are more certain if given a lot more information.
        If the patient gives very little information do not name the disease. For any questions the patient gives, make sure to properly answer the question.
        """ 
        PARAMETER temperature 0.4
        PARAMETER mirostat_eta 0.2
        PARAMETER top_k 30
        PARAMETER top_p 0.7
        '''
    query = """Answer this from patient: Doctor, I've been experiencing Coryza lately, any idea what could be causing it?"""
    await chat_template(modelfile, query)

async def main():
    # await prompt_enhanced()
    # await hyperparameter_tuned()
    await optimal_tuned()


if __name__ == '__main__':
    asyncio.run(main())