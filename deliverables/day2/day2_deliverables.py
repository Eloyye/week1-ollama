from asyncio import run as asyncrun

from ollama_client.ModelAnalysis import ModelAnalysis


async def run_sent_test(model_analysis: ModelAnalysis):
    sentiment_analysis_samples = [""" Just Bombed a React Interview

I finally managed to get an interview after tons of applications and immediate rejections. However, this was though a recruited who reached out to me. The job was for a pure frontend React position and I studied my buns off ahead of it. I've been working as a frontend dev with some backend chops for a few years now but only using Vue and PHP (mostly Laravel) so I spent a ton of time learning React through developing. In a couple weeks I built out a CMS from scratch using Next + Supabase and felt so confident going into the interview.

During the interview I crushed every React question thrown my way and used examples from my experience. Then the live coding part came... I had submitted a form on Codepen using React and walked through the code and made the updates they wanted. The last thing they wanted me to do was write a mock Promise and that's where I tripped up. So much of my experience in the last few years has been with some fetch API and not writing actual raw promises. I fumbled horribly and my confidence was shot so things got worse... Eventually they helped me through it and it worked but it was soul crushing.

I know there are a lot of products/platforms out there to help prepare for coding interviews but I don't know which to go with. I realize there's always going to be a "gotcha" part to these interviews so I want to prepare for the next one.

Does anybody have any recommendations or experiences with any of these platforms? Or even just stories of similar experiences :)
"""]
    await model_analysis.run_sentiment_analysis_test(sentiment_analysis_samples)

async def run_text_gen(model_analysis: ModelAnalysis):
    text_generation_samples = ["""Give a Python implementation the following problem: Given a linked list, swap every two adjacent nodes and return its head. You must solve the problem without modifying the values in the list's nodes (i.e., only nodes themselves may be changed.)
    Example: Input: head = [1,2,3,4], Output: [2,1,4,3]. 
    """,
                               ]
    await model_analysis.run_text_generation_test(text_generation_samples)

async def run_summarization_gen(model_analysis: ModelAnalysis):
    summary_samples = [""" My uncle got stabbed and I still laugh about it to this day.
    CONTENT WARNING: VIOLENCE/DEATH
    
    My family has a long history of domestic violence on my dad’s side. My grandfather wasn’t a nice man and neither are most of my uncles. My one uncle, let’s call him Bill for this story. My uncle Bill started dating this young girl. Young for him. He was 38 and she’s was 24. They seemed to have a good time drinking and partying but as time went on his true colors started to surface. This leads me to the day in question. My uncle Bill in his infinite wisdom decided he was going to beat this girl. He started and she bit him, she bit him in the thumb, clear down to the bone and then she stabbed him multiple times, literally in the back and left him in a bloody puddle on his front porch. The ambulance and police came. My dad called them. Bill lived but it knocked some of the arrogance out of him. She went to jail for a bit and I took care of her cat while she was in there. I haven’t seen her for years but every time I see him at thanksgiving or Christmas, I remember that this happened and I can’t help but laugh. The true definition of FAFO.
    """]

    await model_analysis.run_summarization_test(summary_samples)

async def main():
    models = ['mistral:latest', 'phi3:latest', 'gemma2:2b']
    model_analysis = ModelAnalysis(models)
    await run_text_gen(model_analysis)
    # await run_sent_test(model_analysis)
    # await run_summarization_gen(model_analysis)
if __name__ == '__main__':
    asyncrun(main())