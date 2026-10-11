from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from dotenv import load_dotenv


load_dotenv()  # Load environment variables from .env file

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite",
    temperature=0.7
)


template = PromptTemplate(
    template="Hey you have to give me summary, semantic(positive ,negative, neutral), key themes, pros and cons of this review {review}. \n\n {Format_instructions}",
    input_variables=["review"],
    partial_variables={"Format_instructions": JsonOutputParser().get_format_instructions()}
)



parser = JsonOutputParser()
chain= template | llm | parser


result = chain.invoke({"review": """
            I recently upgraded to the Samsung Galaxy S24 Ultra, and I must say, it’s an absolute powerhouse! The Snapdragon 8 Gen 3 processor makes everything lightning fast—whether I’m gaming, multitasking, or editing photos. The 5000mAh battery easily lasts a full day even with heavy use, and the 45W fast charging is a lifesaver.
             
             The S-Pen integration is a great touch for note-taking and quick sketches, though I don't use it often. What really blew me away is the 200MP camera—the night mode is stunning, capturing crisp, vibrant images even in low light. Zooming up to 100X actually works well for distant objects, but anything beyond 30X loses quality.
             
             However, the weight and size make it a bit uncomfortable for one-handed use. Also, Samsung’s One UI still comes with bloatware—why do I need five different Samsung apps for things Google already provides? The $1,300 price tag is also a hard pill to swallow.
             
             Pros:
             Insanely powerful processor (great for gaming and productivity)
             Stunning 200MP camera with incredible zoom capabilities
             Long battery life with fast charging
             S-Pen support is unique and useful
             
             Cons:
             Bulky and heavy—not great for one-handed use
             Bloatware still exists in One UI
             Expensive compared to competitors
"""})
print("Structured Output : \n\n",result)

# if you do not have dynamic input in template then if you call invoke then you still have to pass a empty dictionary as input to invoke method otherwise it will throw error.

 


