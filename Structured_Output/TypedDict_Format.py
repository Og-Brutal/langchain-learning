# if i inherit from typed dict then i can use the class object as dictionary and used for validating format

from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict,Annotated,Optional,Literal

load_dotenv()  # Load environment variables from .env file


#Schema or format of structured output which i need from LLM
class StructuredOutput(TypedDict):
    summary: Annotated[str, "A brief summary of the reveiw"]
    sentiment: Annotated[Literal["positive", "negative", "neutral"], "Sentiment of the review"]
    key_themes: Annotated[list[str], "List of key themes or topics mentioned in the review"]
    pros: Annotated[Optional[list[str]], "List of positive aspects mentioned in the review"]
    cons: Annotated[Optional[list[str]], "List of negative aspects mentioned in the review"]


llm=ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.7)

structured_llm=llm.with_structured_output(StructuredOutput)

result=structured_llm.invoke(""" 
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
""")

print("Structured Output : \n\n", result)
print("Summary: \n\n", result["summary"])
print("Sentiment: \n\n", result["sentiment"])
print("Key Themes: \n\n", result["key_themes"])
print("Pros: \n\n", result["pros"])
print("Cons: \n\n", result["cons"])
