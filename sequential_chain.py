from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

str_parser = StrOutputParser()

prompt1 = PromptTemplate(
    template = "Write a detailed description on the following topic: {topic} in almost 200 words. ",
    input_types = ["topic"]
)

prompt2 = PromptTemplate(
    template = "Give a summary on the given description : {description}",
    input_variables = ["description"]
)

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

chain = prompt1 | model | str_parser | prompt2 | model | str_parser

result = chain.invoke({"topic" : "football"})
