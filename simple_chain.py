from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

prompt = PromptTemplate(
    template = "Write a small description about the best players of the following sport : {sport}",
    input_variables = ['sport']
) 

model = ChatGoogleGenerativeAI(model = "gemini-2.5-flash")

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({'sport' : 'football'})

print(result)