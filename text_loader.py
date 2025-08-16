from langchain_community.document_loaders import TextLoader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

parser = StrOutputParser()

loader = TextLoader('review.txt')

model = ChatGoogleGenerativeAI(model = 'gemini-2.5-flash')

doc = loader.load()

prompt = PromptTemplate(
    template = "Write a summary for the given text :- {text}",
    input_variables = ["text"]
)

chain = prompt | model | parser
result = chain.invoke({'text' : doc[0].page_content})

print(doc)

print(result)


