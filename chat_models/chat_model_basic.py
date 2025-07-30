# import getpass
# import os

# if not os.environ.get("COHERE_API_KEY"):
#   os.environ["COHERE_API_KEY"] = getpass.getpass("Enter API key for Cohere: ")

# ans = os.environ.get("COHERE_API_KEY")
# print(ans)
# from langchain.chat_models import init_chat_model

# # from langchain_cohere import co

# from langchain_cohere import ChatCohere
# model = init_chat_model("command-r-plus", model_provider="cohere")

# response = model.invoke("Tell me a joke")
# print(response)



# # import cohere
# # co = cohere.Client("MjG1Whi207AR1ZRO90TulKTwFZLjHhgpc83ShM1K")
# # response = co.chat(message="Tell me a joke")
# # print(response.text)




# import getpass
# import os

# if not os.environ.get("GEMINI_API_KEY"):
#   os.environ["GEMINI_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")
# ans = os.environ.get("COHERE_API_KEY")
# print(ans)


# from langchain.chat_models import init_chat_model

# model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# model.invoke("What color is the sky?")



import os
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

# Getting key from .env
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is not set")

os.environ["GEMINI_API_KEY"] = api_key
print("GEMINI_API_KEY set:", os.environ.get("GEMINI_API_KEY"))

# Initialize model using Google Gemini via LangChain
model = init_chat_model("gemini-2.5-flash", model_provider="google_genai")

# Invoke the model
# response = model.invoke("What is the square root of 81")
# print(response.content)

heading = "tree"
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate
system_prompt = SystemMessagePromptTemplate.from_template("You are an AI assisstant called {name} that helps generate a funny dad joke.", input_variables = ["name"])

user_prompt = HumanMessagePromptTemplate.from_template(
    """You are tasked with giving a joke. The topic of the joke is:-  
        -------------   
        {topic}. 
        -------------
        Give only one joke.""", 
        input_variables = ["topic"])

# print(user_prompt.format(topic = "bicycle").content)

from langchain.prompts import ChatPromptTemplate

first_prompt = ChatPromptTemplate.from_messages([system_prompt, user_prompt])

# print(first_prompt.format(
#     name = "popoye",
#     topic = "bicycle"))


chain_one = (
    {   "name" : lambda x: x["name"],
        "topic": lambda x: x["topic"]}
    | first_prompt
    | model
    | {"Joke": lambda x: x.content}
)

joke_msg = chain_one.invoke({"topic": heading,
                             "name" : "popoye"})

print(joke_msg)