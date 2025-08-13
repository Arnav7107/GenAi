from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Optional
from pydantic import BaseModel

load_dotenv()
model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Review(BaseModel):
    key_themes : Annotated[list[str], "Write down all the key features discussed in the description in a list"]
    summary : Annotated[str, "Give a breif summary of the given description"]
    sentiment : Annotated[str, "Give the sentiment of the description either positive, negative or neutral"]
    pros : Annotated[Optional[list[str]], "Write down the pros from the description if any in a list"]
    cons : Annotated[Optional[list[str]], "Write down the cons from the description if any in a list"]

structured_model = model.with_structured_output(Review)

text = """The **AeroX Pro Max** is a sleek, lightweight smartphone with a sharp 6.5-inch AMOLED display, smooth 120Hz refresh rate, and a powerful triple-camera setup that captures crisp shots even in low light. Its battery easily lasts a full day of heavy use, and the fast charging is a lifesaver. While the performance is snappy for gaming and multitasking, the phone’s glossy back tends to attract fingerprints quickly. Overall, it’s a great choice if you want flagship-level performance without the sky-high price tag.
 """
result = structured_model.invoke(text)

print(result)