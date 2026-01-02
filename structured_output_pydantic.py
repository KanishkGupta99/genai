from langchain_google_genai import ChatGoogleGenerativeAI
from dotenv import load_dotenv
from typing import TypedDict, Annotated
from pydantic import BaseModel

load_dotenv()

model=ChatGoogleGenerativeAI(model="gemini-2.5-flash")

class Review(BaseModel):
    summary:str
    sentiment:str

structured_model=model.with_structured_output(Review)
result=structured_model.invoke("""Phone Review: XYZ Pro

The **XYZ Pro** delivers a solid all-round experience at its price point. The **6.6-inch AMOLED display** is bright and vibrant, making videos and scrolling enjoyable. Performance is smooth for daily tasks and casual gaming, thanks to the efficient processor and ample RAM.

The **camera setup** performs well in good lighting, capturing sharp photos with accurate colors, though low-light shots could be better. Battery life is reliable, easily lasting a full day with moderate use, and **fast charging** is a welcome addition.

On the downside, the phone lacks a headphone jack and the camera app can feel slightly slow at times. Overall, the XYZ Pro is a **great value-for-money phone** for users looking for strong performance, a good display, and dependable battery life. a really really bad phone, worst phone.
""")

print(result)