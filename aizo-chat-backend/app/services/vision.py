# app/services/vision.py
import base64
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from app.config import VISION_MODEL
from app.core.prompts import IMAGE_ANALYSIS_PROMPT


async def describe_image(file_path: str) -> str:
    """
    Uses GPT-4o-mini to analyze an image and describe it textually.
    """
    print(f"   [DEBUG] Vision processing started for: {file_path}")
    
    try:
        with open(file_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
        
        vision_llm = ChatOpenAI(model=VISION_MODEL, max_tokens=1000)
        
        msg = HumanMessage(
            content=[
                {"type": "text", "text": IMAGE_ANALYSIS_PROMPT},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded_string}"}}
            ]
        )
        
        print("   [DEBUG] Sending image to OpenAI...")
        res = await vision_llm.ainvoke([msg])
        print("   [DEBUG] Vision response received.")
        return res.content

    except Exception as e:
        print(f"❌ [VISION ERROR] Could not analyze image: {e}")
        return "Error analyzing image. (Check backend logs for details)"