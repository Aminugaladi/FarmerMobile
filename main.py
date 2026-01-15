from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
import base64
from dotenv import load_dotenv

# 1. Loda .env
load_dotenv()

app = FastAPI()

# 2. Saita API
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# FarmerAI System Prompt
SYSTEM_PROMPT = """Sunanka FarmerAI. Kai kwararren masanin noma ne (Agronomist). 
Idan aka turo maka hoton shuka ko bayani, gano cuta, kwari, ko matsalar kasa. 
Bayyana matsalar cikin harshen Hausa mai sauƙi irin ta Najeriya, sannan ka ba da shawarar magani, 
taki, ko hanyar gyara. Kasance mai fara'a da taimako."""

class Query(BaseModel):
    image_data: str = None
    text: str = None

@app.post("/analyze")
async def analyze_crop(query: Query):
    try:
        # Model settings
        model = genai.GenerativeModel('gemini-flash-latest')
        
        if query.image_data:
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": query.image_data 
                }
            ]
            response = model.generate_content([SYSTEM_PROMPT, image_parts[0]])
        else:
            response = model.generate_content(f"{SYSTEM_PROMPT}\nTambaya: {query.text}")
            
        return {"analysis": response.text}
    except Exception as e:
        print(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)