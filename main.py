from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
import base64
from dotenv import load_dotenv

# 1. Loda .env don tsaro
load_dotenv()

app = FastAPI()

# 2. Dauko API Key daga Environment Variable maimakon rubutawa a fili
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
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if query.image_data:
            # 3. Gyara: Tabbatar da cewa Gemini ya karbi hoto a matsayin bytes
            # Wannan bangaren yana gyara yadda Gemini yake karantar hoton
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": query.image_data # Tunda mun riga mun turo shi a matsayin base64 daga App
                }
            ]
            
            response = model.generate_content([SYSTEM_PROMPT, image_parts[0]])
        else:
            # Idan rubutu ne kawai
            response = model.generate_content(f"{SYSTEM_PROMPT}\nTambaya: {query.text}")
            
        return {"analysis": response.text}
    except Exception as e:
        print(f"Error: {e}") # Wannan zai nuna mana takamaiman matsalar a Render logs
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Render yana amfani da $PORT variable, amma don local testing muna amfani da 8000
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)