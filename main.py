from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
import base64
from dotenv import load_dotenv

# 1. Loda .env domin amfani a gida (Local)
load_dotenv()

app = FastAPI()

# 2. Saita Gemini API tare da kariya
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

# FarmerAI System Prompt (Hausa Expert)
SYSTEM_PROMPT = """Sunanka FarmerAI. Kai kwararren masanin noma ne (Agronomist). 
Idan aka turo maka hoton shuka ko bayani, gano cuta, kwari, ko matsalar kasa. 
Bayyana matsalar cikin harshen Hausa mai sauƙi irin ta Najeriya, sannan ka ba da shawarar magani, 
taki, ko hanyar gyara. Kasance mai fara'a da taimako."""

class Query(BaseModel):
    image_data: str = None # Base64 hoto
    text: str = None       # Tambaya

@app.post("/analyze")
async def analyze_crop(query: Query):
    try:
        # MUN CANJA: An yi amfani da -latest don kaucewa 404 Error
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        
        if query.image_data:
            # Saita hoto yadda Gemini zai karanta shi
            image_parts = [
                {
                    "mime_type": "image/jpeg",
                    "data": query.image_data
                }
            ]
            response = model.generate_content([SYSTEM_PROMPT, image_parts[0]])
        else:
            # Idan tambayar rubutu ce kawai
            response = model.generate_content(f"{SYSTEM_PROMPT}\nTambaya: {query.text}")
            
        return {"analysis": response.text}
    except Exception as e:
        print(f"Babban Kuskure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Wannan layin yana taimaka wa Render ya gano Port din
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)