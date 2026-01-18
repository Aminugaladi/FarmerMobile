from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. Loda .env (ma'ajiyarmu na API)
load_dotenv()

app = FastAPI()

# 2. Saita CORS don App din ya samu damar kira daga ko'ina
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Saita API
API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = """Sunanka FarmerAI. Kai kwararren masanin noma ne (Agronomist). 
Idan aka turo maka hoton shuka ko bayani na rubutu, gano cuta, kwari, ko matsalar kasa. 
Bayyana matsalar cikin harshen Hausa mai sauƙi irin ta Najeriya, sannan ka ba da shawarar magani, 
taki, ko hanyar gyara. Kasance mai fara'a da taimako."""

class Query(BaseModel):
    image_data: Optional[str] = None
    text_query: Optional[str] = None  

@app.post("/analyze")
async def analyze_crop(query: Query):
    try:
        model = genai.GenerativeModel('gemini-flash-latest') # Don sabuntawa
        
        prompt_parts = [SYSTEM_PROMPT]
        
        # 1. Idan akwai hoto (handling base64)
        if query.image_data:
            # 
            pure_base64 = query.image_data.split(",")[-1] if "," in query.image_data else query.image_data
            prompt_parts.append({
                "mime_type": "image/jpeg",
                "data": pure_base64
            })
            
        # 2. Idan akwai rubutu
        if query.text_query:
            prompt_parts.append(f"\nTambayar manomi: {query.text_query}")
            
        if len(prompt_parts) == 1:
            return {"analysis": "Don Allah turo hoto ko ka rubuta tambaya."}

        # Kira Gemini
        response = model.generate_content(prompt_parts)
            
        return {"analysis": response.text}
        
    except Exception as e:
        print(f"Kuskure: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)