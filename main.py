from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Loda bayanai daga .env
load_dotenv()

app = FastAPI()

# Saita Gemini API ta amfani da Environment Variable
# Wannan zai duba 'GEMINI_API_KEY' a cikin .env ko Render settings
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Kuskure: Ba a saita GEMINI_API_KEY ba!")
else:
    genai.configure(api_key=api_key)

# FarmerAI System Prompt (Muryar Namiji - Kai kwararre ne)
SYSTEM_PROMPT = """Sunanka FarmerAI. Kai kwararren masanin noma ne (Agronomist). 
Idan aka turo maka hoton shuka ko bayani, gano cuta, kwari, ko matsalar kasa. 
Bayyana matsalar cikin harshen Hausa mai sauƙi irin ta Najeriya, sannan ka ba da shawarar magani, 
taki, ko hanyar gyara. Kasance mai fara'a da taimako."""

class Query(BaseModel):
    image_data: str = None
    text: str = None

@app.post("/analyze")
async def analyze_crop(query: Query):
    if not api_key:
        raise HTTPException(status_code=500, detail="API Key not configured on server.")
        
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        if query.image_data:
            # Idan hoto ne aka turo
            response = model.generate_content([
                SYSTEM_PROMPT,
                {'mime_type': 'image/jpeg', 'data': query.image_data}
            ])
        else:
            # Idan rubutu ne kawai
            response = model.generate_content(f"{SYSTEM_PROMPT}\nTambaya: {query.text}")
            
        return {"analysis": response.text}
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Port din zai koma dynamic don Render
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)