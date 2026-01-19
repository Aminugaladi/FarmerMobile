from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

#  Loda .env
load_dotenv()

app = FastAPI()

# Saita CORS Don magana da Backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. Saita Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GEMINI_API_KEY ba a saita shi ba a cikin .env!")

genai.configure(api_key=API_KEY)

SYSTEM_PROMPT = """Sunanka FarmerAI. Kai kwararren masanin noma ne (Agronomist). 
Aikin ka shi ne taimaka wa manoma gano cututtukan shuka, kwari, ko matsalolin ƙasa daga hotuna ko rubutu.

TSARIN AMSA:
1. Yi amfani da '#' kafin manyan jigogi (misali: # Gano Cuta).
2. Yi amfani da '**' don nuna kalmomi masu muhimmanci.
3. Bayyana matsalar da hanyar magance ta cikin harshen Hausa mai sauƙi na Najeriya.
4. Kada ka cika yawan alamomi (symbols) da yawa don sifikar waya ta iya karantawa sarai.
5. Idan hoton bai nuna shuka ba, gaya wa manomin ya sake ɗaukar hoton shuka."""

class Query(BaseModel):
    image_data: Optional[str] = None
    text_query: Optional[str] = None  

@app.post("/analyze")
async def analyze_crop(query: Query):
    try:
        # Mun yi amfani da gemini-1.5-flash-latest ko gemini-flash-latest
        model = genai.GenerativeModel(
            model_name='gemini-flash-latest',
            generation_config={
                "temperature": 0.7,
                "top_p": 0.95,
                "max_output_tokens": 1024,
            }
        )
        
        prompt_parts = [SYSTEM_PROMPT]
        
        # Sarrafa Bayanan Hoto (Image Handling)
        if query.image_data and len(query.image_data.strip()) > 10:
            try:
                # Tace Base64: Cire prefix na "data:image/jpeg;base64," idan akwai
                if "," in query.image_data:
                    pure_base64 = query.image_data.split(",")[1]
                else:
                    pure_base64 = query.image_data
                
                prompt_parts.append({
                    "mime_type": "image/jpeg",
                    "data": pure_base64
                })
            except Exception as img_err:
                print(f"Hoto Error: {img_err}")
                # Kada mu dakatar da komai idan hoto ya samu matsala, rubutu na iya aiki
            
        #  Sarrafa Bayanan Rubutu (Text Handling)
        if query.text_query and query.text_query.strip():
            prompt_parts.append(f"\nTambayar manomi: {query.text_query}")
            
        # Idan payload din babu hoto kuma babu rubutu
        if len(prompt_parts) == 1:
            return {"analysis": "# Sako Daga FarmerAI\n\nBarka! Don Allah turo hoto ko ka rubuta tambayar ka don in taimake ka."}

        #  Kiran Gemini API
        response = model.generate_content(prompt_parts)
        
        if not response.text:
            return {"analysis": "# Yi Haƙuri\n\nBan iya gano komai ba a wannan karon. Gwada gyara hoton ko ƙara bayani a rubuce."}
            
        return {"analysis": response.text}
        
    except Exception as e:
        print(f"Babban Kuskure (Backend): {str(e)}")
        # Ba da bayani mai dadi maimakon danyen kuskure
        raise HTTPException(status_code=500, detail="An samu matsala wurin tuntubar AI. Duba Intanet dinka ko API Key.")

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "FarmerAI Backend is Running Successfully!",
        "model": "gemini-flash-latest"
    }

if __name__ == "__main__":
    import uvicorn
    # Port din zai canza zuwa wanda Server din da aka tura (kamar Render ko Heroku) ta bayar
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)