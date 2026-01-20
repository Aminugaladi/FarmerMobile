from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional
from pydantic import BaseModel
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Loda .env
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

# Saita Gemini API
API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    print("WARNING: GEMINI_API_KEY ba a saita shi ba a cikin .env!")

genai.configure(api_key=API_KEY)

# SYSTEM PROMPT:
SYSTEM_PROMPT = """Sunanka FarmerAI. Kai kwararren masanin noma ne (Agronomist). 
Aikin ka shi ne taimaka wa manoma gano cututtukan shuka, kwari, ko matsalolin ƙasa.

MUHIMMI: Idan manomi yayi tambaya game da cutar shuka (ko da babu hoto):
1. Kada ka tsaya a gaisuwa kawai. Ka lissafa cututtuka 2 ko 3 da aka saba samu ga waccan shukar.
2. Ka ba da alamomin kowace cuta daki-daki.
3. Ka ba da hanyar magance su (fadi sunayen sinadarai ko maganin da za'a fesa).
4. Ka gaya masa ya turo hoto domin ka tabbatar da takamaiman cutar idan bai turo ba.

TSARIN AMSA:
- Yi amfani da harshen Hausa mai sauƙi ta Najeriya.
- Yi amfani da '#' don manyan jigogi (headers).
- Kada ka cika yawan alamomi (symbols) da yawa domin sifikar waya ta karanta da kyau.
- Idan hoton da aka turo bai nuna shuka ba, bayyana hakan cikin ladabi.
- Kasance mai bayar da mafita kai tsaye (Proactive)."""

class Query(BaseModel):
    image_data: Optional[str] = None
    text_query: Optional[str] = None  

@app.post("/analyze")
async def analyze_crop(query: Query):
    try:
        #  Saita Safety Settings don AI ya yarda ya fadi magungunan feshi (Pesticides)
        safety_settings = [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
        ]

        model = genai.GenerativeModel(
            model_name='gemini-flash-latest',
            generation_config={
                "temperature": 0.8,
                "top_p": 0.95,
                "max_output_tokens": 2048,
            },
            safety_settings=safety_settings
        )
        
        prompt_parts = [SYSTEM_PROMPT]
        
        # Sarrafa Bayanan Hoto (Image Handling)
        if query.image_data and len(query.image_data.strip()) > 10:
            try:
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
            
        # Sarrafa Bayanan Rubutu (Text Handling)
        if query.text_query and query.text_query.strip():
            prompt_parts.append(f"\nTambayar manomi: {query.text_query}")
            
        # Idan babu komai a request din
        if len(prompt_parts) == 1:
            return {"analysis": "# Sako Daga FarmerAI\n\nBarka! Don Allah turo hoto ko ka rubuta tambayar ka don in taimake ka."}

        # 3. Kiran Gemini API
        response = model.generate_content(prompt_parts)
        
        if not response.text:
            return {"analysis": "# Yi Haƙuri\n\nBan iya gano komai ba a wannan karon. Gwada sake turo tambayar."}
            
        return {"analysis": response.text}
        
    except Exception as e:
        print(f"Babban Kuskure: {str(e)}")
        raise HTTPException(status_code=500, detail="An samu matsala wurin tuntubar AI.")

@app.get("/")
def home():
    return {
        "status": "online",
        "message": "FarmerAI Backend is Running Successfully!",
        "model": "gemini-1.5-flash-latest"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)