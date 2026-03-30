from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
from pathlib import Path

# Note: assuming pyrates is in PYTHONPATH
from pyrates.webapp.webapp.routes import predict
from pyrates.webapp.science_utils.smiles_utils import is_valid_smiles, canonicalize_smiles

app = FastAPI(title="PyRates REST API")

class PredictRequest(BaseModel):
    smiles: List[str]

@app.post("/api/predict")
async def api_predict(request: PredictRequest):
    valid = []
    invalid = []
    for s in request.smiles:
        is_val = False
        try:
            is_val = is_valid_smiles(s)
        except Exception:
            pass
        if not is_val:
            invalid.append(s)
            continue
        canon = canonicalize_smiles(s)
        valid.append(canon if canon else s.strip())

    if not valid:
        raise HTTPException(status_code=400, detail="No valid SMILES provided")

    # Run predictions
    try:
        df = predict(valid)
        results = df.to_dict(orient="records")
        return {"results": results, "invalid": invalid}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text("utf8")

if __name__ == "__main__":
    uvicorn.run("pyrates.webapp.fastapi_app:app", host="127.0.0.1", port=8000, reload=True)

