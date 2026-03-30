from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List
import uvicorn
import pandas as pd
from pathlib import Path
import io
import json
import base64
from fastapi.responses import Response

# Note: assuming pyrates is in PYTHONPATH
from pyrates.webapp.webapp.routes import predict
from pyrates.webapp.science_utils.smiles_utils import is_valid_smiles, canonicalize_smiles
from pyrates.utilities.plotting.visualize_spin_population import visualize_density_interactive, visualize_density
from pyrates.utilities.plotting.plot_modes import plot_mode_distribution_single
import matplotlib.pyplot as plt

app = FastAPI(title="PyRates REST API")

# Mount static files (like ketcher)
import os
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

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

class PlotModeRequest(BaseModel):
    mu: List[float]
    sigma: List[float]
    pi: List[float]
    predicted_value: float
    uncertainty: float
    target_name: str
    unit: str

@app.post("/api/plot/mode")
async def api_plot_mode(request: PlotModeRequest):
    try:
        fig = plot_mode_distribution_single(
            mu=request.mu,
            sigma=request.sigma,
            pi=request.pi,
            predicted_value=request.predicted_value,
            uncertainty=request.uncertainty,
            target_name=request.target_name,
            unit=request.unit
        )
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=150)
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.getvalue(), media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class PlotSpinRequest(BaseModel):
    smiles: str
    spin_population: List[float]
    uncertainty: List[float] = None
    show_uncertainty: bool = False

@app.post("/api/plot/spin")
async def api_plot_spin(request: PlotSpinRequest):
    try:
        fig = visualize_density_interactive(
            request.smiles, 
            request.spin_population, 
            request.uncertainty if request.show_uncertainty else None
        )
        return Response(content=fig.to_json(), media_type="application/json")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).parent / "templates" / "index.html"
    return html_path.read_text("utf8")

if __name__ == "__main__":
    uvicorn.run("pyrates.webapp.fastapi_app:app", host="127.0.0.1", port=8000, reload=True)
