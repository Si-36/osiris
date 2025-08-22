"""Minimal Working API - 30 lines total"""
from fastapi import FastAPI
from core.src.aura_intelligence.minimal_production import get_minimal_aura

app = FastAPI(title="AURA Minimal")
aura = get_minimal_aura()

@app.get("/")
def root():
    return {"status": "working", "components": len(aura.registry.components)}

@app.post("/process")
async def process(data: dict):
    return await aura.process(data)

@app.get("/health")
def health():
    stats = aura.registry.get_component_stats()
    return {"health": "ok", "stats": stats}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8099)