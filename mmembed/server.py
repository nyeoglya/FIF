import asyncio
import traceback
import typing as tp
import argparse
import multiprocessing

import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 기존 utils 임포트 유지
from utils import GenerationInput, QueryGenerationInput

@asynccontextmanager
async def lifespan(app: FastAPI):
    device_id = getattr(app.state, "device_id", "cuda:0")
    try:
        from model import MMEmbed
        print(f"[Startup] Loading model on {device_id}...")
        app.state.model = MMEmbed(device=device_id)
    except Exception as e:
        print(f"[Error] Model load failed on {device_id}: {e}")
        app.state.model = None

    print(f"[Startup] Model ready on {device_id}.")
    yield
    print(f"[Shutdown] Cleaning up {device_id}...")
    app.state.model = None

app = FastAPI(
    title="MM-Embed Inference Server",
    lifespan=lifespan
)
app.state.sem = asyncio.Semaphore(1)

class EmbeddingRequest(BaseModel):
    text: str = ""
    img_path: str = ""
    bounding_box: tp.Optional[tp.Tuple[int, int, int, int]] = None

class QueryEmbeddingRequest(BaseModel):
    instruction: str = ""
    text: str = ""
    img_path: str = ""

class EmbeddingResponse(BaseModel):
    embedding: tp.List[float]

@app.post("/embed", response_model=EmbeddingResponse)
async def embed_endpoint(request: EmbeddingRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    gen_input = GenerationInput(
        text=request.text, img_path=request.img_path, bounding_box=request.bounding_box,
    )

    try:
        async with app.state.sem:
            emb = await asyncio.to_thread(app.state.model.embedding, gen_input)
        return EmbeddingResponse(embedding=emb.tolist())
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/embed/query", response_model=EmbeddingResponse)
async def query_embed_endpoint(request: QueryEmbeddingRequest):
    if app.state.model is None:
        raise HTTPException(status_code=503, detail="Model not initialized")

    query_gen_input = QueryGenerationInput(
        instruction=request.instruction, text=request.text, img_path=request.img_path,
    )

    try:
        async with app.state.sem:
            emb = await asyncio.to_thread(app.state.model.query_embedding, query_gen_input)
        return EmbeddingResponse(embedding=emb.squeeze(0).tolist())
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

def run_instance(port: int, device_id: str):
    app.state.device_id = device_id
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", type=str, default="0", help="GPU Number (ex: 0 or 0,1,2)")
    parser.add_argument("--port", type=int, default=8000, help="Port number (ex: 8000 or 8000,8001,8002)")
    args = parser.parse_args()

    gpu_list = [f"cuda:{g.strip()}" for g in args.gpus.split(",")]

    if len(gpu_list) == 1:
        app.state.device_id = gpu_list[0]
        uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1)
    else:
        # 다중 GPU 실행 시 multiprocessing 사용
        processes = []
        for i, gpu in enumerate(gpu_list):
            p_port = args.port + i
            p = multiprocessing.Process(target=run_instance, args=(p_port, gpu))
            p.start()
            processes.append(p)
            print(f"[Manager] Spawned server on {gpu} at port {p_port}")

        for p in processes:
            p.join()
