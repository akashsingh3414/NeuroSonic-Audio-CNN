from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager
import logging
from utils.classifier import AudioClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

classifier: AudioClassifier | None = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global classifier
    try:
        logger.info("Starting NeuroSonic Audio Classifier API")

        classifier = AudioClassifier()
        logger.info(
            f"Model loaded | device={classifier.device} | classes={len(classifier.classes)}"
        )
        yield
    except Exception:
        logger.exception("Failed to initialize AudioClassifier")
        classifier = None
        yield
    finally:
        logger.info("Shutting down NeuroSonic Audio Classifier API")


app = FastAPI(
    title="NeuroSonic Audio Classifier API",
    description="ESC-50 audio classification with CNN",
    version="2.0.1",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class InferenceRequest(BaseModel):
    audio_data: str # Base64 encoded audio file

class InferenceResponse(BaseModel):
    predictions: list
    visualization: dict
    input_spectrogram: dict
    waveform: dict
    metadata: dict

@app.get("/")
def root():
    return {
        "name": "NeuroSonic Audio Classifier",
        "status": "running" if classifier else "model not loaded",
        "endpoints": {
            "health": "/health",
            "inference": "/inference",
            "model_info": "/model-info",
        },
    }


@app.get("/health")
def health():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(classifier.device),
        "num_classes": len(classifier.classes),
    }


@app.get("/model-info")
def model_info():
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return {
        "classes": classifier.classes,
        "metadata": classifier.model_metadata,
        "device": str(classifier.device),
        "sample_rate": classifier.audio_processor.sample_rate,
        "mel_params": classifier.audio_processor.mel_params,
    }


@app.post("/inference", response_model=InferenceResponse)
def inference(request: InferenceRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if not request.audio_data:
        raise HTTPException(status_code=400, detail="No audio data provided")

    try:
        return classifier.predict(request.audio_data)
    except Exception:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail="Inference failed")
