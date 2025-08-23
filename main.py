import base64
import io
import numpy as np
import torch
import torch.nn as nn
import torchaudio.transforms as T
import soundfile as sf
import librosa
from fastapi import FastAPI
from pydantic import BaseModel
from model import AudioCNN

class NormalizeSpec(nn.Module):
    def forward(self, spec):
        mean = spec.mean()
        std = spec.std()
        return (spec - mean) / (std + 1e-6)


class AudioProcessor:
    def __init__(self, sample_rate=22050, mel_params=None):
        if mel_params is None:
            mel_params = {
                "n_fft": 2048,
                "hop_length": 512,
                "n_mels": 128,
                "f_min": 0,
                "f_max": sample_rate // 2,
            }
        self.sample_rate = sample_rate
        self.transform = nn.Sequential(
            T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=mel_params["n_fft"],
                hop_length=mel_params["hop_length"],
                n_mels=mel_params["n_mels"],
                f_min=mel_params["f_min"],
                f_max=mel_params["f_max"],
            ),
            T.AmplitudeToDB(),
            NormalizeSpec(),
        )

    def process_audio_chunk(self, audio_data):
        waveform = torch.from_numpy(audio_data).float().unsqueeze(0)
        spectrogram = self.transform(waveform)
        return spectrogram.unsqueeze(0)


class InferenceRequest(BaseModel):
    audio_data: str


class AudioClassifier:
    def __init__(self, model_path="./saved_models/best_model.pth"):
        print("Loading model...")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        checkpoint = torch.load(model_path, map_location=self.device)
        self.classes = checkpoint["classes"]

        sr = checkpoint.get("sample_rate", 22050)
        mel_params = checkpoint.get("mel_params", None)

        self.model = AudioCNN(num_classes=len(self.classes))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        self.audio_processor = AudioProcessor(sample_rate=sr, mel_params=mel_params)
        print(f"Model loaded with sample_rate={sr}, mel_params={mel_params}.")

    def predict(self, audio_b64: str):
        # Decode base64
        audio_bytes = base64.b64decode(audio_b64)
        audio_data, sample_rate = sf.read(io.BytesIO(audio_bytes), dtype="float32")

        # Stereo → mono
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=1)

        # Resample if input sample rate doesn’t match training
        if sample_rate != self.audio_processor.sample_rate:
            print(
                f"[Resample] Input audio sr={sample_rate}, expected sr={self.audio_processor.sample_rate}. Resampling..."
            )
            audio_data = librosa.resample(
                audio_data,
                orig_sr=sample_rate,
                target_sr=self.audio_processor.sample_rate,
            )

        # Spectrogram
        spectrogram = self.audio_processor.process_audio_chunk(audio_data).to(
            self.device
        )

        # Spectrogram sanity check
        print(
            "spec shape", spectrogram.shape,
            "mean", spectrogram.mean().item(),
            "std", spectrogram.std().item(),
            "min", spectrogram.min().item(),
            "max", spectrogram.max().item(),
        )

        with torch.no_grad():
            output, feature_maps = self.model(spectrogram, return_feature_maps=True)
            output = torch.nan_to_num(output)
            probabilities = torch.softmax(output, dim=1)
            top3_probs, top3_indices = torch.topk(probabilities[0], 3)

            predictions = [
                {"class": self.classes[idx.item()], "confidence": prob.item()}
                for prob, idx in zip(top3_probs, top3_indices)
            ]

        # Feature maps for visualization
        viz_data = {}
        for name, tensor in feature_maps.items():
            if tensor.dim() == 4:  # [batch, channels, height, width]
                aggregated_tensor = torch.mean(tensor, dim=1)
                numpy_array = aggregated_tensor.squeeze(0).cpu().numpy()
                viz_data[name] = {
                    "shape": list(numpy_array.shape),
                    "values": np.nan_to_num(numpy_array).tolist(),
                }

        # Input spectrogram
        spectrogram_np = spectrogram.squeeze(0).squeeze(0).cpu().numpy()
        clean_spectrogram = np.nan_to_num(spectrogram_np)

        # Downsample waveform for visualization only
        max_samples = 8000
        waveform_sample_rate = self.audio_processor.sample_rate
        if len(audio_data) > max_samples:
            step = len(audio_data) // max_samples
            waveform_data = audio_data[::step]
        else:
            waveform_data = audio_data

        return {
            "predictions": predictions,
            "visualization": viz_data,
            "input_spectrogram": {
                "shape": list(clean_spectrogram.shape),
                "values": clean_spectrogram.tolist(),
            },
            "waveform": {
                "values": waveform_data.tolist(),
                "sample_rate": waveform_sample_rate,
                "duration": len(audio_data) / waveform_sample_rate,
            },
        }


app = FastAPI()
classifier = AudioClassifier()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/inference")
def inference(request: InferenceRequest):
    return classifier.predict(request.audio_data)


if __name__ == "__main__":
    import uvicorn
    print("Starting server at http://127.0.0.1:8000")
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
