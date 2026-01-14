# NeuroSonic - Audio Classification with PyTorch & ResNet

Audio classification system for environmental sounds (dog barks, bird chirps) using PyTorch and ResNet-inspired CNN. Converts raw audio to Mel spectrograms with REST API for real-time predictions.

## Features

- **Dataset**: ESC-50 environmental sounds (2000 audio files)
- **Architecture**: ResNet-inspired CNN with Mel spectrogram preprocessing
- **API**: RESTful backend for real-time inference
- **GPU Support**: NVIDIA (Cuda)
- **Monitoring**: TensorBoard integration

## Quick Start

### Setup

```bash
git clone https://github.com/your-username/neurosonic.git
cd neurosonic

# Train the model
python train.py

# Start the backend
uvicorn main:app --reload --port 8000

# Start the fronend
cd frontend
npm run dev
```

### Monitor Training

```bash
tensorboard --logdir ./tensorboard_logs
# View at http://localhost:6006
```

## API Usage

**Base URL:** `http://127.0.0.1:8000`

**Health Check:**

```bash
curl -X GET http://127.0.0.1:8000/health
```

**Prediction:**

```bash
curl -X POST http://127.0.0.1:8000/inference \
  -H "Content-Type: application/json" \
  -d '{"audio_data": "base64_encoded_audio_content"}'
```

**Response:**

```json
{
  "predictions": [
    { "class": "chirping_birds", "confidence": 0.4069520831108093 },
    { "class": "rain", "confidence": 0.0407380647957325 },
    { "class": "water_drops", "confidence": 0.04057849571108818 }
  ],
  "visualization": {
    "conv1": {
      "shape": [32, 207],          // feature map shape (channels x width)
      "values": [
        [0.0012, -0.0034, 0.0005],
        [0.0009, -0.0021, 0.0007]
      ]                            // shortened values (real output has 32x207 floats)
    },
    "layer1": {
      "shape": [32, 207],
      "values": [
        [0.0021, -0.0015, 0.0008],
        [0.0010, -0.0023, 0.0006]
      ]
    },
    "layer2": {
      "shape": [16, 104],
      "values": [
        [0.0015, -0.0027, 0.0009],
        [0.0012, -0.0018, 0.0004]
      ]
    },
    "layer3": {
      "shape": [8, 52],
      "values": [
        [0.0007, -0.0013, 0.0003],
        [0.0005, -0.0009, 0.0002]
      ]
    },
    "layer4": {
      "shape": [4, 26],
      "values": [
        [0.0003, -0.0006, 0.0001],
        [0.0002, -0.0004, 0.0001]
      ]
    }
  },
  "input_spectrogram": {
    "shape": [128, 825],            // mel bins x time frames
    "values": [
      [-3.12, -3.05, -2.98],
      [-3.10, -3.02, -2.95]
    ]                              // shortened for readability
  },
  "waveform": {
    "values": [0.0001, 0.0003, -0.0002, 0.0000],  // downsampled waveform
    "sample_rate": 22050,                         // Hz
    "duration": 19.15201814058957                 // seconds
  }
}

```

## Project Structure

```
neurosonic/
├── train.py               # Training script
├── requirements.txt        # Python dependencies
├── main.py                # API service
├── model.py               # CNN architecture
├── saved_models/          # Model checkpoints
└── tensorboard_logs/      # Training logs
```

## Future Plans
- Model optimization for edge deployment
- Testing/Optimization the model with other dataset

**Tech Stack:** PyTorch | ResNet CNN | FastAPI | TensorBoard
