# CCTV Detection API Setup Guide

This guide explains how to connect your PyTorch model with the GUI application.

## Architecture

- **Backend API**: Flask server (`api.py`) that loads the PyTorch model and processes video uploads
- **Frontend GUI**: Next.js application that sends videos to the API and displays predictions

## Setup Instructions

### 1. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Model Path

The API looks for the model at `models/vit_convlstm_best.pt` by default. You can change this by:

- Setting the `MODEL_PATH` environment variable:
  ```bash
  export MODEL_PATH=path/to/your/model.pt
  ```

- Or editing the `MODEL_PATH` variable in `api.py` (line 60)

### 3. Start the API Server

```bash
python api.py
```

The server will start on `http://localhost:5000` by default.

### 4. Configure Frontend API URL (Optional)

The frontend defaults to `http://localhost:5000`. If your API runs on a different URL, create a `.env.local` file in the `gui` folder:

```env
NEXT_PUBLIC_API_URL=http://localhost:5000
```

### 5. Start the GUI Application

```bash
cd gui
pnpm install  # or npm install
pnpm dev      # or npm run dev
```

The GUI will be available at `http://localhost:3000` (or the port Next.js assigns).

## API Endpoints

### `GET /health`
Health check endpoint. Returns server status and device information.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda" or "cpu"
}
```

### `POST /predict`
Upload a video file and get predictions.

**Request:**
- Method: `POST`
- Content-Type: `multipart/form-data`
- Body: Form data with `video` field containing the video file

**Response:**
```json
{
  "predicted_class": "Violence",
  "predicted_index": 3,
  "confidence": 0.95,
  "all_probabilities": {
    "Emergency": 0.02,
    "Robbery": 0.01,
    "Trespassing": 0.01,
    "Violence": 0.95,
    "Weaponized": 0.01
  }
}
```

### `GET /classes`
Get list of all class names.

**Response:**
```json
{
  "classes": ["Emergency", "Robbery", "Trespassing", "Violence", "Weaponized"]
}
```

## Model Information

- **Model Architecture**: ViTConvLSTMImproved (Vision Transformer + LSTM)
- **Input**: 16 frames of 224x224 RGB images
- **Output**: 5 classes (Emergency, Robbery, Trespassing, Violence, Weaponized)
- **Preprocessing**: CLAHE enhancement, frame sampling, normalization

## Troubleshooting

### Model Not Found
- Ensure the model file exists at the specified path
- Check file permissions
- Verify the model architecture matches the code

### CORS Errors
- The API has CORS enabled by default
- If issues persist, check firewall settings

### CUDA Out of Memory
- The model will automatically fall back to CPU if CUDA is unavailable
- For large videos, consider processing in chunks

### Video Format Issues
- Supported formats: MP4, AVI, MOV, etc. (any format OpenCV can read)
- Ensure videos have at least a few frames

## Testing

Test the API directly with curl:

```bash
curl -X POST http://localhost:5000/predict \
  -F "video=@path/to/your/video.mp4"
```

## Production Deployment

For production:
1. Use a production WSGI server (e.g., Gunicorn)
2. Set up proper error handling and logging
3. Configure reverse proxy (nginx)
4. Use environment variables for configuration
5. Implement rate limiting and authentication

Example with Gunicorn:
```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 api:app
```

