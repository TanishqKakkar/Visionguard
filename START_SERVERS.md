# How to Start Both Servers

## Quick Start

You need to run **two servers** simultaneously:

### 1. Start the API Server (Backend)

Open a terminal/command prompt and run:

```bash
# Make sure you're in the project root (D:\cctv)
python api.py
```

You should see:
```
🚀 Starting CCTV Detection API Server
========================================
Model path: models/vit_convlstm_best.pt
Device: cuda (or cpu)
Classes: ['Emergency', 'Robbery', 'Trespassing', 'Violence', 'Weaponized']
========================================
Model loaded successfully on cuda
 * Running on http://0.0.0.0:5000
```

**Keep this terminal open!** The API server must stay running.

### 2. Start the GUI Server (Frontend)

Open a **new** terminal/command prompt and run:

```bash
# Navigate to the GUI folder
cd gui

# Install dependencies (if not already done)
npm install --legacy-peer-deps

# Start the development server
npm run dev
```

You should see:
```
  ▲ Next.js 16.0.0
  - Local:        http://localhost:3000
```

**Keep this terminal open too!**

### 3. Open the Application

Open your browser and go to: **http://localhost:3000**

## Troubleshooting

### "Failed to fetch" Error

This means the API server isn't running or isn't accessible.

**Solution:**
1. Make sure the API server is running (step 1 above)
2. Check that it's running on port 5000
3. Try accessing http://localhost:5000/health in your browser - you should see `{"status":"healthy",...}`

### Port Already in Use

If port 5000 is already in use:
1. Find what's using it: `netstat -ano | findstr :5000` (Windows) or `lsof -i :5000` (Mac/Linux)
2. Change the port in `api.py` (line 364): `app.run(host='0.0.0.0', port=5001, debug=False)`
3. Update the frontend: Create `gui/.env.local` with `NEXT_PUBLIC_API_URL=http://localhost:5001`

### Model Not Found

If you see "Model file not found":
1. Check that `models/vit_convlstm_best.pt` exists
2. Or set the `MODEL_PATH` environment variable:
   ```bash
   set MODEL_PATH=path\to\your\model.pt
   python api.py
   ```

### CORS Errors

If you see CORS errors in the browser console:
- The API already has CORS enabled
- Make sure you're accessing the frontend from `http://localhost:3000` (not `127.0.0.1`)

## Testing the Connection

1. **Test API directly:**
   ```bash
   curl http://localhost:5000/health
   ```
   Should return: `{"status":"healthy","device":"cuda"}`

2. **Test from browser:**
   Open: http://localhost:5000/health
   Should show the JSON response

3. **Test file upload:**
   Use the GUI to upload a video file and click "Analyze Video"

## Windows Batch File (Optional)

You can create a `start_all.bat` file to start both servers:

```batch
@echo off
start "API Server" cmd /k "python api.py"
timeout /t 3
start "GUI Server" cmd /k "cd gui && npm run dev"
```

Then just double-click `start_all.bat` to start both servers.

