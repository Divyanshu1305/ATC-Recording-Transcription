import os
import tempfile
import uvicorn
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pathlib import Path
from transcribe_atc import AudioTranscriptionPipeline
import database

from fastapi.staticfiles import StaticFiles

# Initialize SQLite database
database.init_db()

app = FastAPI(
    title="Air Traffic Control ASR API",
    description="ASR Transcription API for ATC audio files using crispasr.exe and denoising pipeline",
    version="1.0.0"
)

# Initialize the transcription pipeline
# Defaulting to paths relative to workspace directory
pipeline = AudioTranscriptionPipeline(
    exe_path="bin/cuda/crispasr.exe",
    model_path="models/ggufs/speech-model.gguf",
    sample_rate=16000
)

# Output directory for intermediate files/runs
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Serve the output folder statically
app.mount("/output", StaticFiles(directory=OUTPUT_DIR), name="output")

@app.get("/")
def read_root():
    return {
        "message": "ATC ASR Transcription API is running.",
        "docs": "/docs",
        "supported_features": ["Bandpass Filter", "Spectral Noise Reduction", "Overlapping 30s Chunks Saving", "Fast transcription via crispasr.exe"]
    }

@app.post("/transcribe")
async def transcribe_audio(
    file: UploadFile = File(...),
    apply_noise_removal: bool = Form(True),
    noise_reduction: float = Form(0.75),
    language: str = Form("en"),
    save_intermediate: bool = Form(True),
    device: str = Form("cuda")
):
    """
    Upload an audio file to transcribe it with the ATC pipeline.
    
    Parameters:
    - **file**: The audio file (WAV, MP3, etc.)
    - **apply_noise_removal**: Whether to apply bandpass filter + spectral noise reduction
    - **noise_reduction**: Denoising coefficient (0.0 to 1.0)
    - **language**: Target language code (default 'en')
    - **save_intermediate**: If true, intermediate filtered/denoised overlapping 30s chunks are saved to the 'output' directory.
    - **device**: Compute device to use ('cuda' or 'cpu')
    """
    # Verify file is uploaded
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded.")
        
    # Create a temporary file to write the upload to
    suffix = Path(file.filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        temp_input_path = tmp.name
        
    try:
        # Read the file in chunks and write to temp file
        content = await file.read()
        with open(temp_input_path, "wb") as f:
            f.write(content)
            
        # Determine the binary path based on the selected device
        if device == "cpu":
            exe_path = "bin/cpu/crispasr.exe"
            use_gpu = False
        else:
            exe_path = "bin/cuda/crispasr.exe"
            use_gpu = True
            
        req_pipeline = AudioTranscriptionPipeline(
            exe_path=exe_path,
            model_path="models/ggufs/speech-model.gguf",
            sample_rate=16000,
            use_gpu=use_gpu
        )
        
        # Run transcription pipeline
        result = req_pipeline.process_audio(
            audio_path=temp_input_path,
            apply_vad=True,
            apply_noise_removal=apply_noise_removal,
            noise_reduction=noise_reduction,
            language=language,
            save_intermediate=save_intermediate,
            output_dir=OUTPUT_DIR
        )
        
        # Save transcription JSON to output folder matching standard CLI behavior
        import datetime
        timestamp = result.get("timestamp", datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
        original_name = Path(file.filename).stem
        json_output_path = os.path.join(OUTPUT_DIR, f"{original_name}_transcription_{timestamp}.json")
        req_pipeline.save_transcription(result, json_output_path)
        
        # Format the response data explicitly to ensure intermediate paths are returned
        response_data = {
            "text": result.get("text", ""),
            "segments": result.get("segments", []),
            "language": result.get("language", ""),
            "processing_time_seconds": result.get("processing_time_seconds", 0.0),
            "speech_duration_seconds": result.get("speech_duration_seconds", 0.0),
            "vad_enabled": result.get("vad_enabled", True),
            "noise_removal_enabled": result.get("noise_removal_enabled", True),
            "device": device,
            "timestamp": timestamp,
            "saved_transcription_file": json_output_path,
            "filtered_full_path": result.get("filtered_full_path"),
            "filtered_full_path_rel": result.get("filtered_full_path_rel"),
            "denoised_full_path": result.get("denoised_full_path"),
            "denoised_full_path_rel": result.get("denoised_full_path_rel"),
            "intermediate_chunks": result.get("intermediate_chunks", [])
        }
        
        # Save transcription to database
        try:
            inserted_id = database.save_transcription(
                filename=file.filename,
                text=response_data["text"],
                segments=response_data["segments"],
                language=response_data["language"],
                processing_time_seconds=response_data["processing_time_seconds"],
                speech_duration_seconds=response_data["speech_duration_seconds"],
                vad_enabled=response_data["vad_enabled"],
                noise_removal_enabled=response_data["noise_removal_enabled"],
                device=response_data["device"],
                timestamp=response_data["timestamp"],
                saved_transcription_file=response_data["saved_transcription_file"],
                filtered_full_path_rel=response_data["filtered_full_path_rel"],
                denoised_full_path_rel=response_data["denoised_full_path_rel"],
                intermediate_chunks=response_data["intermediate_chunks"]
            )
            response_data["id"] = inserted_id
        except Exception as db_err:
            print(f"Warning: Failed to save transcription to database: {db_err}")
            
        return response_data
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Transcription failed: {str(e)}")
        
    finally:
        # Clean up input temporary file
        if os.path.exists(temp_input_path):
            try:
                os.remove(temp_input_path)
            except Exception as e:
                print(f"Warning: Could not remove temp file {temp_input_path}: {e}")

@app.get("/transcriptions")
def get_transcriptions(page: int = 1, limit: int = 10):
    if page < 1:
        page = 1
    if limit < 1:
        limit = 10
    transcriptions, total = database.get_transcriptions(page, limit)
    import math
    pages = math.ceil(total / limit) if total > 0 else 1
    return {
        "transcriptions": transcriptions,
        "total": total,
        "page": page,
        "limit": limit,
        "pages": pages
    }

@app.get("/transcriptions/{id}")
def get_transcription(id: int):
    record = database.get_transcription_by_id(id)
    if not record:
        raise HTTPException(status_code=404, detail="Transcription not found")
    return record

if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)
