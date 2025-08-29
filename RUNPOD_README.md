# RunPod Serverless Handler for WhisperX FastAPI

This handler is fully compatible with RunPod serverless architecture. It allows you to call your existing WhisperX endpoints from RunPod serverless using `run()` and `runsync()` commands.

## ✅ RunPod Compatibility

- **Proper RunPod handler signature**: Uses `handler(job)` as required by RunPod
- **RunPod SDK integration**: Uses `runpod.serverless.start()` to keep container alive
- **Correct Dockerfile**: Runs `handler.py` instead of just importing and exiting
- **Local testing**: Includes test script to verify handler before deployment

## Usage

### Basic Structure
```python
# From RunPod client
result = pod.run({
    "input": {
        "action": "transcribe",
        "file": "base64_encoded_audio_or_url",
        "filename": "audio.mp3",
        "model_params": {"language": "en"},
        "asr_options": {},
        "vad_options": {}
    }
})
```

### Available Actions

#### 1. **speech_to_text** - Full pipeline (transcribe + align + diarize + combine)
```python
{
    "action": "speech_to_text",
    "file": "base64_audio_data_or_url",
    "filename": "audio.mp3",
    "model_params": {"language": "en", "model": "tiny"},
    "align_params": {"min_speakers": 1, "max_speakers": 2},
    "diarize_params": {"min_speakers": 1, "max_speakers": 2},
    "asr_options": {"beam_size": 5},
    "vad_options": {"min_speech_duration_ms": 250}
}
```

#### 2. **speech_to_text_url** - Full pipeline from URL
```python
{
    "action": "speech_to_text_url",
    "file": "https://example.com/audio.mp3",
    "model_params": {"language": "en"},
    "align_params": {},
    "diarize_params": {},
    "asr_options": {},
    "vad_options": {}
}
```

#### 3. **transcribe** - Transcription only (individual service)
```python
{
    "action": "transcribe",
    "file": "base64_audio_data_or_url",
    "filename": "audio.mp3",
    "model_params": {"language": "en", "model": "tiny"},
    "asr_options": {"beam_size": 5},
    "vad_options": {"min_speech_duration_ms": 250}
}
```

#### 4. **align** - Alignment only (individual service)
```python
{
    "action": "align",
    "file": "base64_audio_data_or_url",
    "filename": "audio.mp3",
    "transcript": "{\"language\": \"en\", \"segments\": [...]}",
    "device": "cpu",
    "align_params": {"min_speakers": 1, "max_speakers": 2}
}
```

#### 5. **diarize** - Diarization only (individual service)
```python
{
    "action": "diarize",
    "file": "base64_audio_data_or_url",
    "filename": "audio.mp3",
    "device": "cpu",
    "diarize_params": {"min_speakers": 1, "max_speakers": 2}
}
```

#### 6. **combine** - Combine transcript and diarization only (individual service)
```python
{
    "action": "combine",
    "aligned_transcript": "{\"segments\": [...], \"word_segments\": [...]}",
    "diarization_result": "[{\"start\": 0.0, \"end\": 10.0, \"speaker\": \"SPEAKER_00\"}, ...]"
}
```

#### 7. **get_task** - Get task status
```python
{
    "action": "get_task",
    "identifier": "task_id_here"
}
```

#### 8. **get_all_tasks** - Get all tasks
```python
{
    "action": "get_all_tasks"
}
```

#### 9. **delete_task** - Delete task
```python
{
    "action": "delete_task",
    "identifier": "task_id_here"
}
```

#### 10. **health** - Health check
```python
{
    "action": "health"
}
```

## Key Differences

### **Full Pipeline Endpoints:**
- **`speech_to_text`** - Does transcribe + align + diarize + combine in one call
- **`speech_to_text_url`** - Same as above but from URL
- Uses `process_audio_common()` function
- Task type: `TaskType.full_process`

### **Individual Service Endpoints:**
- **`transcribe`** - Only transcription using `process_transcribe()`
- **`align`** - Only alignment using `process_alignment()`
- **`diarize`** - Only diarization using `process_diarize()`
- **`combine`** - Only combination using `process_speaker_assignment()`
- Task types: `TaskType.transcription`, `TaskType.transcription_alignment`, `TaskType.diarization`, `TaskType.combine_transcript_diarization`

## Important Notes

### **For `align` endpoint:**
- `transcript` should be JSON data matching the `Transcript` schema
- Example: `{"language": "en", "segments": [{"start": 0.0, "end": 10.0, "text": "Hello world"}]}`

### **For `combine` endpoint:**
- `aligned_transcript` should be JSON data matching the `AlignedTranscription` schema
- `diarization_result` should be a list of diarization segments
- Both are processed exactly as your FastAPI endpoints expect

### **Device parameter:**
- Available for `align` and `diarize` endpoints
- Defaults to your configured device (from `app.whisperx_services.device`)
- Can be overridden with `"device": "cpu"` or `"device": "cuda"`

## File Input Formats

### Base64 Data
```python
"file": "data:audio/mp3;base64,UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT..."
```

### URL
```python
"file": "https://example.com/audio.mp3"
```

### Plain Base64
```python
"file": "UklGRnoGAABXQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YQoGAACBhYqFbF1fdJivrJBhNjVgodDbq2EcBj+a2/LDciUFLIHO8tiJNwgZaLvt559NEAxQp+PwtmMcBjiR1/LMeSwFJHfH8N2QQAoUXrTp66hVFApGn+DyvmwhBSuBzvLZiTYIG2m98OScTgwOUarm7blmGgU7k9n1unEiBC13yO/eizEIHWq+8+OWT..."
```

## Response Format

### Success
```python
{
    "identifier": "task_id",
    "result": "processing_result",
    "message": "Operation completed successfully"
}
```

### Error
```python
{
    "error": "Error message here"
}
```

## Notes

- **Full pipeline endpoints** (`speech_to_text`, `speech_to_text_url`) do everything in one call
- **Individual service endpoints** (`transcribe`, `align`, `diarize`, `combine`) do one operation at a time
- **Exact function signatures** - Each handler calls the exact same functions with the exact same parameters as your FastAPI endpoints
- **Database integration** - All endpoints create proper database entries with correct task types
- **File validation** - Uses your existing validation functions
- **Error handling** - Matches your endpoint error handling patterns
- The handler automatically handles temporary file cleanup
- All existing WhisperX functionality is preserved
- No changes to your existing codebase required

## 🚀 Deployment to RunPod

### 1. Test Locally First
```bash
# Test your handler before deployment
python test_handler_local.py
```

### 2. Deploy to RunPod
1. **Use `dockerfile.runpod`** (not the original Dockerfile)
2. **Set Dockerfile path** to `dockerfile.runpod` in RunPod console
3. **Build context** should be `.` (root directory)
4. **Deploy** and wait for container to build

### 3. Test on RunPod
```bash
# Test health endpoint
curl -X POST "https://api.runpod.ai/v2/YOUR_ENDPOINT_ID/runsync" \
  -H "Content-Type: application/json" \
  -d '{"input": {"action": "health"}}'
```

## 🔧 Key Changes Made

1. **Handler signature**: Changed from `handler(event)` to `handler(job)` (RunPod requirement)
2. **RunPod SDK**: Added `runpod.serverless.start()` to keep container alive
3. **Dockerfile**: Updated to run `handler.py` instead of just importing
4. **Requirements**: Added `runpod==1.7.9` package
5. **Database initialization**: Added automatic table creation when container starts
6. **Local testing**: Added `test_handler_local.py` for pre-deployment testing


Para correr estou a correr o comando  source venv/bin/activate && python test_runpod.py 

depois POST http://127.0.0.1:8001/runsync com
{
  "input": {
    "action": "get_task",
    "identifier":"2d63e42e-6acf-4e31-b05b-96cb6e4742ac"
  }
}
{
  "input": {
    "action": "get_all_tasks"
  }
}
{
  "input": {
    "action": "speech_to_text_url",
    "url": "https://assets.noop.pt/output5m.mp3",
    "model_params": {
      "language": "pt",
      "task": "transcribe",
      "model": "tiny",
      "device": "cpu",
      "device_index": 0,
      "threads": 0,
      "batch_size": 8,
      "chunk_size": 20,
      "compute_type": "int8",
      "interpolate_method": "nearest",
      "return_char_alignments": false,
      "beam_size": 5,
      "best_of": 5,
      "patience": 1,
      "length_penalty": 1,
      "temperatures": 0,
      "compression_ratio_threshold": 2.4,
      "log_prob_threshold": -1,
      "no_speech_threshold": 0.6,
      "suppress_tokens": -1,
      "suppress_numerals": false
    },
    "vad_options": {
      "vad_onset": 0.5,
      "vad_offset": 0.363
    }
  }
}