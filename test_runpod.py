#!/usr/bin/env python3
"""
RunPod Test Server - Working version
This allows you to test your handler exactly as it would be called from RunPod.
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import asyncio
import time

# Import handler
from handler import handler

app = FastAPI(title="RunPod Handler Test Server", description="Simulates RunPod serverless environment")

class RunPodRequest(BaseModel):
    input: dict

# Simulate RunPod's async job queue
job_queue = {}
job_counter = 0

@app.post("/run")
async def run_endpoint(request: RunPodRequest):
    """
    Simulates RunPod's run() method - asynchronous execution
    Returns immediately with a job ID
    """
    global job_counter
    job_id = f"test_job_{job_counter}"
    job_counter += 1
    
    # Add job to queue
    job_queue[job_id] = {
        "status": "IN_PROGRESS",
        "input": request.input,
        "start_time": time.time(),
        "output": None,
        "error": None
    }
    
    # Process job asynchronously
    asyncio.create_task(process_job_async(job_id))
    
    return {
        "id": job_id,
        "status": "IN_PROGRESS"
    }

@app.post("/runsync")
async def runsync_endpoint(request: RunPodRequest):
    """
    Simulates RunPod's runsync() method - synchronous execution
    Waits for completion and returns result
    """
    start_time = time.time()
    
    try:
        result = handler({"input": request.input})
        execution_time = time.time() - start_time
        
        return {
            "id": "sync_job",
            "status": "COMPLETED",
            "output": result,
            "execution_time": execution_time
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        return {
            "id": "sync_job",
            "status": "FAILED",
            "error": str(e),
            "execution_time": execution_time
        }

@app.get("/status/{job_id}")
async def get_job_status(job_id: str):
    """Get the status of an async job"""
    if job_id not in job_queue:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = job_queue[job_id]
    return {
        "id": job_id,
        "status": job["status"],
        "output": job["output"],
        "error": job["error"]
    }

async def process_job_async(job_id: str):
    """Process a job asynchronously"""
    try:
        job = job_queue[job_id]
        
        # Simulate processing time
        await asyncio.sleep(1)
        
        # Call the handler
        result = handler({"input": job["input"]})
        
        # Update job status
        job["status"] = "COMPLETED"
        job["output"] = result
        job["execution_time"] = time.time() - job["start_time"]
        
    except Exception as e:
        job["status"] = "FAILED"
        job["error"] = str(e)
        job["execution_time"] = time.time() - job["start_time"]

@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "message": "RunPod Test Server running"}

@app.get("/jobs")
async def list_jobs():
    """List all jobs"""
    return {
        "total_jobs": len(job_queue),
        "jobs": job_queue
    }

@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job"""
    if job_id not in job_queue:
        raise HTTPException(status_code=404, detail="Job not found")
    
    del job_queue[job_id]
    return {"message": f"Job {job_id} deleted"}

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting RunPod Test Server...")
    print("📱 Test with Postman:")
    print("   - POST /run      -> Simulates RunPod run() method")
    print("   - POST /runsync  -> Simulates RunPod runsync() method")
    print("   - GET  /status/{job_id} -> Check async job status")
    print("   - GET  /jobs     -> List all jobs")
    print("   - GET  /health   -> Health check")
    print("\n🌐 Server will run at: http://127.0.0.1:8001")
    print("📖 API docs at: http://127.0.0.1:8001/docs")
    
    uvicorn.run(app, host="127.0.0.1", port=8001)
