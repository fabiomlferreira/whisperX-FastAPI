"""
RunPod handler for WhisperX FastAPI endpoints - PERFECT REPLICATION
This handler provides a perfect replication of all FastAPI endpoints for RunPod serverless.
"""

import base64
import json
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional
from urllib.parse import urlparse
import runpod

# Import your existing services and functions
from app.whisperx_services import process_audio_common, device
from app.services import process_transcribe, process_alignment, process_diarize, process_speaker_assignment
from app.tasks import add_task_to_db, get_task_status_from_db, get_all_tasks_status_from_db, delete_task_from_db
from app.db import get_db_session, engine
from app.schemas import (
    TaskStatus, TaskType, WhisperModelParams, AlignmentParams, DiarizationParams, 
    ASROptions, VADOptions, Transcript, AlignedTranscription, DiarizationSegment,
    SpeechToTextProcessingParams, Response, Result, ResultTasks
)
from app.files import save_temporary_file, validate_extension, ALLOWED_EXTENSIONS
from app.audio import process_audio_file, get_audio_duration
from app.logger import logger
from app.transcript import filter_aligned_transcription
from app.config import Config
from datetime import datetime, timezone
import pandas as pd
import sqlalchemy
import re

# Configure logging
logging.basicConfig(level=logging.INFO)

def clean_query_parameters(params_dict):
    """
    Clean FastAPI Query objects by extracting their default values or actual values.
    
    Args:
        params_dict: Dictionary containing parameters that might be Query objects
        
    Returns:
        Dictionary with clean parameter values
    """
    clean_params = {}
    if params_dict:
        for key, value in params_dict.items():
            # Handle various types of Query-like objects
            if hasattr(value, 'default'):
                # Try to get the default value
                try:
                    if callable(value.default):
                        clean_params[key] = value.default()
                    else:
                        clean_params[key] = value.default
                except Exception:
                    # If default access fails, try to get the actual value
                    try:
                        clean_params[key] = value.value
                    except Exception:
                        # If all else fails, try to convert to string and extract value
                        try:
                            str_value = str(value)
                            # Extract the actual value from the string representation
                            if '(' in str_value and ')' in str_value:
                                # Extract value between parentheses
                                actual_value = str_value.split('(')[1].split(')')[0]
                                # Try to convert to appropriate type
                                if actual_value.isdigit():
                                    clean_params[key] = int(actual_value)
                                elif actual_value in ['true', 'false']:
                                    clean_params[key] = actual_value == 'true'
                                else:
                                    clean_params[key] = actual_value
                            else:
                                clean_params[key] = value
                        except Exception:
                            # If all else fails, use the object as is
                            clean_params[key] = value
            else:
                # Not a Query object, use as is
                clean_params[key] = value
    return clean_params

def handler(job: Dict[str, Any]) -> Dict[str, Any]:
    """
    Main RunPod handler function that perfectly replicates all FastAPI endpoints.
    
    Args:
        job: RunPod job containing 'input' and 'id'
    
    Returns:
        Dict containing the result or error
    """
    try:
        input_data = job.get('input', {})
        action = input_data.get('action', 'health')
        
        logger.info(f"RunPod handler called with action: {action}")
        
        # Route to appropriate function based on action
        if action == 'speech_to_text':
            return handle_speech_to_text(input_data)
        elif action == 'speech_to_text_url':
            return handle_speech_to_text_url(input_data)
        elif action == 'service_transcribe':
            return handle_service_transcribe(input_data)
        elif action == 'service_align':
            return handle_service_align(input_data)
        elif action == 'service_diarize':
            return handle_service_diarize(input_data)
        elif action == 'service_combine':
            return handle_service_combine(input_data)
        elif action == 'get_task':
            return handle_get_task(input_data)
        elif action == 'get_all_tasks':
            return handle_get_all_tasks(input_data)
        elif action == 'delete_task':
            return handle_delete_task(input_data)
        elif action == 'health':
            return handle_health()
        elif action == 'health_live':
            return handle_health_live()
        elif action == 'health_ready':
            return handle_health_ready()
        else:
            return {
                'error': f'Unknown action: {action}',
                'available_actions': [
                    'speech_to_text', 'speech_to_text_url', 
                    'service_transcribe', 'service_align', 'service_diarize', 'service_combine',
                    'get_task', 'get_all_tasks', 'delete_task', 
                    'health', 'health_live', 'health_ready'
                ]
            }
            
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {'error': str(e)}
    try:
        input_data = event.get('input', {})
        action = input_data.get('action', 'health')
        
        logger.info(f"RunPod handler called with action: {action}")
        
        # Route to appropriate function based on action
        if action == 'speech_to_text':
            return handle_speech_to_text(input_data)
        elif action == 'speech_to_text_url':
            return handle_speech_to_text_url(input_data)
        elif action == 'service_transcribe':
            return handle_service_transcribe(input_data)
        elif action == 'service_align':
            return handle_service_align(input_data)
        elif action == 'service_diarize':
            return handle_service_diarize(input_data)
        elif action == 'service_combine':
            return handle_service_combine(input_data)
        elif action == 'get_task':
            return handle_get_task(input_data)
        elif action == 'get_all_tasks':
            return handle_get_all_tasks(input_data)
        elif action == 'delete_task':
            return handle_delete_task(input_data)
        elif action == 'health':
            return handle_health()
        elif action == 'health_live':
            return handle_health_live()
        elif action == 'health_ready':
            return handle_health_ready()
        else:
            return {
                'error': f'Unknown action: {action}',
                'available_actions': [
                    'speech_to_text', 'speech_to_text_url', 
                    'service_transcribe', 'service_align', 'service_diarize', 'service_combine',
                    'get_task', 'get_all_tasks', 'delete_task', 
                    'health', 'health_live', 'health_ready'
                ]
            }
            
    except Exception as e:
        logger.error(f"Handler error: {str(e)}")
        return {'error': str(e)}

def handle_speech_to_text(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perfect replication of /speech-to-text endpoint from stt.py"""
    try:
        # Get file data (base64 or URL)
        file_data = input_data.get('file')
        if not file_data:
            return {'error': 'No file provided'}
        
        # Get parameters with defaults and ensure they are dictionaries
        model_params = input_data.get('model_params', {}) or {}
        align_params = input_data.get('align_params', {}) or {}
        diarize_params = input_data.get('diarize_params', {}) or {}
        asr_options = input_data.get('asr_options', {}) or {}
        vad_options = input_data.get('vad_options', {}) or {}
        
        # Process file
        if file_data.startswith('data:'):
            # Handle base64 data URL
            file_content = base64.b64decode(file_data.split(',')[1])
            filename = input_data.get('filename', 'audio.mp3')
        elif file_data.startswith('http'):
            # Handle URL
            import requests
            response = requests.get(file_data)
            file_content = response.content
            filename = os.path.basename(urlparse(file_data).path) or 'audio.mp3'
        else:
            # Assume base64 string
            file_content = base64.b64decode(file_data)
            filename = input_data.get('filename', 'audio.mp3')
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Validate extension
            validate_extension(filename, ALLOWED_EXTENSIONS)
            
            # Process audio
            audio = process_audio_file(temp_file_path)
            audio_duration = get_audio_duration(audio)
            logger.info("Audio file %s length: %s seconds", filename, audio_duration)
            
            # Add task to database (exact same as endpoint)
            session = next(get_db_session())
            identifier = add_task_to_db(
                status=TaskStatus.processing,
                file_name=filename,
                audio_duration=audio_duration,
                language=model_params.get('language', 'en'),
                task_type=TaskType.full_process,
                task_params={
                    **model_params,
                    **align_params,
                    "asr_options": asr_options,
                    "vad_options": vad_options,
                    **diarize_params,
                },
                start_time=datetime.now(tz=timezone.utc),
                session=session
            )
            logger.info("Task added to database: ID %s", identifier)
            
            # Process full pipeline using process_audio_common (exact same as endpoint)
            # Clean parameters to avoid Query object issues
            clean_model_params = clean_query_parameters(model_params)
            clean_align_params = clean_query_parameters(align_params)
            clean_diarize_params = clean_query_parameters(diarize_params)
            clean_asr_options = clean_query_parameters(asr_options)
            clean_vad_options = clean_query_parameters(vad_options)
            
            # Ensure all required parameters are provided to avoid Query object defaults
            # These are the default values from the schemas
            complete_model_params = {
                "language": "en",
                "task": "transcribe", 
                "model": "tiny",
                "device": "cpu",
                "device_index": 0,
                "threads": 0,
                "batch_size": 8,
                "chunk_size": 20,
                "compute_type": "float16"
            }
            complete_model_params.update(clean_model_params)
            
            complete_align_params = {
                "align_model": None,
                "interpolate_method": "nearest",
                "return_char_alignments": False
            }
            complete_align_params.update(clean_align_params)
            
            complete_diarize_params = {
                "min_speakers": None,
                "max_speakers": None
            }
            complete_diarize_params.update(clean_diarize_params)
            
            # Ensure all required parameters for ASROptions to avoid Query object defaults
            complete_asr_options = {
                "beam_size": 5,
                "best_of": 5,
                "patience": 1.0,
                "length_penalty": 1.0,
                "temperatures": 0.0,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "initial_prompt": None,
                "suppress_tokens": [-1],
                "suppress_numerals": False,
                "hotwords": None
            }
            complete_asr_options.update(clean_asr_options)
            
            # Ensure all required parameters for VADOptions to avoid Query object defaults
            complete_vad_options = {
                "vad_onset": 0.500,
                "vad_offset": 0.363
            }
            complete_vad_options.update(clean_vad_options)
            
            # Create Pydantic models with complete parameters
            whisper_params = WhisperModelParams(**complete_model_params)
            align_params_obj = AlignmentParams(**complete_align_params)
            diarize_params_obj = DiarizationParams(**complete_diarize_params)
            asr_options_obj = ASROptions(**complete_asr_options)
            vad_options_obj = VADOptions(**complete_vad_options)
            
            audio_params = SpeechToTextProcessingParams(
                audio=audio,
                identifier=identifier,
                vad_options=vad_options_obj,
                asr_options=asr_options_obj,
                whisper_model_params=whisper_params,
                alignment_params=align_params_obj,
                diarization_params=diarize_params_obj,
            )
            
            # Call the same function as the FastAPI endpoint
            result = process_audio_common(audio_params, session)
            logger.info("Background task scheduled for processing: ID %s", identifier)
            
            return {
                'identifier': identifier,
                'result': result,
                'message': 'Task queued'
            }
            
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Speech to text error: {str(e)}")
        return {'error': str(e)}

def handle_speech_to_text_url(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perfect replication of /speech-to-text-url endpoint from stt.py"""
    try:
        # Get URL
        url = input_data.get('url')
        if not url:
            return {'error': 'URL is required for speech_to_text_url'}
        
        # Get parameters with defaults
        model_params = input_data.get('model_params', {}) or {}
        align_params = input_data.get('align_params', {}) or {}
        diarize_params = input_data.get('diarize_params', {}) or {}
        asr_options = input_data.get('asr_options', {}) or {}
        vad_options = input_data.get('vad_options', {}) or {}
        
        # Download file from URL (exact same logic as endpoint)
        import requests
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            
            # Check for filename in Content-Disposition header
            content_disposition = response.headers.get("Content-Disposition")
            if content_disposition and "filename=" in content_disposition:
                filename = content_disposition.split("filename=")[1].strip('"')
            else:
                # Fall back to extracting from the URL path
                filename = os.path.basename(url)
                filename = secure_filename(filename)  # Sanitize the filename
            
            # Get the file extension
            _, original_extension = os.path.splitext(filename)
            original_extension = original_extension.lower()
            if original_extension not in {ext.lower() for ext in ALLOWED_EXTENSIONS}:
                raise ValueError(f"Invalid file extension: {original_extension}")
            
            # Save the file to a temporary location
            temp_audio_file = tempfile.NamedTemporaryFile(suffix=original_extension, delete=False)
            for chunk in response.iter_content(chunk_size=8192):
                temp_audio_file.write(chunk)
            temp_audio_file.close()
        
        try:
            logger.info("File downloaded and saved temporarily: %s", temp_audio_file.name)
            validate_extension(temp_audio_file.name, ALLOWED_EXTENSIONS)
            
            # Process audio
            audio = process_audio_file(temp_audio_file.name)
            audio_duration = get_audio_duration(audio)
            logger.info("Audio file processed: duration %s seconds", audio_duration)
            
            # Add task to database (exact same as endpoint)
            session = next(get_db_session())
            identifier = add_task_to_db(
                status=TaskStatus.processing,
                file_name=temp_audio_file.name,
                audio_duration=audio_duration,
                language=model_params.get('language', 'en'),
                task_type=TaskType.full_process,
                task_params={
                    **model_params,
                    **align_params,
                    "asr_options": asr_options,
                    "vad_options": vad_options,
                    **diarize_params,
                },
                url=url,
                start_time=datetime.now(tz=timezone.utc),
                session=session
            )
            logger.info("Task added to database: ID %s", identifier)
            
            # Process full pipeline using process_audio_common (exact same as endpoint)
            # Clean parameters to avoid Query object issues
            clean_model_params = clean_query_parameters(model_params)
            clean_align_params = clean_query_parameters(align_params)
            clean_diarize_params = clean_query_parameters(diarize_params)
            clean_asr_options = clean_query_parameters(asr_options)
            clean_vad_options = clean_query_parameters(vad_options)
            
            # Ensure all required parameters are provided to avoid Query object defaults
            # These are the default values from the schemas
            complete_model_params = {
                "language": "en",
                "task": "transcribe", 
                "model": "tiny",
                "device": "cpu",
                "device_index": 0,
                "threads": 0,
                "batch_size": 8,
                "chunk_size": 20,
                "compute_type": "float16"
            }
            complete_model_params.update(clean_model_params)
            
            complete_align_params = {
                "align_model": None,
                "interpolate_method": "nearest",
                "return_char_alignments": False
            }
            complete_align_params.update(clean_align_params)
            
            complete_diarize_params = {
                "min_speakers": None,
                "max_speakers": None
            }
            complete_diarize_params.update(clean_diarize_params)
            
            # Ensure all required parameters for ASROptions to avoid Query object defaults
            complete_asr_options = {
                "beam_size": 5,
                "best_of": 5,
                "patience": 1.0,
                "length_penalty": 1.0,
                "temperatures": 0.0,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "initial_prompt": None,
                "suppress_tokens": [-1],
                "suppress_numerals": False,
                "hotwords": None
            }
            complete_asr_options.update(clean_asr_options)
            
            # Ensure all required parameters for VADOptions to avoid Query object defaults
            complete_vad_options = {
                "vad_onset": 0.500,
                "vad_offset": 0.363
            }
            complete_vad_options.update(clean_vad_options)
            
            # Create Pydantic models with complete parameters
            whisper_params = WhisperModelParams(**complete_model_params)
            align_params_obj = AlignmentParams(**complete_align_params)
            diarize_params_obj = DiarizationParams(**complete_diarize_params)
            asr_options_obj = ASROptions(**complete_asr_options)
            vad_options_obj = VADOptions(**complete_vad_options)
            
            audio_params = SpeechToTextProcessingParams(
                audio=audio,
                identifier=identifier,
                vad_options=vad_options_obj,
                asr_options=asr_options_obj,
                whisper_model_params=whisper_params,
                alignment_params=align_params_obj,
                diarization_params=diarize_params_obj,
            )
            
            # Call the same function as the FastAPI endpoint
            result = process_audio_common(audio_params, session)
            logger.info("Background task scheduled for processing: ID %s", identifier)
            
            return {
                'identifier': identifier,
                'result': result,
                'message': 'Task queued'
            }
            
        finally:
            # Clean up temp file
            os.unlink(temp_audio_file.name)
            
    except Exception as e:
        logger.error(f"Speech to text URL error: {str(e)}")
        return {'error': str(e)}

def handle_service_transcribe(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perfect replication of /service/transcribe endpoint from stt_services.py"""
    try:
        # Get file data (base64 or URL)
        file_data = input_data.get('file')
        if not file_data:
            return {'error': 'No file provided'}
        
        # Get parameters with defaults
        model_params = input_data.get('model_params', {}) or {}
        asr_options = input_data.get('asr_options', {}) or {}
        vad_options = input_data.get('vad_options', {}) or {}
        
        # Process file
        if file_data.startswith('data:'):
            file_content = base64.b64decode(file_data.split(',')[1])
            filename = input_data.get('filename', 'audio.mp3')
        elif file_data.startswith('http'):
            import requests
            response = requests.get(file_data)
            file_content = response.content
            filename = os.path.basename(urlparse(file_data).path) or 'audio.mp3'
        else:
            file_content = base64.b64decode(file_data)
            filename = input_data.get('filename', 'audio.mp3')
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Validate extension
            validate_extension(filename, ALLOWED_EXTENSIONS)
            
            # Process audio
            audio = process_audio_file(temp_file_path)
            audio_duration = get_audio_duration(audio)
            
            # Add task to database (exact same as endpoint)
            session = next(get_db_session())
            identifier = add_task_to_db(
                status=TaskStatus.processing,
                file_name=filename,
                audio_duration=audio_duration,
                language=model_params.get('language', 'en'),
                task_type=TaskType.transcription,
                task_params={
                    **model_params,
                    "asr_options": asr_options,
                    "vad_options": vad_options,
                },
                start_time=datetime.now(tz=timezone.utc),
                session=session
            )
            
            # Process transcription only using process_transcribe (exact same signature as endpoint)
            # Clean parameters to avoid Query object issues
            clean_model_params = clean_query_parameters(model_params)
            clean_asr_options = clean_query_parameters(asr_options)
            clean_vad_options = clean_query_parameters(vad_options)
            
            # Ensure all required parameters are provided to avoid Query object defaults
            complete_model_params = {
                "language": "en",
                "task": "transcribe", 
                "model": "tiny",
                "device": "cpu",
                "device_index": 0,
                "threads": 0,
                "batch_size": 8,
                "chunk_size": 20,
                "compute_type": "float16"
            }
            complete_model_params.update(clean_model_params)
            
            # Ensure all required parameters for ASROptions to avoid Query object defaults
            complete_asr_options = {
                "beam_size": 5,
                "best_of": 5,
                "patience": 1.0,
                "length_penalty": 1.0,
                "temperatures": 0.0,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
                "initial_prompt": None,
                "suppress_tokens": [-1],
                "suppress_numerals": False,
                "hotwords": None
            }
            complete_asr_options.update(clean_asr_options)
            
            # Ensure all required parameters for VADOptions to avoid Query object defaults
            complete_vad_options = {
                "vad_onset": 0.500,
                "vad_offset": 0.363
            }
            complete_vad_options.update(clean_vad_options)
            
            # Create Pydantic models with complete parameters
            whisper_params = WhisperModelParams(**complete_model_params)
            asr_options_obj = ASROptions(**complete_asr_options)
            vad_options_obj = VADOptions(**complete_vad_options)
            
            result = process_transcribe(
                audio=audio,
                identifier=identifier,
                model_params=whisper_params,
                asr_options=asr_options_obj,
                vad_options=vad_options_obj,
                session=session
            )
            
            logger.info("Background task scheduled for processing: ID %s", identifier)
            return {
                'identifier': identifier,
                'result': result,
                'message': 'Task queued'
            }
            
        finally:
            # Clean up temp file
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Service transcribe error: {str(e)}")
        return {'error': str(e)}

def handle_service_align(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perfect replication of /service/align endpoint from stt_services.py"""
    try:
        # Get file and transcript data
        file_data = input_data.get('file')
        transcript_data = input_data.get('transcript')
        
        if not file_data or not transcript_data:
            return {'error': 'Both file and transcript are required'}
        
        # Get parameters
        align_params = input_data.get('align_params', {}) or {}
        device_param = input_data.get('device', device)
        
        # Process audio file
        if file_data.startswith('data:'):
            file_content = base64.b64decode(file_data.split(',')[1])
            filename = input_data.get('filename', 'audio.mp3')
        elif file_data.startswith('http'):
            import requests
            response = requests.get(file_data)
            file_content = response.content
            filename = os.path.basename(urlparse(file_data).path) or 'audio.mp3'
        else:
            file_content = base64.b64decode(file_data)
            filename = input_data.get('filename', 'audio.mp3')
        
        # Save temporary audio file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Process transcript data (should be JSON)
            if isinstance(transcript_data, str):
                # Assume it's JSON string
                transcript_dict = json.loads(transcript_data)
            else:
                # Assume it's already a dict
                transcript_dict = transcript_data
            
            # Create Transcript object (exact same as endpoint)
            transcript = Transcript(**transcript_dict)
            
            # Validate extension
            validate_extension(filename, ALLOWED_EXTENSIONS)
            
            # Process audio
            audio = process_audio_file(temp_file_path)
            audio_duration = get_audio_duration(audio)
            
            # Add task to database (exact same as endpoint)
            session = next(get_db_session())
            identifier = add_task_to_db(
                status=TaskStatus.processing,
                file_name=filename,
                audio_duration=audio_duration,
                language=transcript.language,
                task_type=TaskType.transcription_alignment,
                task_params={
                    **align_params,
                    "device": device_param,
                },
                start_time=datetime.now(tz=timezone.utc),
                session=session
            )
            
            # Process alignment using process_alignment (exact same signature as endpoint)
            # Clean parameters to avoid Query object issues
            clean_align_params = clean_query_parameters(align_params)
            
            # Ensure all required parameters are provided to avoid Query object defaults
            complete_align_params = {
                "align_model": None,
                "interpolate_method": "nearest",
                "return_char_alignments": False
            }
            complete_align_params.update(clean_align_params)
            
            # Create Pydantic models with clean parameters
            align_params_obj = AlignmentParams(**complete_align_params)
            
            result = process_alignment(
                audio=audio,
                transcript=transcript.model_dump(),
                identifier=identifier,
                device=device_param,
                align_params=align_params_obj,
                session=session
            )
            
            logger.info("Background task scheduled for processing: ID %s", identifier)
            return {
                'identifier': identifier,
                'result': result,
                'message': 'Task queued'
            }
            
        finally:
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Service align error: {str(e)}")
        return {'error': str(e)}

def handle_service_diarize(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perfect replication of /service/diarize endpoint from stt_services.py"""
    try:
        file_data = input_data.get('file')
        if not file_data:
            return {'error': 'No file provided'}
        
        # Get parameters
        diarize_params = input_data.get('diarize_params', {}) or {}
        device_param = input_data.get('device', device)
        
        # Process file
        if file_data.startswith('data:'):
            file_content = base64.b64decode(file_data.split(',')[1])
            filename = input_data.get('filename', 'audio.mp3')
        elif file_data.startswith('http'):
            import requests
            response = requests.get(file_data)
            file_content = response.content
            filename = os.path.basename(urlparse(file_data).path) or 'audio.mp3'
        else:
            file_content = base64.b64decode(file_data)
            filename = input_data.get('filename', 'audio.mp3')
        
        # Save temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(filename)[1]) as temp_file:
            temp_file.write(file_content)
            temp_file_path = temp_file.name
        
        try:
            # Validate extension
            validate_extension(filename, ALLOWED_EXTENSIONS)
            
            # Process audio
            audio = process_audio_file(temp_file_path)
            audio_duration = get_audio_duration(audio)
            
            # Add task to database (exact same as endpoint)
            session = next(get_db_session())
            identifier = add_task_to_db(
                status=TaskStatus.processing,
                file_name=filename,
                audio_duration=audio_duration,
                task_type=TaskType.diarization,
                task_params={
                    **diarize_params,
                    "device": device_param,
                },
                start_time=datetime.now(tz=timezone.utc),
                session=session
            )
            
            # Process diarization using process_diarize (exact same signature as endpoint)
            # Clean parameters to avoid Query object issues
            clean_diarize_params = clean_query_parameters(diarize_params)
            
            # Ensure all required parameters are provided to avoid Query object defaults
            complete_diarize_params = {
                "min_speakers": None,
                "max_speakers": None
            }
            complete_diarize_params.update(clean_diarize_params)
            
            # Create Pydantic models with clean parameters
            diarize_params_obj = DiarizationParams(**complete_diarize_params)
            
            result = process_diarize(
                audio=audio,
                identifier=identifier,
                device=device_param,
                diarize_params=diarize_params_obj,
                session=session
            )
            
            logger.info("Background task scheduled for processing: ID %s", identifier)
            return {
                'identifier': identifier,
                'result': result,
                'message': 'Task queued'
            }
            
        finally:
            os.unlink(temp_file_path)
            
    except Exception as e:
        logger.error(f"Service diarize error: {str(e)}")
        return {'error': str(e)}

def handle_service_combine(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perfect replication of /service/combine endpoint from stt_services.py"""
    try:
        aligned_transcript_data = input_data.get('aligned_transcript')
        diarization_result_data = input_data.get('diarization_result')
        
        if not aligned_transcript_data or not diarization_result_data:
            return {'error': 'Both aligned_transcript and diarization_result are required'}
        
        # Process aligned transcript data
        if isinstance(aligned_transcript_data, str):
            transcript_dict = json.loads(aligned_transcript_data)
        else:
            transcript_dict = aligned_transcript_data
        
        # Create AlignedTranscription object and filter it (exact same as endpoint)
        transcript = AlignedTranscription(**transcript_dict)
        transcript = filter_aligned_transcription(transcript)
        
        # Process diarization result data
        if isinstance(diarization_result_data, str):
            diarization_list = json.loads(diarization_result_data)
        else:
            diarization_list = diarization_result_data
        
        # Create DiarizationSegment objects (exact same as endpoint)
        diarization_segments = []
        for item in diarization_list:
            diarization_segments.append(DiarizationSegment(**item))
        
        # Add task to database (exact same as endpoint)
        session = next(get_db_session())
        identifier = add_task_to_db(
            status=TaskStatus.processing,
            file_name=None,
            task_type=TaskType.combine_transcript_diarization,
            start_time=datetime.now(tz=timezone.utc),
            session=session
        )
        
        # Process combination using process_speaker_assignment (exact same signature as endpoint)
        result = process_speaker_assignment(
            diarization=pd.json_normalize([segment.model_dump() for segment in diarization_segments]),
            transcript=transcript.model_dump(),
            identifier=identifier,
            session=session
        )
        
        logger.info("Background task scheduled for processing: ID %s", identifier)
        return {
            'identifier': identifier,
            'result': result,
            'message': 'Task queued'
        }
        
    except Exception as e:
        logger.error(f"Service combine error: {str(e)}")
        return {'error': str(e)}

def handle_get_task(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perfect replication of /task/{identifier} endpoint from task.py"""
    try:
        identifier = input_data.get('identifier')
        if not identifier:
            return {'error': 'Task identifier required'}
        
        session = next(get_db_session())
        result = get_task_status_from_db(identifier, session)
        
        if result:
            logger.info("Status retrieved for task ID: %s", identifier)
            return {'result': result} # result is already a dict
        else:
            logger.error("Task ID not found: %s", identifier)
            return {'error': 'Identifier not found'}
            
    except Exception as e:
        logger.error(f"Get task error: {str(e)}")
        return {'error': str(e)}

def handle_get_all_tasks(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perfect replication of /task/all endpoint from task.py"""
    try:
        logger.info("Retrieving status of all tasks")
        session = next(get_db_session())
        result = get_all_tasks_status_from_db(session)
        
        return {'result': result.model_dump()} # Convert Pydantic model to dict
        
    except Exception as e:
        logger.error(f"Get all tasks error: {str(e)}")
        return {'error': str(e)}

def handle_delete_task(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Perfect replication of /task/{identifier}/delete endpoint from task.py"""
    try:
        identifier = input_data.get('identifier')
        if not identifier:
            return {'error': 'Task identifier required'}
        
        logger.info("Deleting task ID: %s", identifier)
        session = next(get_db_session())
        success = delete_task_from_db(identifier, session)
        
        if success:
            logger.info("Task deleted: ID %s", identifier)
            return {'identifier': identifier, 'message': 'Task deleted'}
        else:
            logger.error("Task not found: ID %s", identifier)
            return {'error': 'Task not found'}
            
    except Exception as e:
        logger.error(f"Delete task error: {str(e)}")
        return {'error': str(e)}

def handle_health() -> Dict[str, Any]:
    """Perfect replication of /health endpoint from main.py"""
    return {
        'status': 'ok',
        'message': 'Service is running'
    }

def handle_health_live() -> Dict[str, Any]:
    """Perfect replication of /health/live endpoint from main.py"""
    return {
        'status': 'ok',
        'timestamp': time.time(),
        'message': 'Application is live'
    }

def handle_health_ready() -> Dict[str, Any]:
    """Perfect replication of /health/ready endpoint from main.py"""
    try:
        # Check database connection (exact same logic as endpoint)
        with engine.connect() as conn:
            conn.execute(sqlalchemy.text("SELECT 1"))
        
        return {
            'status': 'ok',
            'database': 'connected',
            'message': 'Application is ready to accept requests'
        }
    except Exception:
        logging.exception("Readiness check failed:")
        return {
            'status': 'error',
            'database': 'disconnected',
            'message': 'Application is not ready due to an internal error.'
        }

# Helper function for secure filename (exact same as stt.py)
def secure_filename(filename):
    """Sanitize the filename to ensure it is safe for use in file systems."""
    filename = os.path.basename(filename)
    # Only allow alphanumerics, dash, underscore, and dot
    filename = re.sub(r"[^A-Za-z0-9_.-]", "_", filename)
    # Replace multiple consecutive dots or underscores with a single underscore
    filename = re.sub(r"[._]{2,}", "_", filename)
    # Remove leading dots or underscores
    filename = re.sub(r"^[._]+", "", filename)
    # Ensure filename is not empty or problematic
    if not filename or filename in {".", ".."}:
        raise ValueError(
            "Filename is empty or contains only special characters after sanitization."
        )
    return filename

# RunPod serverless start - this keeps the container alive and makes it serverless
if __name__ == "__main__":
    # Initialize database tables first
    from app.models import Base
    from app.db import engine
    Base.metadata.create_all(bind=engine)
    print("Database tables created successfully")
    
    # Then start RunPod serverless
    import runpod
    runpod.serverless.start({"handler": handler})
