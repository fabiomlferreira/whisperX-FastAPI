#!/usr/bin/env python3
"""
Simple test script for the RunPod handler
"""

import json
from handler import handler

def test_health():
    """Test health endpoint"""
    event = {"input": {"action": "health"}}
    result = handler(event)
    print("Health test:", result)
    assert result.get('status') == 'ok'

def test_health_live():
    """Test health live endpoint"""
    event = {"input": {"action": "health_live"}}
    result = handler(event)
    print("Health live test:", result)
    assert result.get('status') == 'ok'
    assert 'timestamp' in result

def test_health_ready():
    """Test health ready endpoint"""
    event = {"input": {"action": "health_ready"}}
    result = handler(event)
    print("Health ready test:", result)
    assert 'status' in result
    assert 'database' in result

def test_get_all_tasks():
    """Test get all tasks endpoint"""
    event = {"input": {"action": "get_all_tasks"}}
    result = handler(event)
    print("Get all tasks test:", result)
    assert 'result' in result or 'error' in result

def test_speech_to_text_url():
    """Test speech to text URL endpoint"""
    event = {
        "input": {
            "action": "speech_to_text_url",
            "url": "https://assets.noop.pt/output5m.mp3",
            "model_params": {
                "language": "pt",
                "model": "tiny",
                "device": "cpu"
            }
        }
    }
    result = handler(event)
    print("Speech to text URL test:", result)
    assert 'identifier' in result or 'error' in result

if __name__ == "__main__":
    print("Testing RunPod handler...")
    
    try:
        test_health()
        test_health_live()
        test_health_ready()
        test_get_all_tasks()
        test_speech_to_text_url()
        print("✅ All tests passed!")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
