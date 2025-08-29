#!/usr/bin/env python3
"""
Test RunPod Handler Locally
This script tests your handler exactly as RunPod would call it.
"""

import json
from handler import handler

def test_handler():
    """Test the handler with various inputs"""
    
    # Test health endpoint
    print("Testing health endpoint...")
    health_job = {
        "id": "test_health",
        "input": {"action": "health"}
    }
    result = handler(health_job)
    print(f"Health result: {result}")
    
    # Test get_all_tasks endpoint
    print("\nTesting get_all_tasks endpoint...")
    tasks_job = {
        "id": "test_tasks",
        "input": {"action": "get_all_tasks"}
    }
    result = handler(tasks_job)
    print(f"Tasks result: {result}")
    
    # Test unknown action
    print("\nTesting unknown action...")
    unknown_job = {
        "id": "test_unknown",
        "input": {"action": "unknown_action"}
    }
    result = handler(unknown_job)
    print(f"Unknown action result: {result}")

if __name__ == "__main__":
    print("🧪 Testing RunPod Handler Locally...")
    print("=" * 50)
    
    try:
        test_handler()
        print("\n✅ All tests passed! Handler is ready for RunPod deployment.")
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
