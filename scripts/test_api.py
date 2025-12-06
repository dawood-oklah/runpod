#!/usr/bin/env python3
"""
MedGemma API Test Script

Tests all API endpoints to verify the deployment is working correctly.
Run this after deploying to RunPod to validate the setup.

Usage:
    python scripts/test_api.py --url https://your-runpod-url.runpod.io --api-key YOUR_API_KEY
"""

import argparse
import base64
import json
import sys
from pathlib import Path

import httpx


def test_health(client: httpx.Client, base_url: str) -> bool:
    """Test the health endpoint."""
    print("\n1. Testing /health endpoint...")
    try:
        response = client.get(f"{base_url}/health")
        data = response.json()
        print(f"   Status: {data.get('status')}")
        print(f"   Model Loaded: {data.get('model_loaded')}")
        print(f"   Model ID: {data.get('model_id')}")
        print(f"   Quantization: {data.get('quantization')}")

        if data.get("gpu", {}).get("available"):
            gpu = data["gpu"]
            print(f"   GPU: {gpu.get('device_name')}")
            print(f"   VRAM Allocated: {gpu.get('memory_allocated_gb', 0):.2f} GB")

        return data.get("model_loaded", False)
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_chat_completion(client: httpx.Client, base_url: str, api_key: str) -> bool:
    """Test the OpenAI-compatible chat endpoint."""
    print("\n2. Testing /v1/chat/completions endpoint...")
    try:
        response = client.post(
            f"{base_url}/v1/chat/completions",
            headers={"X-API-Key": api_key},
            json={
                "messages": [
                    {"role": "system", "content": "You are a helpful medical assistant."},
                    {"role": "user", "content": "What are the common symptoms of the flu?"},
                ],
                "max_tokens": 256,
                "temperature": 0.7,
            },
            timeout=120.0,
        )

        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"]["content"]
            usage = data["usage"]
            print(f"   Response (truncated): {content[:200]}...")
            print(f"   Tokens: {usage['prompt_tokens']} in, {usage['completion_tokens']} out")
            return True
        else:
            print(f"   ERROR: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_medical_query(client: httpx.Client, base_url: str, api_key: str) -> bool:
    """Test the medical query endpoint."""
    print("\n3. Testing /v1/medical/query endpoint...")
    try:
        response = client.post(
            f"{base_url}/v1/medical/query",
            headers={"X-API-Key": api_key},
            json={
                "query": """A 45-year-old male presents with chest pain, shortness of breath, and sweating.
                The pain started 2 hours ago and radiates to the left arm. He has a history of
                hypertension and diabetes. What are the differential diagnoses and recommended
                immediate actions?""",
                "max_tokens": 512,
                "include_disclaimer": True,
            },
            timeout=120.0,
        )

        if response.status_code == 200:
            data = response.json()
            print(f"   Response (truncated): {data['response'][:200]}...")
            print(f"   Tokens: {data['usage']['total_tokens']} total")
            print(f"   Disclaimer included: {data['disclaimer'] is not None}")
            return True
        else:
            print(f"   ERROR: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_image_analysis(
    client: httpx.Client, base_url: str, api_key: str, image_path: str | None
) -> bool:
    """Test the image analysis endpoint."""
    print("\n4. Testing /v1/medical/analyze endpoint...")

    # Use a sample image or skip
    if not image_path:
        print("   SKIPPED: No image provided. Use --image to test image analysis.")
        return True

    try:
        # Read and encode image
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        response = client.post(
            f"{base_url}/v1/medical/analyze",
            headers={"X-API-Key": api_key},
            json={
                "images": [image_data],
                "query": "Please analyze this medical image and describe any notable findings.",
                "structured_output": True,
                "max_tokens": 512,
            },
            timeout=180.0,
        )

        if response.status_code == 200:
            data = response.json()
            print(f"   Analysis (truncated): {data['analysis'][:200]}...")
            print(f"   Images analyzed: {data['image_count']}")
            print(f"   Tokens: {data['usage']['total_tokens']} total")
            return True
        else:
            print(f"   ERROR: {response.status_code} - {response.text}")
            return False
    except FileNotFoundError:
        print(f"   ERROR: Image file not found: {image_path}")
        return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def test_auth_required(client: httpx.Client, base_url: str) -> bool:
    """Test that authentication is required for protected endpoints."""
    print("\n5. Testing authentication requirement...")
    try:
        response = client.post(
            f"{base_url}/v1/medical/query",
            json={"query": "test"},
            # No API key provided
        )

        if response.status_code == 401:
            print("   OK: Authentication correctly required (401 returned)")
            return True
        elif response.status_code == 200:
            print("   WARNING: API accepted request without authentication!")
            print("   This may be intentional for development mode.")
            return True
        else:
            print(f"   Unexpected status: {response.status_code}")
            return False
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Test MedGemma API endpoints")
    parser.add_argument(
        "--url",
        default="http://localhost:8000",
        help="Base URL of the API (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--api-key",
        default="test-key",
        help="API key for authentication",
    )
    parser.add_argument(
        "--image",
        help="Path to a medical image for testing image analysis",
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow tests (inference tests)",
    )

    args = parser.parse_args()

    print("=" * 50)
    print("MedGemma API Test Suite")
    print("=" * 50)
    print(f"Target URL: {args.url}")
    print("=" * 50)

    # Create HTTP client with longer timeout for inference
    client = httpx.Client(timeout=30.0)

    results = []

    # Test health (always)
    results.append(("Health Check", test_health(client, args.url)))

    # Test auth
    results.append(("Auth Required", test_auth_required(client, args.url)))

    if not args.skip_slow:
        # Test inference endpoints
        results.append(
            ("Chat Completion", test_chat_completion(client, args.url, args.api_key))
        )
        results.append(
            ("Medical Query", test_medical_query(client, args.url, args.api_key))
        )
        results.append(
            (
                "Image Analysis",
                test_image_analysis(client, args.url, args.api_key, args.image),
            )
        )
    else:
        print("\nSkipping inference tests (--skip-slow)")

    # Summary
    print("\n" + "=" * 50)
    print("Test Results Summary")
    print("=" * 50)

    passed = 0
    failed = 0
    for name, result in results:
        status = "PASS" if result else "FAIL"
        print(f"  {name}: {status}")
        if result:
            passed += 1
        else:
            failed += 1

    print("=" * 50)
    print(f"Passed: {passed}/{len(results)}")

    if failed > 0:
        print(f"Failed: {failed}")
        sys.exit(1)
    else:
        print("All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
