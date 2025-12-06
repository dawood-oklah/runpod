# MedGemma API Documentation

**Version:** 1.0.0
**Base URL:** `https://your-runpod-id-8000.proxy.runpod.net`
**API Key:** `david`

---

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints Overview](#endpoints-overview)
3. [Health Check](#health-check)
4. [Chat Completions (OpenAI-Compatible)](#chat-completions)
5. [Medical Query](#medical-query)
6. [Medical Image Analysis](#medical-image-analysis)
7. [Medical Report Generation](#medical-report-generation)
8. [Error Handling](#error-handling)
9. [Code Examples](#code-examples)

---

## Authentication

All endpoints except `/health` require API key authentication via the `X-API-Key` header.

```
X-API-Key: david
```

**Authentication Errors:**

| Status Code | Error Code | Description |
|-------------|------------|-------------|
| 401 | `missing_api_key` | No API key provided |
| 403 | `invalid_api_key` | Invalid API key |

---

## Endpoints Overview

| Method | Endpoint | Description | Auth Required |
|--------|----------|-------------|---------------|
| GET | `/health` | Health check and model status | No |
| POST | `/v1/chat/completions` | OpenAI-compatible chat | Yes |
| POST | `/v1/medical/query` | Text-only medical queries | Yes |
| POST | `/v1/medical/analyze` | Medical image analysis | Yes |
| POST | `/v1/medical/report` | Medical report generation | Yes |

---

## Health Check

Check service status and GPU information.

**Endpoint:** `GET /health`
**Authentication:** Not required

### Request

```bash
curl -X GET "https://your-runpod-id-8000.proxy.runpod.net/health"
```

### Response

```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_id": "google/medgemma-4b-it",
  "quantization": "8bit",
  "gpu": {
    "available": true,
    "device_count": 1,
    "current_device": 0,
    "device_name": "NVIDIA A6000",
    "memory_allocated_gb": 15.32,
    "memory_reserved_gb": 16.50
  },
  "version": "1.0.0",
  "timestamp": 1733500800
}
```

### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `healthy`, `loading`, or `error` |
| `model_loaded` | boolean | Whether model is ready for inference |
| `model_id` | string | HuggingFace model identifier |
| `quantization` | string | Quantization method (`none`, `4bit`, `8bit`) |
| `gpu.available` | boolean | GPU availability |
| `gpu.device_name` | string | GPU model name |
| `gpu.memory_allocated_gb` | float | Current GPU memory usage |

---

## Chat Completions

OpenAI-compatible chat completion endpoint. Works with OpenAI client libraries.

**Endpoint:** `POST /v1/chat/completions`
**Authentication:** Required

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `messages` | array | Yes | - | List of conversation messages |
| `max_tokens` | integer | No | 2048 | Maximum tokens to generate (1-8192) |
| `temperature` | float | No | 0.7 | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | 0.9 | Nucleus sampling (0.0-1.0) |
| `top_k` | integer | No | 50 | Top-k sampling (1-100) |
| `stream` | boolean | No | false | Streaming (not yet supported) |

### Message Object

| Field | Type | Description |
|-------|------|-------------|
| `role` | string | `system`, `user`, or `assistant` |
| `content` | string | Message content (max 100,000 characters) |

### Example Request

```bash
curl -X POST "https://your-runpod-id-8000.proxy.runpod.net/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: david" \
  -d '{
    "messages": [
      {
        "role": "system",
        "content": "You are a helpful medical assistant."
      },
      {
        "role": "user",
        "content": "What are the common symptoms of Type 2 Diabetes?"
      }
    ],
    "max_tokens": 1024,
    "temperature": 0.7
  }'
```

### Example Response

```json
{
  "id": "chatcmpl-abc12345",
  "object": "chat.completion",
  "created": 1733500800,
  "model": "google/medgemma-4b-it",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Common symptoms of Type 2 Diabetes include:\n\n1. **Increased thirst (polydipsia)**\n2. **Frequent urination (polyuria)**\n3. **Unexplained weight loss**\n4. **Fatigue and weakness**\n5. **Blurred vision**\n6. **Slow-healing wounds**\n7. **Tingling or numbness in hands/feet**\n8. **Recurrent infections**\n\nMany people with Type 2 Diabetes may have no symptoms initially, which is why regular screening is important for at-risk individuals."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 45,
    "completion_tokens": 128,
    "total_tokens": 173
  }
}
```

### Using with OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://your-runpod-id-8000.proxy.runpod.net/v1",
    api_key="david"
)

response = client.chat.completions.create(
    model="medgemma",  # Model name is ignored, uses loaded model
    messages=[
        {"role": "system", "content": "You are a helpful medical assistant."},
        {"role": "user", "content": "What are the symptoms of pneumonia?"}
    ],
    max_tokens=1024,
    temperature=0.7
)

print(response.choices[0].message.content)
```

---

## Medical Query

Text-only medical queries optimized for long prompts (up to ~10,000 tokens).

**Endpoint:** `POST /v1/medical/query`
**Authentication:** Required

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `query` | string | Yes | - | Medical question or case description |
| `system_prompt` | string | No | null | Custom system prompt |
| `conversation_history` | array | No | null | Previous conversation messages |
| `max_tokens` | integer | No | 2048 | Maximum tokens to generate |
| `temperature` | float | No | 0.7 | Sampling temperature |
| `include_disclaimer` | boolean | No | true | Include medical disclaimer |

### Use Cases

- Clinical case analysis
- Differential diagnosis
- Treatment recommendations
- Medical literature questions
- Patient history analysis

### Example Request - Clinical Case

```bash
curl -X POST "https://your-runpod-id-8000.proxy.runpod.net/v1/medical/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: david" \
  -d '{
    "query": "A 55-year-old male presents with crushing chest pain radiating to the left arm, diaphoresis, and shortness of breath that started 2 hours ago. He has a history of hypertension and hyperlipidemia. Vitals: BP 160/95, HR 110, SpO2 94% on room air. What is the most likely diagnosis and immediate management steps?",
    "max_tokens": 2048,
    "temperature": 0.3,
    "include_disclaimer": true
  }'
```

### Example Response

```json
{
  "response": "Based on the clinical presentation, the most likely diagnosis is **Acute Coronary Syndrome (ACS)**, specifically concerning for **ST-Elevation Myocardial Infarction (STEMI)** or Non-STEMI.\n\n**Key Findings Supporting Diagnosis:**\n- Classic crushing chest pain with left arm radiation\n- Diaphoresis (autonomic response)\n- Cardiovascular risk factors (HTN, hyperlipidemia)\n- Elevated heart rate and blood pressure\n\n**Immediate Management:**\n\n1. **MONA Protocol:**\n   - Morphine (if pain persists)\n   - Oxygen (if SpO2 < 94%)\n   - Nitroglycerin (sublingual)\n   - Aspirin (325mg chewable)\n\n2. **Urgent Investigations:**\n   - 12-lead ECG (within 10 minutes)\n   - Serial troponins\n   - CBC, BMP, coagulation studies\n   - Chest X-ray\n\n3. **Activate Cath Lab** if STEMI confirmed on ECG\n\n4. **Anticoagulation** per protocol (heparin/enoxaparin)\n\n5. **Cardiology consultation** immediately",
  "usage": {
    "prompt_tokens": 156,
    "completion_tokens": 287,
    "total_tokens": 443
  },
  "model": "google/medgemma-4b-it",
  "disclaimer": "\n**DISCLAIMER**: This response is generated by an AI model and is for informational\npurposes only. It should not be used as a substitute for professional medical advice,\ndiagnosis, or treatment. Always seek the advice of a qualified healthcare provider\nwith any questions you may have regarding a medical condition.\n",
  "created": 1733500800
}
```

### Example - Differential Diagnosis

```bash
curl -X POST "https://your-runpod-id-8000.proxy.runpod.net/v1/medical/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: david" \
  -d '{
    "query": "Provide a differential diagnosis for a 30-year-old female with fatigue, weight gain, cold intolerance, constipation, and dry skin for the past 6 months.",
    "system_prompt": "You are an endocrinologist. Provide differential diagnoses ranked by likelihood with key distinguishing features.",
    "max_tokens": 1500
  }'
```

---

## Medical Image Analysis

Analyze medical images including X-rays, CT, MRI, pathology slides, and dermatology images.

**Endpoint:** `POST /v1/medical/analyze`
**Authentication:** Required

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `images` | array[string] | Yes | - | Base64-encoded images (1-4 images) |
| `query` | string | Yes | - | Question about the image(s) |
| `image_modality` | string | No | null | Type: `chest_xray`, `ct_scan`, `mri`, `pathology`, `dermatology` |
| `clinical_context` | string | No | null | Patient history, symptoms |
| `system_prompt` | string | No | null | Custom system prompt |
| `max_tokens` | integer | No | 2048 | Maximum tokens to generate |
| `temperature` | float | No | 0.7 | Sampling temperature |
| `structured_output` | boolean | No | false | Return structured findings |

### Supported Image Modalities

- **Radiology:** X-ray, CT, MRI
- **Pathology:** Histopathology slides
- **Dermatology:** Skin lesions, conditions
- **Ophthalmology:** Retinal scans
- **Clinical photography**

### Image Format

Images must be provided as:
- **Base64 encoded string:** `"iVBORw0KGgoAAAANS..."`
- **Data URL:** `"data:image/png;base64,iVBORw0KGgoAAAANS..."`

Maximum image size: 20MB per image
Recommended resolution: 896x896 (images are normalized internally)

### Example Request - Chest X-Ray

```bash
curl -X POST "https://your-runpod-id-8000.proxy.runpod.net/v1/medical/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: david" \
  -d '{
    "images": ["data:image/png;base64,iVBORw0KGgoAAAANS..."],
    "query": "Analyze this chest X-ray and describe any abnormalities.",
    "image_modality": "chest_xray",
    "clinical_context": "65-year-old male smoker with 2-week history of cough and fever",
    "structured_output": true,
    "max_tokens": 2048
  }'
```

### Example Response

```json
{
  "analysis": "**Chest X-Ray Analysis**\n\n**Observations:**\n- Frontal chest radiograph of adequate quality\n- Heart size is within normal limits (cardiothoracic ratio < 0.5)\n- Mediastinal contours appear normal\n\n**Findings:**\n1. Right lower lobe consolidation with air bronchograms\n2. Blunting of the right costophrenic angle suggesting small pleural effusion\n3. No pneumothorax identified\n4. Osseous structures appear intact\n\n**Assessment:**\nFindings consistent with right lower lobe pneumonia, likely community-acquired given clinical history. Small associated parapneumonic effusion.\n\n**Recommendations:**\n1. Clinical correlation with inflammatory markers\n2. Consider CT chest if symptoms worsen\n3. Follow-up chest X-ray in 4-6 weeks to confirm resolution",
  "findings": [
    "1. Right lower lobe consolidation with air bronchograms",
    "2. Blunting of the right costophrenic angle suggesting small pleural effusion",
    "3. No pneumothorax identified",
    "4. Osseous structures appear intact"
  ],
  "usage": {
    "prompt_tokens": 412,
    "completion_tokens": 234,
    "total_tokens": 646
  },
  "model": "google/medgemma-4b-it",
  "image_count": 1,
  "created": 1733500800
}
```

### Example - Dermatology

```bash
curl -X POST "https://your-runpod-id-8000.proxy.runpod.net/v1/medical/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: david" \
  -d '{
    "images": ["BASE64_IMAGE_DATA_HERE"],
    "query": "Evaluate this skin lesion and provide a differential diagnosis.",
    "image_modality": "dermatology",
    "clinical_context": "45-year-old female, lesion on upper back present for 6 months, slowly enlarging, occasional itching",
    "structured_output": true
  }'
```

---

## Medical Report Generation

Generate structured medical reports from images following clinical documentation standards.

**Endpoint:** `POST /v1/medical/report`
**Authentication:** Required

### Request Body

| Field | Type | Required | Default | Description |
|-------|------|----------|---------|-------------|
| `images` | array[string] | Yes | - | Base64-encoded images (1-4 images) |
| `report_type` | string | No | `general` | Report type (see below) |
| `patient_info` | string | No | null | Patient demographics and history |
| `clinical_indication` | string | No | null | Reason for examination |
| `comparison_studies` | string | No | null | Prior studies for comparison |
| `max_tokens` | integer | No | 4096 | Maximum tokens to generate |

### Report Types

| Type | Description |
|------|-------------|
| `radiology` | Standard radiology report format |
| `pathology` | Pathology/histology report |
| `dermatology` | Dermatology assessment |
| `ophthalmology` | Eye examination report |
| `general` | Generic medical image report |

### Example Request - Radiology Report

```bash
curl -X POST "https://your-runpod-id-8000.proxy.runpod.net/v1/medical/report" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: david" \
  -d '{
    "images": ["BASE64_IMAGE_DATA_HERE"],
    "report_type": "radiology",
    "patient_info": "72-year-old female",
    "clinical_indication": "Fall from standing height, left hip pain",
    "comparison_studies": "No prior imaging available",
    "max_tokens": 4096
  }'
```

### Example Response

```json
{
  "report": "PATIENT INFORMATION: 72-year-old female\n\n1. EXAMINATION: Left Hip Radiograph - AP and Lateral views\n\n2. CLINICAL INDICATION: Fall from standing height, left hip pain\n\n3. COMPARISON: No prior imaging available\n\n4. TECHNIQUE: Standard AP pelvis and lateral left hip radiographs\n\n5. FINDINGS:\n- There is a displaced intracapsular fracture of the left femoral neck\n- Fracture line is oriented vertically (Pauwels Type III)\n- Femoral head appears viable without evidence of avascular necrosis\n- No additional fractures of the pelvis identified\n- Right hip demonstrates moderate degenerative changes\n- Soft tissue swelling noted over the left hip region\n\n6. IMPRESSION:\n- Left femoral neck fracture, displaced, intracapsular\n- Pauwels Type III (vertical orientation)\n- Garden Stage III-IV displacement\n\n7. RECOMMENDATIONS:\n- Urgent orthopedic consultation\n- Consider hemiarthroplasty vs total hip arthroplasty given patient age and fracture pattern\n- Pre-operative medical optimization",
  "report_type": "radiology",
  "sections": {
    "EXAMINATION": "Left Hip Radiograph - AP and Lateral views",
    "CLINICAL INDICATION": "Fall from standing height, left hip pain",
    "COMPARISON": "No prior imaging available",
    "FINDINGS": "- There is a displaced intracapsular fracture of the left femoral neck...",
    "IMPRESSION": "- Left femoral neck fracture, displaced, intracapsular...",
    "RECOMMENDATIONS": "- Urgent orthopedic consultation..."
  },
  "usage": {
    "prompt_tokens": 385,
    "completion_tokens": 312,
    "total_tokens": 697
  },
  "model": "google/medgemma-4b-it",
  "created": 1733500800
}
```

### Example - Pathology Report

```bash
curl -X POST "https://your-runpod-id-8000.proxy.runpod.net/v1/medical/report" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: david" \
  -d '{
    "images": ["BASE64_PATHOLOGY_IMAGE"],
    "report_type": "pathology",
    "patient_info": "58-year-old male",
    "clinical_indication": "Colon mass identified on colonoscopy"
  }'
```

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "error_code",
    "message": "Human-readable error message",
    "type": "error_type",
    "param": "parameter_name"
  }
}
```

### Common Error Codes

| HTTP Status | Error Code | Description |
|-------------|------------|-------------|
| 400 | `invalid_request` | Malformed request or invalid parameters |
| 401 | `missing_api_key` | No API key provided |
| 403 | `invalid_api_key` | API key is invalid |
| 500 | `internal_error` | Server error during processing |
| 501 | `streaming_not_supported` | Streaming feature not available |
| 503 | `model_not_loaded` | Model is still loading |

### Example Error Response

```json
{
  "error": {
    "code": "invalid_request",
    "message": "At least one user message is required",
    "type": "invalid_request_error"
  }
}
```

### Rate Limiting

Currently no rate limits are enforced. However, requests are processed sequentially due to GPU constraints.

---

## Code Examples

### Python - Complete Example

```python
import base64
import requests
from pathlib import Path

# Configuration
API_URL = "https://your-runpod-id-8000.proxy.runpod.net"
API_KEY = "david"
HEADERS = {
    "Content-Type": "application/json",
    "X-API-Key": API_KEY
}

def check_health():
    """Check API health status."""
    response = requests.get(f"{API_URL}/health")
    return response.json()

def medical_query(query: str, max_tokens: int = 2048):
    """Send a text-only medical query."""
    payload = {
        "query": query,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "include_disclaimer": True
    }
    response = requests.post(
        f"{API_URL}/v1/medical/query",
        headers=HEADERS,
        json=payload
    )
    return response.json()

def analyze_image(image_path: str, query: str, modality: str = None):
    """Analyze a medical image."""
    # Read and encode image
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "images": [f"data:image/png;base64,{image_data}"],
        "query": query,
        "structured_output": True
    }
    if modality:
        payload["image_modality"] = modality

    response = requests.post(
        f"{API_URL}/v1/medical/analyze",
        headers=HEADERS,
        json=payload
    )
    return response.json()

def generate_report(image_path: str, report_type: str, indication: str):
    """Generate a medical report from an image."""
    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    payload = {
        "images": [image_data],
        "report_type": report_type,
        "clinical_indication": indication,
        "max_tokens": 4096
    }

    response = requests.post(
        f"{API_URL}/v1/medical/report",
        headers=HEADERS,
        json=payload
    )
    return response.json()

# Usage examples
if __name__ == "__main__":
    # Check health
    health = check_health()
    print(f"Status: {health['status']}")
    print(f"Model: {health['model_id']}")

    # Medical query
    result = medical_query(
        "What are the diagnostic criteria for rheumatoid arthritis?"
    )
    print(result["response"])

    # Image analysis (uncomment with valid image)
    # result = analyze_image("chest_xray.png", "Describe any abnormalities", "chest_xray")
    # print(result["analysis"])
```

### JavaScript/Node.js Example

```javascript
const axios = require('axios');
const fs = require('fs');

const API_URL = 'https://your-runpod-id-8000.proxy.runpod.net';
const API_KEY = 'david';

const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
    'X-API-Key': API_KEY
  }
});

// Health check
async function checkHealth() {
  const response = await axios.get(`${API_URL}/health`);
  return response.data;
}

// Medical query
async function medicalQuery(query, maxTokens = 2048) {
  const response = await api.post('/v1/medical/query', {
    query,
    max_tokens: maxTokens,
    temperature: 0.7
  });
  return response.data;
}

// Image analysis
async function analyzeImage(imagePath, query, modality = null) {
  const imageBuffer = fs.readFileSync(imagePath);
  const base64Image = imageBuffer.toString('base64');

  const payload = {
    images: [`data:image/png;base64,${base64Image}`],
    query,
    structured_output: true
  };

  if (modality) {
    payload.image_modality = modality;
  }

  const response = await api.post('/v1/medical/analyze', payload);
  return response.data;
}

// Chat completion (OpenAI-compatible)
async function chat(messages, maxTokens = 1024) {
  const response = await api.post('/v1/chat/completions', {
    messages,
    max_tokens: maxTokens,
    temperature: 0.7
  });
  return response.data;
}

// Usage
(async () => {
  // Health check
  const health = await checkHealth();
  console.log(`Status: ${health.status}`);

  // Chat completion
  const chatResult = await chat([
    { role: 'system', content: 'You are a medical assistant.' },
    { role: 'user', content: 'What is hypertension?' }
  ]);
  console.log(chatResult.choices[0].message.content);

  // Medical query
  const queryResult = await medicalQuery('What causes migraines?');
  console.log(queryResult.response);
})();
```

### cURL Examples Collection

```bash
# 1. Health Check
curl -X GET "https://your-runpod-id-8000.proxy.runpod.net/health"

# 2. Simple Chat
curl -X POST "https://your-runpod-id-8000.proxy.runpod.net/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: david" \
  -d '{
    "messages": [
      {"role": "user", "content": "What is the treatment for strep throat?"}
    ],
    "max_tokens": 1024
  }'

# 3. Medical Query with System Prompt
curl -X POST "https://your-runpod-id-8000.proxy.runpod.net/v1/medical/query" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: david" \
  -d '{
    "query": "A patient has an INR of 4.5 on warfarin. What should I do?",
    "system_prompt": "You are an anticoagulation pharmacist specialist.",
    "max_tokens": 1024
  }'

# 4. Image Analysis (replace BASE64_IMAGE with actual data)
curl -X POST "https://your-runpod-id-8000.proxy.runpod.net/v1/medical/analyze" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: david" \
  -d '{
    "images": ["BASE64_IMAGE_DATA"],
    "query": "Describe this medical image",
    "image_modality": "chest_xray"
  }'

# 5. Generate Radiology Report
curl -X POST "https://your-runpod-id-8000.proxy.runpod.net/v1/medical/report" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: david" \
  -d '{
    "images": ["BASE64_IMAGE_DATA"],
    "report_type": "radiology",
    "clinical_indication": "Chest pain, rule out pneumonia"
  }'
```

---

## Interactive Documentation

The API provides interactive documentation:

- **Swagger UI:** `https://your-runpod-id-8000.proxy.runpod.net/docs`
- **ReDoc:** `https://your-runpod-id-8000.proxy.runpod.net/redoc`
- **OpenAPI JSON:** `https://your-runpod-id-8000.proxy.runpod.net/openapi.json`

---

## Token Limits

| Parameter | Limit |
|-----------|-------|
| Max input tokens | 12,000 |
| Max output tokens | 8,192 |
| Default output tokens | 2,048 |
| Max message length | 100,000 characters |
| Max images per request | 4 |
| Max image size | 20 MB |

---

## Important Notes

1. **Medical Disclaimer:** All responses are AI-generated and should not replace professional medical advice.

2. **Model Loading:** The model takes 3-5 minutes to load on startup. Check `/health` endpoint for status.

3. **GPU Memory:** The API uses quantization to fit on 48GB VRAM (A6000). Complex requests with multiple large images may require more memory.

4. **Streaming:** Not yet supported. All responses are returned as complete JSON.

5. **Concurrency:** Requests are processed sequentially. High concurrency may result in queuing.

---

## Support

- **Interactive Docs:** `/docs` endpoint
- **Health Status:** `/health` endpoint
- **API Version:** 1.0.0
