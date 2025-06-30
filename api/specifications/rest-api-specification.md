# Fraud Detection API Specification

Version: 0.1.1  
Content-Type: `application/json`

## Overview

REST API for real-time SCO fraud detection.

## Endpoints

### Health Check

GET `/`

Returns API health status and version information.

#### Example Response (200 OK):
```json
{
  "status": "healthy",
  "version": "1.2.3"
}
```

### Fraud Prediction

POST `/fraud-prediction`

Analyzes a transaction for potential fraud.

#### Example Request Body:
```json
{
  "transaction_header": {
    "store_id": "6ba7b810-9dad-11d1-80b4-00c04fd430c8",
    "cash_desk": 1,
    "transaction_start": "2023-09-13T17:29:51",
    "transaction_end": "2023-09-13T17:31:56",
    "total_amount": 29.90,
    "payment_medium": "CREDIT_CARD",
    "customer_feedback": 4.5
  },
  "transaction_lines": [
    {
      "id": 1,
      "product_id": "87654321-4321-8765-cbaa-210987654321",
      "timestamp": "2023-09-13T17:30:00",
      "pieces_or_weight": 2.0,
      "sales_price": 15.50,
      "was_voided": false,
      "camera_product_similar": true,
      "camera_certainty": 0.95
    },
    {
      "id": 2,
      ...
    }
  ]
}
```

#### Example Response (200 OK): No Fraud Detected
```json
{
  "version": "1.2.3",
  "is_fraud": false,
  "fraud_proba": 0.125,
  "estimated_damage": null,
  "explanation": null
}
```

#### Example Response (200 OK): Fraud Detected:
```json
{
  "version": "1.2.3",
  "is_fraud": true,
  "fraud_proba": 0.875,
  "estimated_damage": 26.16,
  "explanation": {
    "human_readable_reason": "Low camera certainty on products; High transaction amount",
    "offending_products": ["PERSONAL CARE"]
  }
}
```

## Data Models

Non-required fields in request and response models may be null or omitted entirely.

### Request
#### TransactionHeader
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `store_id` | UUID | Yes | Unique store identifier |
| `cash_desk` | integer | Yes | Cash register number |
| `transaction_start` | datetime | Yes | Transaction start time |
| `transaction_end` | datetime | Yes | Transaction end time  |
| `total_amount` | float | Yes | Total transaction amount |
| `payment_medium` | string | Yes | Payment method |
| `customer_feedback` | integer | No | Customer satisfaction rating |

#### TransactionLine
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `id` | integer | Yes | Line item number |
| `product_id` | UUID | Yes | Product identifier |
| `timestamp` | datetime | Yes | Scan time |
| `pieces_or_weight` | float | Yes | Quantity |
| `sales_price` | float | Yes | Total price |
| `was_voided` | bool | Yes | Whether line was voided |
| `camera_product_similar` | bool | Yes | Camera product match |
| `camera_certainty` | float | Yes | Camera confidence score |

### Response
#### FraudPrediction
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `version` | string | Yes | API version (semantic versioning) |
| `is_fraud` | bool | Yes | Fraud prediction result |
| `fraud_proba` | float | No | Fraud probability (0.0-1.0) |
| `estimated_damage` | float | No | Estimated financial impact in EUR |
| `explanation` | dict | No | Fraud explanation (if detected) |

#### Explanation
| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `human_readable_reason` | string | No | Human-readable explanation for fraud |
| `offending_products` | string[] | No | Products flagged as suspicious or missing (category names) |


## Additional Type Constraints

- Timestamps: ISO 8601 format (UTC, seconds, no time zone designator)
- Versions: Follow semantic versioning (x.y.z)
- Fraud probability: Between 0.0 and 1.0
- `human_readable_reason`, `offending_products`: Intended for store staff to read so basket checks can be done quicker.


## Performance Requirements

- Response time: ≤ 1.5 seconds
