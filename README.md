# LLM Document Processing System

An intelligent query-retrieval system that processes insurance policy documents and answers natural language queries with structured responses. The system uses OpenAI's GPT models and FAISS for semantic search to provide accurate, explainable decisions for insurance coverage queries.

## Features

- **Natural Language Processing**: Understands queries like "46M, knee surgery, Pune, 3-month policy"
- **Semantic Search**: Uses OpenAI embeddings and FAISS for intelligent document retrieval
- **Structured Responses**: Returns detailed JSON responses with decisions, justifications, and amounts
- **Multi-format Support**: Handles PDF documents with section preservation
- **Batch Processing**: Process multiple queries efficiently
- **RESTful API**: FastAPI-based API with comprehensive endpoints
- **Local Vector Database**: FAISS for fast similarity search without external dependencies

## Tech Stack

- **Backend**: FastAPI
- **LLM**: OpenAI GPT-4/GPT-3.5-turbo
- **Vector Database**: FAISS (local)
- **Document Processing**: PyPDF2, pdfplumber
- **Embeddings**: OpenAI text-embedding-ada-002
- **Framework**: LangChain
- **Language**: Python 3.9+

## Project Structure

```
document_processor/
├── main.py                 # FastAPI application entry point
├── config.py              # Configuration and environment variables
├── requirements.txt       # Dependencies
├── README.md             # This file
├── data/
│   ├── policies/          # PDF documents directory
│   └── processed/         # Processed documents cache
├── src/
│   ├── __init__.py
│   ├── document_loader.py # PDF processing and text extraction
│   ├── embeddings.py      # Vector embeddings and FAISS index
│   ├── query_processor.py # Query parsing and structuring
│   ├── retrieval.py       # Semantic search and clause matching
│   ├── decision_engine.py # Logic evaluation and decision making
│   └── response_formatter.py # JSON response formatting
└── tests/
    └── test_queries.py    # Sample queries for testing
```

## Quick Start

### 1. Environment Setup

```bash
# Clone the repository
git clone <repository-url>
cd document_processor

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Create a `.env` file in the root directory:

```env
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-4
EMBEDDING_MODEL=text-embedding-ada-002
FAISS_INDEX_PATH=./data/faiss_index
DOCUMENTS_PATH=./data/policies
PROCESSED_DOCS_PATH=./data/processed
LOG_LEVEL=INFO
MAX_TOKENS=4000
TEMPERATURE=0.1
```

### 3. Run the Application

```bash
# Start the FastAPI server
python main.py

# Or using uvicorn directly
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 4. API Documentation

Once running, visit:
- **Interactive API Docs**: http://localhost:8000/docs
- **Alternative API Docs**: http://localhost:8000/redoc

## API Endpoints

### Core Endpoints

#### `POST /query`
Process a single natural language query.

**Request:**
```json
{
  "query": "46M, knee surgery, Pune, 3-month policy",
  "document_path": "optional_specific_document.pdf"
}
```

**Response:**
```json
{
  "query": "46M, knee surgery, Pune, 3-month policy",
  "decision": "covered",
  "confidence": 0.85,
  "amount": {
    "covered_amount": 50000,
    "patient_responsibility": 5000,
    "currency": "INR"
  },
  "justification": {
    "reasoning": "Knee surgery is covered under in-patient hospitalization treatment...",
    "applicable_clauses": [
      {
        "clause_id": "C.1.1",
        "section": "In-patient Hospitalization Treatment",
        "text": "If You are advised Hospitalization within India...",
        "relevance_score": 0.92
      }
    ],
    "conditions": [
      "24-hour hospitalization required",
      "Treatment at network hospital recommended"
    ],
    "exclusions_checked": [
      "Pre-existing conditions: Not applicable",
      "Waiting periods: Satisfied"
    ]
  },
  "metadata": {
    "processing_time": "2.3s",
    "tokens_used": 1250,
    "document_source": "bajaj_allianz_policy.pdf"
  }
}
```

#### `POST /batch-query`
Process multiple queries in batch.

**Request:**
```json
{
  "queries": [
    "46M, knee surgery, Pune, 3-month policy",
    "Does this policy cover maternity expenses?",
    "What is the waiting period for pre-existing diseases?"
  ]
}
```

### Document Management

#### `POST /upload-document`
Upload and process a new PDF policy document.

#### `GET /documents`
List all processed documents.

#### `POST /reindex`
Reindex all documents in the documents directory.

#### `DELETE /documents/{document_name}`
Delete a specific document from the index.

### System Information

#### `GET /health`
Health check endpoint.

#### `GET /stats`
Get system statistics.

## Usage Examples

### Python Client Example

```python
import requests
import json

# Base URL
base_url = "http://localhost:8000"

# Single query
def process_query(query):
    response = requests.post(f"{base_url}/query", json={"query": query})
    return response.json()

# Example usage
result = process_query("46M, knee surgery, Pune, 3-month policy")
print(json.dumps(result, indent=2))

# Batch processing
def process_batch_queries(queries):
    response = requests.post(f"{base_url}/batch-query", json={"queries": queries})
    return response.json()

# Upload document
def upload_document(file_path):
    with open(file_path, 'rb') as f:
        files = {'file': f}
        response = requests.post(f"{base_url}/upload-document", files=files)
    return response.json()
```

### cURL Examples

```bash
# Process a query
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{"query": "46M, knee surgery, Pune, 3-month policy"}'

# Upload a document
curl -X POST "http://localhost:8000/upload-document" \
  -F "file=@policy_document.pdf"

# Get system health
curl -X GET "http://localhost:8000/health"

# Get system stats
curl -X GET "http://localhost:8000/stats"
```

## Testing

### Test Queries

The system includes comprehensive test queries in `tests/test_queries.py`:

```python
from tests.test_queries import test_queries, get_test_queries_by_category

# Get all test queries
all_queries = test_queries

# Get queries by category
coverage_queries = get_test_queries_by_category("coverage_check")
amount_queries = get_test_queries_by_category("amount_inquiry")
```

### Running Tests

```bash
# Test single query
python -c "
import requests
response = requests.post('http://localhost:8000/query', 
                        json={'query': '46M, knee surgery, Pune, 3-month policy'})
print(response.json())
"

# Test batch processing
python -c "
import requests
from tests.test_queries import test_queries[:5]
response = requests.post('http://localhost:8000/batch-query', 
                        json={'queries': test_queries[:5]})
print(response.json())
"
```

## Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENAI_API_KEY` | Your OpenAI API key | Required |
| `OPENAI_MODEL` | GPT model to use | `gpt-4` |
| `EMBEDDING_MODEL` | Embedding model | `text-embedding-ada-002` |
| `FAISS_INDEX_PATH` | Path for FAISS index | `./data/faiss_index` |
| `DOCUMENTS_PATH` | PDF documents directory | `./data/policies` |
| `PROCESSED_DOCS_PATH` | Processed documents cache | `./data/processed` |
| `LOG_LEVEL` | Logging level | `INFO` |
| `MAX_TOKENS` | Maximum tokens for LLM | `4000` |
| `TEMPERATURE` | LLM temperature | `0.1` |

### Processing Settings

| Setting | Description | Default |
|---------|-------------|---------|
| `CHUNK_SIZE` | Document chunk size | `1000` |
| `CHUNK_OVERLAP` | Chunk overlap | `200` |
| `TOP_K_RESULTS` | Number of top results | `5` |
| `SIMILARITY_THRESHOLD` | Similarity threshold | `0.7` |

## System Architecture

### Core Components

1. **Document Loader**: Processes PDF documents and extracts structured text
2. **Embeddings Manager**: Creates and manages vector embeddings using OpenAI
3. **Query Processor**: Parses natural language queries and extracts entities
4. **Retrieval System**: Performs semantic search using FAISS
5. **Decision Engine**: Evaluates policy logic and makes coverage decisions
6. **Response Formatter**: Structures final JSON responses

### Processing Pipeline

1. **Document Processing**: PDF → Text → Chunks → Embeddings → FAISS Index
2. **Query Processing**: Natural Language → Entities → Search Queries
3. **Retrieval**: Semantic Search → Relevant Clauses
4. **Decision Making**: Clause Analysis → Coverage Decision → Amount Calculation
5. **Response Formatting**: Structured JSON with justification

## Performance Considerations

### Optimization Tips

1. **Batch Processing**: Use `/batch-query` for multiple queries
2. **Index Caching**: FAISS index is automatically cached
3. **Document Chunking**: Optimize chunk size for your documents
4. **API Rate Limits**: Monitor OpenAI API usage

### Monitoring

- Use `/health` endpoint for system status
- Use `/stats` endpoint for performance metrics
- Monitor processing times in response metadata

## Error Handling

The system includes comprehensive error handling:

- **API Errors**: Proper HTTP status codes and error messages
- **Document Processing**: Graceful handling of corrupted PDFs
- **LLM Errors**: Fallback mechanisms for API failures
- **Validation**: Input validation and sanitization

## Security Considerations

- **API Key Management**: Secure storage of OpenAI API keys
- **Input Validation**: All inputs are validated and sanitized
- **Rate Limiting**: Consider implementing rate limiting for production
- **Data Privacy**: Local processing of sensitive documents

## Deployment

### Local Development

```bash
# Development mode with auto-reload
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production Deployment

```bash
# Production mode
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

### Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## Troubleshooting

### Common Issues

1. **OpenAI API Key Error**: Ensure your API key is set in `.env`
2. **No Documents Found**: Upload PDF documents to `data/policies/`
3. **Index Not Found**: Run `/reindex` endpoint to rebuild index
4. **Memory Issues**: Reduce chunk size or use smaller documents

### Logs

Check application logs for detailed error information:

```bash
# Set log level in .env
LOG_LEVEL=DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License.

## Support

For issues and questions:
1. Check the troubleshooting section
2. Review the API documentation at `/docs`
3. Check the logs for error details
4. Open an issue on GitHub

## Roadmap

- [ ] Support for more document formats (DOCX, TXT)
- [ ] Advanced caching mechanisms
- [ ] Multi-language support
- [ ] Real-time document updates
- [ ] Advanced analytics dashboard
- [ ] Integration with external policy databases 