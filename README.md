# fastapi-pydantic-ai-agent

Agentic AI framework built with FastAPI, Pydantic AI, and LlamaIndex Workflows. Type-safe agents, event-driven RAG, SSE streaming.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com/)
[![Pydantic AI](https://img.shields.io/badge/Pydantic_AI-1.70+-purple.svg)](https://ai.pydantic.dev/)

## 🎯 Overview

An internal developer framework that demonstrates three canonical AI agent patterns:

1. **Tool-Calling Agent** — Pydantic AI agent with typed dependency injection and conversation history
2. **Event-Driven RAG** — LlamaIndex Corrective RAG workflow with multi-step retrieval and evaluation
3. **SSE Streaming** — Server-Sent Events streaming for real-time LLM responses

Built as a starter kit for engineers to rapidly prototype AI agent APIs with zero lock-in — swap LLM providers, vector stores, and session backends via environment variables or dependency injection.

## ✨ Features

- 🔒 **Type-Safe**: Full Python 3.13+ type annotations with Pydantic models
- 🔌 **Pluggable**: Protocol-based interfaces for vector stores, session stores, and stream adapters
- 🌐 **Provider-Agnostic**: Switch between OpenAI, Anthropic, Ollama, or custom LLM providers via configuration
- 📊 **Observable**: Pydantic Logfire integration for automatic AI agent tracing and token usage tracking
- 🐳 **Production-Ready**: Multi-stage Docker build with uv for fast dependency management
- 🧪 **Test Suite**: Unit, integration, and E2E tests with 80%+ coverage

## 🚀 Quick Start

### Prerequisites

- **Python 3.13+**
- **[mise](https://mise.jdx.dev/)** (recommended) or **uv** directly
- **LLM Provider API Key** (OpenAI, Anthropic, etc.) or local Ollama installation

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd fastapi-pydantic-ai-agent

# Install dependencies with uv (via mise)
mise install
```

### 2. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your settings
# Required:
#   - API_KEY: Your API key for X-API-Key authentication
#   - LLM_MODEL: Model identifier (e.g., "openai:gpt-4o", "anthropic:claude-3-5-sonnet")
#   - LLM_API_KEY: Your LLM provider API key (or leave empty for Ollama)
```

**Example `.env` for OpenAI:**

```bash
API_KEY=my-secret-api-key
LLM_MODEL=openai:gpt-4o
LLM_API_KEY=sk-...your-openai-key...
```

**Example `.env` for Ollama (local):**

```bash
API_KEY=my-secret-api-key
LLM_MODEL=ollama:llama3
LLM_BASE_URL=http://localhost:11434/v1
# LLM_API_KEY not required for Ollama
```

### 3. Run Development Server

```bash
# Start FastAPI dev server with hot reload
mise run dev

# Server starts at http://localhost:8000
# OpenAPI docs: http://localhost:8000/docs
```

### 4. Verify Health Check

```bash
# Health check endpoint (no authentication required)
curl http://localhost:8000/health
```

**Response:**

```json
{
  "status": "healthy",
  "service": "fastapi-pydantic-ai-agent"
}
```

## 📚 API Examples

All `/v1/` endpoints require the `X-API-Key` header. Replace `your-api-key-here` with the value from your `.env` file.

### Pattern 1: Tool-Calling Agent (Chat)

#### Standard Request/Response

```bash
curl -X POST http://localhost:8000/v1/agent/chat \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "message": "What is the weather like today?",
    "session_id": "user-123"
  }'
```

**Response:**

```json
{
  "reply": "I checked the weather using my search tool. Currently it's 72°F and sunny.",
  "session_id": "user-123",
  "tool_calls_made": 1
}
```

**Features:**

- Conversation history maintained per `session_id`
- Agent can call registered tools (e.g., `mock_web_search`)
- Tool invocations are counted and returned
- Type-safe dependency injection via [`AgentDeps`](app/agents/deps.py)

### Pattern 2: Event-Driven RAG

#### Step 1: Ingest Documents

```bash
curl -X POST http://localhost:8000/v1/rag/ingest \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "chunks": [
      "FastAPI is a modern web framework for building APIs with Python 3.13+.",
      "Pydantic AI provides type-safe agent definitions with structured outputs.",
      "LlamaIndex Workflows enable event-driven orchestration of LLM calls."
    ]
  }'
```

**Response:**

```json
{
  "ingested": 3
}
```

#### Step 2: Query with Corrective RAG

```bash
curl -X POST http://localhost:8000/v1/rag/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -d '{
    "query": "What is Pydantic AI used for?",
    "max_retries": 3
  }'
```

**Response:**

```json
{
  "answer": "Pydantic AI provides type-safe agent definitions with structured outputs, enabling developers to build AI agents with guaranteed output schemas.",
  "context_found": true,
  "search_count": 1
}
```

**Workflow Steps:**

1. **Search**: Retrieve top-k chunks from vector store
2. **Evaluate**: LLM assesses relevance; refines query if insufficient
3. **Synthesize**: Generate final answer from relevant context

See [`CorrectiveRAGWorkflow`](app/workflows/corrective_rag.py) for implementation.

### Pattern 3: SSE Streaming

```bash
curl -X POST http://localhost:8000/v1/agent/stream \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-api-key-here" \
  -N \
  -d '{
    "message": "Write a haiku about Python",
    "session_id": "user-123"
  }'
```

**Response Stream:**

```
 {"type": "delta", "content": "Code"}
data: {"type": "delta", "content": " flows"}
 {"type": "delta", "content": " like"}
 {"type": "delta", "content": " water,\n"}
 {"type": "delta", "content": "Types"}
 {"type": "delta", "content": " guide"}
 {"type": "delta", "content": " the"}
 {"type": "delta", "content": " way"}
 {"type": "delta", "content": " forward"}
 {"type": "done", "content": ""}
```

**Features:**

- Server-Sent Events (SSE) with `text/event-stream` media type
- Token-by-token streaming via Pydantic AI's [`run_stream()`](app/agents/chat_agent.py)
- Pluggable adapters (extend [`StreamAdapter`](app/api/v1/agent.py) Protocol for Vercel AI or AG-UI formats)
- Conversation history saved after stream completes

## 🏗️ Project Structure

```
app/
├── main.py                     # FastAPI app factory, lifespan, global middleware
├── config.py                   # Pydantic Settings (all env vars)
├── observability.py            # Logfire initialization helpers
│
├── api/                        # FastAPI route handlers
│   ├── health.py               # GET /health
│   └── v1/
│       ├── router.py           # Aggregates v1 sub-routers
│       ├── agent.py            # POST /v1/agent/chat, /v1/agent/stream
│       └── rag.py              # POST /v1/rag/query, /v1/rag/ingest
│
├── agents/                     # Pydantic AI agent layer
│   ├── deps.py                 # AgentDeps dataclass, dependency factories
│   └── chat_agent.py           # Agent definition and registered tools
│
├── workflows/                  # LlamaIndex Workflow layer
│   ├── events.py               # Event classes: SearchEvent, EvaluateEvent
│   ├── state.py                # WorkflowState Pydantic model
│   └── corrective_rag.py       # CorrectiveRAGWorkflow (steps)
│
├── models/                     # Request/Response Pydantic schemas
│   ├── agent.py                # ChatRequest, ChatResponse, StreamEvent
│   ├── rag.py                  # RAGQueryRequest, RAGQueryResponse, IngestRequest
│   └── errors.py               # ErrorResponse, structured error models
│
├── deps/                       # FastAPI dependency functions
│   ├── auth.py                 # api_key_header dependency (X-API-Key)
│   └── workflow.py             # get_rag_workflow — per-request factory
│
└── stores/                     # Pluggable store implementations
    ├── vector_store.py         # VectorStore Protocol + InMemoryVectorStore
    └── session_store.py        # SessionStore Protocol + InMemorySessionStore
```

## 🧪 Testing

```bash
# Run all tests with coverage
mise run test

# Run specific test suites
mise run test:unit          # Fast unit tests (no LLM calls)
mise run test:integration   # Integration tests (real stores, mocked LLM)
mise run test:e2e           # End-to-end HTTP tests

# Run linter and type checker
mise run lint

# Format code
mise run format
```

**Test Layers:**

- **Unit** (`tests/unit/`) — Isolated component tests with no I/O
- **Integration** (`tests/integration/`) — Multi-component tests with `FunctionModel` LLM
- **E2E** (`tests/e2e/`) — Full HTTP stack with `AsyncClient`

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
mise run build
# Or: docker build -t fastapi-pydantic-ai-agent:latest .

# Run container
docker run -p 8000:8000 \
  -e API_KEY=your-api-key \
  -e LLM_MODEL=openai:gpt-4o \
  -e LLM_API_KEY=sk-... \
  fastapi-pydantic-ai-agent:latest
```

### Docker Compose Example

```yaml
version: '3.8'
services:
  api:
    build: .
    ports:
      - '8000:8000'
    environment:
      - API_KEY=${API_KEY}
      - LLM_MODEL=${LLM_MODEL}
      - LLM_API_KEY=${LLM_API_KEY}
      - LOGFIRE_TOKEN=${LOGFIRE_TOKEN}
    restart: unless-stopped
```

## 🔧 Configuration

All configuration is managed via environment variables or `.env` file. See [`.env.example`](.env.example) for complete reference.

### Required Variables

| Variable    | Description            | Example                                        |
| ----------- | ---------------------- | ---------------------------------------------- |
| `API_KEY`   | X-API-Key header value | `my-secret-key`                                |
| `LLM_MODEL` | LLM provider and model | `openai:gpt-4o`, `anthropic:claude-3-5-sonnet` |

### Optional Variables

| Variable               | Default                     | Description                                                    |
| ---------------------- | --------------------------- | -------------------------------------------------------------- |
| `LLM_API_KEY`          | `None`                      | Provider API key (not required for Ollama)                     |
| `LLM_BASE_URL`         | `None`                      | Custom endpoint (e.g., `http://localhost:11434/v1` for Ollama) |
| `MAX_OUTPUT_RETRIES`   | `3`                         | Pydantic AI output validation retries                          |
| `LOGFIRE_TOKEN`        | `None`                      | Pydantic Logfire observability token                           |
| `LOGFIRE_SERVICE_NAME` | `fastapi-pydantic-ai-agent` | Service name for traces                                        |

## 🔌 Extension Points

The framework is designed for extensibility via Protocol-based interfaces:

### Custom Vector Store

Implement [`VectorStore`](app/stores/vector_store.py) Protocol:

```python
from app.stores.vector_store import VectorStore

class ChromaVectorStore:
    async def add_documents(self, chunks: list[str]) -> None: ...
    async def query(self, query: str, top_k: int = 5) -> list[str]: ...
    async def clear(self) -> None: ...

# Replace in app/main.py lifespan
app.state.vector_store = ChromaVectorStore()
```

### Custom Session Store

Implement [`SessionStore`](app/stores/session_store.py) Protocol:

```python
from app.stores.session_store import SessionStore
from pydantic_ai.messages import ModelMessage

class RedisSessionStore:
    async def get_history(self, session_id: str) -> list[ModelMessage]: ...
    async def save_history(self, session_id: str, messages: list[ModelMessage]) -> None: ...
    async def clear(self, session_id: str) -> None: ...

# Replace in app/main.py lifespan
app.state.session_store = RedisSessionStore()
```

### Custom Stream Adapter

Implement [`StreamAdapter`](app/api/v1/agent.py) Protocol:

```python
class VercelAIAdapter:
    def format_event(self, event_type: str, content: str) -> str:
        # Vercel AI SDK format
        return f"0:{json.dumps({'type': event_type, 'content': content})}\n"

    def format_done(self) -> str:
        return "d:{}\n"

    def format_error(self, message: str) -> str:
        return f"3:{json.dumps({'error': message})}\n"

# Use in route handler
@router.post("/agent/stream")
async def stream_agent(...):
    adapter = VercelAIAdapter()  # Replace DefaultSSEAdapter
    ...
```

## 🏛️ Architecture

### Core Concepts

**Type Safety First**

- All public interfaces use Python 3.13+ type annotations
- Pydantic models for configuration, requests, responses, and workflow state
- No use of `Any` — strict typing throughout

**Protocol-Based Extensibility**

- `VectorStore` Protocol — swap TF-IDF for Chroma, pgvector, or custom backends
- `SessionStore` Protocol — replace in-memory with Redis or database-backed stores
- `StreamAdapter` Protocol — support multiple SSE formats without changing agent logic

**Zero Lock-In Design**

- LLM provider configured entirely via environment variables (`LLM_MODEL`, `LLM_API_KEY`, `LLM_BASE_URL`)
- Switch between OpenAI, Anthropic, Ollama, or custom providers with no code changes
- Dependency injection via FastAPI and Pydantic AI `RunContext`

### Data Flow

```
┌─────────────┐
│   Client    │
└──────┬──────┘
       │ HTTP Request (X-API-Key)
       ▼
┌─────────────────────────────────────┐
│      FastAPI Application            │
│  ┌──────────────────────────────┐  │
│  │  Authentication Middleware    │  │
│  └──────────┬───────────────────┘  │
│             ▼                        │
│  ┌──────────────────────────────┐  │
│  │    Route Handler             │  │
│  │  • /v1/agent/chat            │  │
│  │  • /v1/agent/stream          │  │
│  │  • /v1/rag/ingest            │  │
│  │  • /v1/rag/query             │  │
│  └──────────┬───────────────────┘  │
│             ▼                        │
│  ┌──────────────────────────────┐  │
│  │   Pydantic AI Agent          │  │
│  │   (with RunContext[Deps])    │  │
│  └──────────┬───────────────────┘  │
│             │                        │
│             ├──────────────────────►│ SessionStore
│             │                        │ (conversation history)
│             │                        │
│             └──────────────────────►│ Tools
│                                      │ (mock_web_search, etc.)
└──────────────────────────────────────┘
       │
       ▼
┌─────────────────────────────────────┐
│   LlamaIndex Workflow               │
│  ┌──────────────────────────────┐  │
│  │  CorrectiveRAGWorkflow        │  │
│  │  ┌─────┐   ┌──────┐  ┌──────┐│  │
│  │  │Search├──►Evaluate├►Synthesize
│  │  └─────┘   └──────┘  └──────┘│  │
│  └──────────┬───────────────────┘  │
│             │                        │
│             └──────────────────────►│ VectorStore
│                                      │ (TF-IDF retrieval)
└──────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│  LLM Provider│
│ (OpenAI/etc) │
└──────────────┘
```

### Observability

Every component is instrumented with Pydantic Logfire:

- **HTTP Requests** — Automatic FastAPI span per request
- **Agent Runs** — Token usage, cost, tool calls, validation retries
- **Workflow Steps** — Per-step spans showing latency and events
- **Custom Spans** — Add `logfire.span()` context managers anywhere

## 📖 Learning Resources

### Official Documentation

- [FastAPI](https://fastapi.tiangolo.com/) — Modern Python web framework
- [Pydantic AI](https://ai.pydantic.dev/) — Type-safe AI agent framework
- [LlamaIndex Workflows](https://docs.llamaindex.ai/en/stable/module_guides/workflow/) — Event-driven LLM orchestration
- [Pydantic Logfire](https://logfire.pydantic.dev/) — AI-native observability

### Key Patterns

**Corrective RAG**

- [Corrective RAG Paper](https://arxiv.org/abs/2401.15884) — Self-reflective retrieval with relevance evaluation

**Tool-Calling Agents**

- [Pydantic AI Tools](https://ai.pydantic.dev/tools/) — Type-safe function calling with dependency injection

**SSE Streaming**

- [Server-Sent Events Spec](https://html.spec.whatwg.org/multipage/server-sent-events.html) — Standard SSE protocol

## 🤝 Contributing

This is an internal starter kit. When extending for your use case:

1. Fork and customize for your domain
2. Replace in-memory stores with production backends
3. Add domain-specific tools to the agent
4. Extend the RAG workflow with custom evaluation logic
5. Implement custom stream adapters for your frontend

## 📄 License

See [LICENSE](LICENSE) file for details.

## 🔗 Related Projects

- [LangChain](https://github.com/langchain-ai/langchain) — Alternative agent framework
- [LlamaIndex](https://github.com/run-llama/llama_index) — Data framework for LLM apps
- [AG2](https://github.com/ag2ai/ag2) — Multi-agent conversation framework

---

**Built with ❤️ using FastAPI, Pydantic AI, and LlamaIndex Workflows**
