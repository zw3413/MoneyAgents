# Trading Assistant

A cloud-native multi-agent stock analysis system powered by LangChain and various LLMs.

## Project Structure

```
TradingAsst/
├── trading_asst/            # Main package directory
│   ├── api/                # API layer
│   │   ├── routes/        # API route handlers
│   │   ├── models/        # API models/schemas
│   │   └── dependencies.py # FastAPI dependencies
│   ├── core/              # Core business logic
│   │   ├── agents/       # Trading analysis agents
│   │   ├── services/     # Business services
│   │   └── config.py     # Core configuration
│   ├── infrastructure/    # Infrastructure layer
│   │   ├── database/     # Database connections
│   │   ├── cloud/        # Cloud service integrations
│   │   └── cache/       # Caching mechanisms
│   └── utils/            # Utility functions
├── deploy/                # Deployment configurations
│   ├── kubernetes/       # K8s manifests
│   ├── terraform/        # IaC definitions
│   └── docker/          # Docker configurations
├── tests/                # Test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── e2e/            # End-to-end tests
├── docs/                 # Documentation
│   ├── api/             # API documentation
│   ├── architecture/    # Architecture diagrams
│   └── deployment/      # Deployment guides
├── scripts/             # Utility scripts
├── .github/             # GitHub Actions workflows
├── Dockerfile          # Main Dockerfile
├── docker-compose.yml  # Local development setup
├── pyproject.toml      # Project configuration
├── .env.example       # Environment variables template
└── README.md          # Project documentation
```

## Development Setup

1. Install uv (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Create and activate a virtual environment:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
# Install all dependencies (including development tools)
uv pip install -e ".[dev]"

# Or install only production dependencies
uv pip install -e .
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

5. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

## Running the Application

You can start the application using uvicorn in one of two ways:

1. From the project root directory (recommended):
```bash
uvicorn trading_asst.api.main:app --reload
```

2. Using Python module syntax:
```bash
python -m uvicorn trading_asst.api.main:app --reload
```

The `--reload` flag enables hot reloading during development. Remove it in production.

Once running, you can access:
- API documentation: http://localhost:8000/docs
- Alternative API docs: http://localhost:8000/redoc

## Local Development

### Prerequisites

- Docker and Docker Compose
- Python 3.11+
- OpenAI API key

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/TradingAsst.git
cd TradingAsst
```

2. Create and configure environment variables:
```bash
cp .env.example .env
# Edit .env file with your OpenAI API key and other settings
```

3. Start the service:
```bash
# Build and start containers
docker-compose up --build -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

4. Access the API:
- API endpoint: http://localhost:8080
- Health check: http://localhost:8080/
- API documentation: http://localhost:8080/docs

### Development Notes

- The service uses FastAPI for the REST API
- LangChain is used for the multi-agent system
- Docker Compose is configured for local development with hot reload
- Environment variables are managed through .env file

### Available Endpoints

- `GET /`: Health check endpoint
- `POST /api/v1/analyze`: Full stock analysis
- `POST /api/v1/quick-analysis`: Quick technical and news analysis

### Example Usage

```bash
# Quick Analysis
curl -X POST http://localhost:8080/api/v1/quick-analysis \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'

# Full Analysis
curl -X POST http://localhost:8080/api/v1/analyze \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL", "timeframe": "1y"}'
```

## Deployment

### Local Docker Build
```bash
docker build -t trading-assistant:latest .
```

### Cloud Deployment
See [deployment documentation](docs/deployment/README.md) for detailed instructions on:
- GCP Cloud Run deployment
- Kubernetes deployment
- Infrastructure as Code setup

## API Documentation

Once running, API documentation is available at:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests and linters
5. Submit a pull request

## License

MIT License - see LICENSE file for details

# start
## Project Proposal on google doc:
- https://docs.google.com/document/d/1MdKpiF53Z4Lo1Ee7T09S6hioN0wutwxoW0Njd97uCxw/edit?usp=sharing



## References:

- https://tradingagents-ai.github.io/
- https://github.com/charliedream1/ai_quant_trade
- https://arxiv.org/pdf/2411.08804
- https://arxiv.org/pdf/2306.06031

## Other resources:
- https://aws.amazon.com/cn/blogs/china/build-an-ai-stock-analysis-assistant-in-3-low-code-steps-using-amazon-bedrock/
- https://github.com/charliedream1/ai_quant_trade




This multi-agents system, employing LangChain as the base framework, Orchestrating agents to develivery functions of stock analysis. Including modeules:
1. User Interface, we use a chrome extension to add an layer on the tradingview UI
2. Backend Service, using Cloud Run/ App Engine container: implemented with LangChain, receiveing request and orchestrate other agents to process the request.  Stateless services, only response apon input and LLM service.
3. Datasource and tools, backend can visit outside data via encapsulated functions or tools, including yfinance api, history data api, news api, social media data api, and OCR etc.
4. 