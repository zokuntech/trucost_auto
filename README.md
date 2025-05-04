# TruCost Auto Backend

Backend service for the car repair audit platform.

## Setup

1. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your configuration:
```env
SECRET_KEY=your-secret-key-here
```

4. Run the application:
```bash
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

## Project Structure

```
├── api/                # API endpoints
├── agents/            # Smart agent microservices
│   ├── quote_parser/
│   └── parts_research/
├── services/          # Supporting services
├── models/            # Data models
├── utils/             # Utility functions
├── main.py           # Main FastAPI application
├── config.py         # Configuration
├── requirements.txt  # Dependencies
└── .env.example     # Example environment variables
```

## API Documentation

Once the server is running, visit:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc` 