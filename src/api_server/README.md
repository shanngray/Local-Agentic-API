## Usage:
1. Start the server:
```Bash
-m src.api_server.server
```

2. Access the APIs:
```
Simple React Agent: http://127.0.0.1:8000/simple-react/query
Franken Agent: http://127.0.0.1:8001/franken/query
```

3. OpenAPI documentation will be available at:
```
http://127.0.0.1:8000/docs
http://127.0.0.1:8001/docs
```

## Notes:
- The server uses async/await for concurrent operation
- Each agent has its own router and endpoint
- Configuration is centralized in config.py
- Input/output models are defined using Pydantic
- Error handling is implemented at the route level
- OpenAPI documentation is automatically generated