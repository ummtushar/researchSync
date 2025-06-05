version: '3.8'
services:
  ingestion:
    build: ./ingestion_service
    environment:
      - NOTION_TOKEN=${NOTION_TOKEN}
      - NOTION_DATABASE_ID=${NOTION_DATABASE_ID}
      - X_USERS=${X_USERS}
      - REDIS_URL=redis://redis:6379/0
    depends_on:
      - redis
      - chroma
  mcp:
    build: ./mcp_service
    ports:
      - "8000:8000"
    environment:
      - NOTION_TOKEN=${NOTION_TOKEN}
      - NOTION_DATABASE_ID=${NOTION_DATABASE_ID}
  dashboard:
    build: ./dashboard_service
    ports:
      - "7860:7860"
    depends_on:
      - mcp
  redis:
    image: redis:latest
    ports:
      - "6379:6379"
  chroma:
    image: chromadb/chroma:latest
    ports:
      - "8001:8000"