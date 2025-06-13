# reasearchSync

researchSync offers two services:

1. **FastAPI-MCP Server**: Connects to MCP Hosts like Claude Desktop to retrieve papers' metadata from arXiv and store it in a Notion database. You can then query this database to find the best paper you need.
  
2. **Web App**: Powered by Gradio (with plans to migrate to React in the future) for the frontend and FastAPI for the backend. It uses a Notion and Redis database to store metadata of the papers from arXiv and vector embeddings for retrieving the best papers based on user queries.

## Python Version
- **Python**: 3.11

## System Overview

The system will:

- Monitor arXiv publications from ML/AI related tags (e.g., 'cs.AI').
- Fetch metadata (title, authors, abstract) and add to Notion Database using Prefect workflow.
- Store data in a Notion database.
- Convert the paper's abstract to vector embeddings for fast retrieval and store on Redis.
- Allow an AI assistant to query or manage the database via MCP.

## Prefect Workflow Snippet

![prefect-workflow.png](prefect-workflow.png)



