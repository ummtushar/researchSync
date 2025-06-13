import os
import json
import asyncio
from typing import List, Dict, Optional, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import redis
import numpy as np
from notion_client import Client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import uvicorn

# Load environment variables from parent directory
load_dotenv("../.env")

class MCPRequest(BaseModel):
    tool: str
    params: Dict[str, Any]

class MCPResponse(BaseModel):
    result: Any
    error: Optional[str] = None

class ResearchPapersMCP:
    """MCP Server for Research Papers using Redis Vector Database and Notion"""
    
    def __init__(self):
        # Initialize Notion client
        self.notion_token = 'ntn_I8446634157a83A61IttkJePdbRglYfIl1xrf621koSgmm'
        self.database_id = '2041f77568ad80389d49fd3a1610c7c5'
        self.notion = Client(auth=self.notion_token)
        
        # Initialize Redis client
        self.redis_client = redis.Redis(
            host='localhost', 
            port=6379, 
            db=0,
            decode_responses=False
        )
        
        # Initialize embedding model
        print("🧠 Loading embedding model...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.vector_set_name = "research_papers_vectors"
        
        # Test connections
        self._test_connections()
    
    def _test_connections(self):
        """Test Redis and Notion connections"""
        try:
            self.redis_client.ping()
            print("✅ Redis connection successful")
        except Exception as e:
            print(f"❌ Redis connection failed: {e}")
            print("💡 Start Redis with: brew services start redis")
        
        try:
            # Test Notion connection
            self.notion.databases.retrieve(self.database_id)
            print("✅ Notion connection successful")
        except Exception as e:
            print(f"❌ Notion connection failed: {e}")
    
    def search_papers(self, keyword: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Search for papers similar to the keyword using Redis vector database.
        
        Args:
            keyword: Search query
            top_k: Number of results to return
            
        Returns:
            Dictionary with papers list or error
        """
        try:
            print(f"🔍 Searching for: '{keyword}'")
            
            # Create embedding for query
            query_embedding = self.model.encode([keyword])[0]
            
            # Get all stored vectors from Redis
            pattern = f"{self.vector_set_name}:*"
            keys = self.redis_client.keys(pattern)
            
            if not keys:
                return {"papers": [], "message": "No papers found in database"}
            
            similarities = []
            
            for key in keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
                if key_str.endswith(':metadata'):
                    continue
                    
                try:
                    # Get paper data
                    paper_data = self.redis_client.hgetall(key)
                    if not paper_data:
                        continue
                    
                    # Decode the data
                    paper_info = {}
                    for k, v in paper_data.items():
                        k_str = k.decode('utf-8') if isinstance(k, bytes) else str(k)
                        if k_str == 'vector':
                            # Convert bytes back to numpy array
                            vector = np.frombuffer(v, dtype=np.float32)
                            paper_info['vector'] = vector
                        else:
                            paper_info[k_str] = v.decode('utf-8') if isinstance(v, bytes) else str(v)
                    
                    # Calculate cosine similarity
                    if 'vector' in paper_info:
                        similarity = np.dot(query_embedding, paper_info['vector']) / (
                            np.linalg.norm(query_embedding) * np.linalg.norm(paper_info['vector'])
                        )
                        
                        similarities.append({
                            'paper_id': paper_info.get('paper_id', 'Unknown'),
                            'title': paper_info.get('title', 'Unknown Title'),
                            'abstract': paper_info.get('abstract', 'No abstract')[:300] + "...",
                            'similarity': float(similarity),
                            'source': 'arxiv'  # Default source
                        })
                        
                except Exception as e:
                    print(f"⚠️ Error processing key {key_str}: {e}")
                    continue
            
            # Sort by similarity and return top k
            similarities.sort(key=lambda x: x['similarity'], reverse=True)
            top_results = similarities[:top_k]
            
            # Format for response
            papers = []
            for paper in top_results:
                papers.append({
                    "title": paper['title'],
                    "paper_id": paper['paper_id'],
                    "abstract": paper['abstract'],
                    "similarity_score": f"{paper['similarity']:.4f}",
                    "source": paper['source']
                })
            
            return {
                "papers": papers,
                "total_found": len(similarities),
                "showing": len(papers)
            }
            
        except Exception as e:
            return {"error": f"Search failed: {str(e)}"}
    
    def add_to_reading_list(self, paper_id: str) -> Dict[str, Any]:
        """
        Add a paper to reading list by updating its status in Notion.
        
        Args:
            paper_id: ID of the paper to add to reading list
            
        Returns:
            Dictionary with status or error
        """
        try:
            # Query Notion database for the paper
            response = self.notion.databases.query(
                self.database_id,
                filter={"property": "Paper ID", "rich_text": {"equals": paper_id}}
            )
            
            if not response["results"]:
                return {"error": f"Paper with ID '{paper_id}' not found in Notion database"}
            
            # Update the paper status
            page_id = response["results"][0]["id"]
            self.notion.pages.update(
                page_id,
                properties={
                    "Status": {"select": {"name": "To Read"}}
                }
            )
            
            # Get paper title for confirmation
            paper_title = "Unknown"
            try:
                title_prop = response["results"][0]["properties"].get("Title", {})
                if title_prop.get("title"):
                    paper_title = title_prop["title"][0]["text"]["content"]
            except:
                pass
            
            return {
                "status": "success",
                "message": f"Added '{paper_title[:50]}...' to reading list",
                "paper_id": paper_id
            }
            
        except Exception as e:
            return {"error": f"Failed to add to reading list: {str(e)}"}
    
    def recommend_papers(self, limit: int = 3) -> Dict[str, Any]:
        """
        Recommend papers based on sentiment and publication date.
        
        Args:
            limit: Number of recommendations to return
            
        Returns:
            Dictionary with recommended papers or error
        """
        try:
            # Query all papers from Notion
            response = self.notion.databases.query(
                self.database_id,
                page_size=100
            )
            
            papers = []
            for page in response["results"]:
                try:
                    properties = page["properties"]
                    
                    # Extract paper information
                    paper_id = "Unknown"
                    if properties.get("Paper ID", {}).get("rich_text"):
                        paper_id = properties["Paper ID"]["rich_text"][0]["text"]["content"]
                    
                    title = "Unknown Title"
                    if properties.get("Title", {}).get("title"):
                        title = properties["Title"]["title"][0]["text"]["content"]
                    
                    sentiment = 0.0
                    if properties.get("Sentiment", {}).get("number") is not None:
                        sentiment = properties["Sentiment"]["number"]
                    
                    published = "Unknown"
                    if properties.get("Publication Date", {}).get("date"):
                        published = properties["Publication Date"]["date"]["start"]
                    
                    # Get current status
                    status = "Unread"
                    if properties.get("Status", {}).get("select"):
                        status = properties["Status"]["select"]["name"]
                    
                    papers.append({
                        "paper_id": paper_id,
                        "title": title,
                        "sentiment": sentiment,
                        "published": published,
                        "status": status
                    })
                    
                except Exception as e:
                    print(f"Error processing paper: {e}")
                    continue
            
            # Filter unread papers and sort by sentiment and date
            unread_papers = [p for p in papers if p["status"] in ["Unread", "To Read"]]
            
            if not unread_papers:
                return {
                    "recommended_papers": [],
                    "message": "No unread papers available for recommendation"
                }
            
            # Sort by sentiment (higher is better) and publication date (newer is better)
            sorted_papers = sorted(
                unread_papers, 
                key=lambda x: (x["sentiment"], x["published"]), 
                reverse=True
            )[:limit]
            
            # Format recommendations
            recommendations = []
            for paper in sorted_papers:
                recommendations.append({
                    "paper_id": paper["paper_id"],
                    "title": paper["title"],
                    "sentiment_score": f"{paper['sentiment']:.3f}",
                    "published_date": paper["published"]
                })
            
            return {
                "recommended_papers": recommendations,
                "total_unread": len(unread_papers)
            }
            
        except Exception as e:
            return {"error": f"Recommendation failed: {str(e)}"}
    
    def get_database_stats(self) -> Dict[str, Any]:
        """Get statistics about the databases."""
        try:
            # Redis stats
            redis_keys = self.redis_client.keys(f"{self.vector_set_name}:*")
            redis_paper_count = len([k for k in redis_keys if not k.decode().endswith(':metadata')])
            
            # Notion stats - get all results by paginating
            all_papers = []
            has_more = True
            next_cursor = None
            
            while has_more:
                response = self.notion.databases.query(
                    self.database_id, 
                    start_cursor=next_cursor,
                    page_size=100
                )
                all_papers.extend(response.get("results", []))
                has_more = response.get("has_more", False)
                next_cursor = response.get("next_cursor")
            
            notion_total = len(all_papers)
            
            return {
                "redis_papers": redis_paper_count,
                "notion_papers": notion_total,
                "vector_dimension": self.model.get_sentence_embedding_dimension(),
                "status": "healthy"
            }
        except Exception as e:
            return {"error": f"Stats retrieval failed: {str(e)}"}

# Initialize FastAPI app
app = FastAPI(title="Research Papers MCP Server", version="1.0.0")

# Initialize MCP server
mcp_server = ResearchPapersMCP()

@app.get("/")
def root():
    """Health check endpoint"""
    return {
        "message": "Research Papers MCP Server is running",
        "version": "1.0.0",
        "status": "healthy"
    }

@app.get("/health")
def health_check():
    """Detailed health check"""
    stats = mcp_server.get_database_stats()
    return {
        "service": "healthy",
        "databases": stats
    }

@app.post("/mcp", response_model=MCPResponse)
async def mcp_endpoint(request: MCPRequest):
    """Main MCP endpoint for tool execution"""
    try:
        tool_name = request.tool
        params = request.params
        
        # Route to appropriate tool
        if tool_name == "search_papers":
            keyword = params.get("keyword", "")
            top_k = params.get("top_k", 5)
            result = mcp_server.search_papers(keyword, top_k)
            
        elif tool_name == "add_to_reading_list":
            paper_id = params.get("paper_id", "")
            result = mcp_server.add_to_reading_list(paper_id)
            
        elif tool_name == "recommend_papers":
            limit = params.get("limit", 3)
            result = mcp_server.recommend_papers(limit)
            
        elif tool_name == "get_stats":
            result = mcp_server.get_database_stats()
            
        else:
            result = {"error": f"Unknown tool: {tool_name}"}
        
        return MCPResponse(result=result)
        
    except Exception as e:
        return MCPResponse(result=None, error=str(e))

# Add CORS middleware for development
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if __name__ == "__main__":
    print("🚀 Starting Research Papers MCP Server...")
    print(f"📊 Redis Vector Database: localhost:6379")
    print(f"📝 Notion Database: {mcp_server.database_id}")
    print(f"🌐 Server will run on: http://localhost:8000")
    print(f"📚 API Documentation: http://localhost:8000/docs")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_level="info"
    )