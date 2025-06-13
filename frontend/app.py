import gradio as gr
import requests
import pandas as pd
import json
from typing import List, Dict, Any, Optional
import time
import os

class ResearchPapersClient:
    """Client for connecting to the Research Papers MCP Server"""
    
    def __init__(self, mcp_server_url: str = "http://localhost:8000"):
        self.server_url = mcp_server_url
        self.session = requests.Session()
        self._test_connection()
    
    def _test_connection(self):
        """Test connection to MCP server"""
        try:
            response = self.session.get(f"{self.server_url}/health", timeout=5)
            if response.status_code == 200:
                print("✅ Connected to MCP Server")
                health_data = response.json()
                databases = health_data.get('databases', {})
                print(f"📊 Redis Papers: {databases.get('redis_papers', 'Unknown')}")
                print(f"📝 Notion Papers: {databases.get('notion_papers', 'Unknown')}")
            else:
                print(f"⚠️ MCP Server responded with status: {response.status_code}")
        except Exception as e:
            print(f"❌ Cannot connect to MCP Server: {e}")
            print("💡 Make sure to run: cd mcp_service && python main.py")
    
    def _make_request(self, tool: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the MCP server"""
        try:
            payload = {"tool": tool, "params": params}
            response = self.session.post(
                f"{self.server_url}/mcp", 
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json().get("result", {})
        except requests.exceptions.Timeout:
            return {"error": "Request timed out. The server might be processing a large request."}
        except requests.exceptions.ConnectionError:
            return {"error": "Cannot connect to MCP server. Make sure it's running: cd mcp_service && python main.py"}
        except Exception as e:
            return {"error": f"Request failed: {str(e)}"}
    
    def search_papers(self, keyword: str, top_k: int = 5) -> Dict[str, Any]:
        """Search for papers using semantic similarity"""
        if not keyword.strip():
            return {"error": "Please enter a search keyword"}
        
        return self._make_request("search_papers", {
            "keyword": keyword.strip(),
            "top_k": top_k
        })
    
    def add_to_reading_list(self, paper_id: str) -> Dict[str, Any]:
        """Add a paper to the reading list"""
        if not paper_id.strip():
            return {"error": "Please enter a paper ID"}
        
        return self._make_request("add_to_reading_list", {
            "paper_id": paper_id.strip()
        })
    
    def get_recommendations(self, limit: int = 5) -> Dict[str, Any]:
        """Get paper recommendations"""
        return self._make_request("recommend_papers", {
            "limit": limit
        })
    
    def get_server_stats(self) -> Dict[str, Any]:
        """Get server and database statistics"""
        return self._make_request("get_stats", {})

# Initialize client
client = ResearchPapersClient()

def search_papers_ui(keyword: str, num_results: int = 5):
    """UI function for searching papers"""
    if not keyword.strip():
        return "❌ Please enter a search keyword", None
    
    try:
        result = client.search_papers(keyword, num_results)
        
        if "error" in result:
            return f"❌ Search Error: {result['error']}", None
        
        papers = result.get("papers", [])
        
        if not papers:
            return f"📚 No papers found for '{keyword}'. Try different keywords or run: python main_redis.py", None
        
        # Create DataFrame for display
        df_data = []
        for paper in papers:
            df_data.append({
                "Title": paper.get("title", "Unknown"),
                "Paper ID": paper.get("paper_id", "Unknown"),
                "Similarity": paper.get("similarity_score", "0.0000"),
                "Source": paper.get("source", "Unknown"),
                "Abstract Preview": paper.get("abstract", "No abstract")[:200] + "..."
            })
        
        df = pd.DataFrame(df_data)
        
        # Create summary message
        total_found = result.get("total_found", len(papers))
        showing = result.get("showing", len(papers))
        
        summary = f"✅ Found {total_found} papers, showing top {showing} results for '{keyword}'"
        
        return summary, df
        
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}", None

def add_to_reading_list_ui(paper_id: str):
    """UI function for adding papers to reading list"""
    if not paper_id.strip():
        return "❌ Please enter a Paper ID"
    
    try:
        result = client.add_to_reading_list(paper_id)
        
        if "error" in result:
            return f"❌ Error: {result['error']}"
        
        if result.get("status") == "success":
            message = result.get("message", "Successfully added to reading list")
            return f"✅ {message}"
        
        return f"⚠️ Unexpected response: {result}"
        
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}"

def get_recommendations_ui(num_recommendations: int = 5):
    """UI function for getting recommendations"""
    try:
        result = client.get_recommendations(num_recommendations)
        
        if "error" in result:
            return f"❌ Recommendation Error: {result['error']}", None
        
        papers = result.get("recommended_papers", [])
        
        if not papers:
            message = result.get("message", "No recommendations available")
            return f"📚 {message}", None
        
        # Create DataFrame for display
        df_data = []
        for paper in papers:
            df_data.append({
                "Title": paper.get("title", "Unknown"),
                "Paper ID": paper.get("paper_id", "Unknown"),
                "Sentiment Score": paper.get("sentiment_score", "0.000"),
                "Published Date": paper.get("published_date", "Unknown")
            })
        
        df = pd.DataFrame(df_data)
        
        total_unread = result.get("total_unread", len(papers))
        summary = f"✅ Top {len(papers)} recommendations from {total_unread} unread papers"
        
        return summary, df
        
    except Exception as e:
        return f"❌ Unexpected error: {str(e)}", None

def refresh_stats():
    """Get current database statistics"""
    try:
        result = client.get_server_stats()
        
        if "error" in result:
            return f"❌ Stats Error: {result['error']}"
        
        stats_text = f"""📊 **Database Statistics**
        
🔢 **Redis Vector Database:** {result.get('redis_papers', 'Unknown')} papers
📝 **Notion Database:** {result.get('notion_papers', 'Unknown')} papers  
🧠 **Vector Dimension:** {result.get('vector_dimension', 'Unknown')}
🟢 **Status:** {result.get('status', 'Unknown')}

Last updated: {time.strftime('%Y-%m-%d %H:%M:%S')}
"""
        return stats_text
        
    except Exception as e:
        return f"❌ Error getting stats: {str(e)}"

# Create Gradio interface
with gr.Blocks(
    title="ResearchSync AI Dashboard"
) as demo:
    
    gr.Markdown("""
    # 🔬 ResearchSync AI Dashboard
    
    **Semantic Search & Management for Research Papers**
    
    Connect to your Redis vector database and Notion workspace to search, organize, and discover research papers using AI-powered semantic similarity.
    """)
    
    # Connection status
    with gr.Row():
        with gr.Column():
            stats_display = gr.Markdown(value=refresh_stats())
            refresh_btn = gr.Button("🔄 Refresh Stats")
            refresh_btn.click(refresh_stats, outputs=stats_display)
    
    with gr.Tabs():
        
        # Search Tab
        with gr.Tab("🔍 Search Papers"):
            gr.Markdown("**Search for research papers using semantic similarity**")
            
            with gr.Row():
                with gr.Column(scale=3):
                    keyword_input = gr.Textbox(
                        label="Search Query",
                        placeholder="e.g., transformer architecture, reinforcement learning, computer vision...",
                        lines=2
                    )
                with gr.Column(scale=1):
                    num_results = gr.Slider(
                        label="Max Results",
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1
                    )
            
            search_button = gr.Button("🚀 Search Papers", variant="primary")
            
            with gr.Row():
                search_status = gr.Markdown()
            
            with gr.Row():
                search_output = gr.Dataframe(
                    label="Search Results",
                    wrap=True,
                    interactive=False
                )
            
            search_button.click(
                search_papers_ui,
                inputs=[keyword_input, num_results],
                outputs=[search_status, search_output]
            )
            
            # Example queries
            gr.Markdown("**💡 Try these example queries:**")
            gr.Examples(
                examples=[
                    ["machine learning interpretability"],
                    ["graph neural networks"],
                    ["attention mechanisms"],
                    ["federated learning"],
                    ["computer vision transformers"]
                ],
                inputs=keyword_input
            )
        
        # Reading List Tab
        with gr.Tab("📚 Reading List"):
            gr.Markdown("**Add papers to your Notion reading list**")
            
            with gr.Row():
                with gr.Column():
                    paper_id_input = gr.Textbox(
                        label="Paper ID",
                        placeholder="e.g., 2505.23615v1, 1706.03762, etc.",
                        info="Enter the Paper ID from your search results"
                    )
                    add_button = gr.Button("➕ Add to Reading List", variant="primary")
                    add_output = gr.Markdown()
            
            add_button.click(
                add_to_reading_list_ui,
                inputs=paper_id_input,
                outputs=add_output
            )
            
            gr.Markdown("""
            **📋 How to use:**
            1. Search for papers in the Search tab
            2. Copy the Paper ID from your search results  
            3. Paste it here and click "Add to Reading List"
            4. The paper status will be updated in your Notion database
            """)
        
        # Recommendations Tab
        with gr.Tab("⭐ Recommendations"):
            gr.Markdown("**Get AI-powered paper recommendations**")
            
            with gr.Row():
                with gr.Column(scale=1):
                    num_recommendations = gr.Slider(
                        label="Number of Recommendations",
                        minimum=1,
                        maximum=10,
                        value=5,
                        step=1
                    )
                    recommend_button = gr.Button("🎯 Get Recommendations", variant="primary")
                
                with gr.Column(scale=3):
                    recommend_status = gr.Markdown()
            
            recommend_output = gr.Dataframe(
                label="Recommended Papers",
                wrap=True,
                interactive=False
            )
            
            recommend_button.click(
                get_recommendations_ui,
                inputs=num_recommendations,
                outputs=[recommend_status, recommend_output]
            )
            
            gr.Markdown("""
            **🤖 Recommendation Algorithm:**
            - Analyzes paper sentiment scores
            - Considers publication dates
            - Filters unread papers
            - Ranks by relevance and quality
            """)
        
        # Database Info Tab
        with gr.Tab("ℹ️ Database Info"):
            gr.Markdown("**System Information & Troubleshooting**")
            
            with gr.Accordion("🔧 Connection Details", open=True):
                gr.Markdown("""
                **MCP Server:** `http://localhost:8000`
                **Redis Database:** `localhost:6379`
                **Notion Integration:** ✅ Connected
                
                **Vector Search:** Semantic similarity using sentence-transformers
                **Model:** `all-MiniLM-L6-v2` (384 dimensions)
                """)
            
            with gr.Accordion("📖 How It Works", open=False):
                gr.Markdown("""
                1. **Paper Ingestion**: ArXiv papers are scraped and stored in Notion
                2. **Vector Creation**: Abstracts are converted to 384-dimensional vectors
                3. **Redis Storage**: Vectors stored as `<Paper ID, Abstract Vector>` pairs
                4. **Semantic Search**: Query text is vectorized and compared using cosine similarity
                5. **Smart Recommendations**: Papers ranked by sentiment analysis and recency
                """)
            
            with gr.Accordion("🛠️ Troubleshooting", open=False):
                gr.Markdown("""
                **Common Issues:**
                
                🔴 **"Cannot connect to MCP server"**
                - Run: `cd mcp_service && python main.py`
                - Check: Server should be running on `localhost:8000`
                
                🔴 **"No papers found"**  
                - Run: `python main_redis.py` (from root directory)
                - Check: Your Notion database has papers with abstracts
                
                🔴 **"Redis connection failed"**
                - Run: `brew services start redis`
                - Check: Redis should be running on `localhost:6379`
                """)

if __name__ == "__main__":
    print("🚀 Starting ResearchSync AI Dashboard...")
    print("📊 Connecting to MCP Server at http://localhost:8000")
    print("🌐 Dashboard will be available at: http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public sharing
        show_error=True
    )