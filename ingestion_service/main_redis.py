import os
import redis
import numpy as np
from typing import List, Dict, Optional
from notion_client import Client
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
import json
import time
from datetime import datetime
# import sys

# Load environment variables
load_dotenv("../.env")




class NotionRedisSync:
    def __init__(self, 
                 notion_token: str = None, 
                 database_id: str = None,
                 redis_host: str = 'localhost',
                 redis_port: int = 6379,
                 redis_db: int = 0,
                 embedding_model: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the Notion-Redis synchronization service.
        
        Args:
            notion_token: Notion API token
            database_id: Notion database ID
            redis_host: Redis server host
            redis_port: Redis server port
            redis_db: Redis database number
            embedding_model: SentenceTransformer model name
        """
        # Notion configuration
        self.notion_token = notion_token or 'ntn_I8446634157a83A61IttkJePdbRglYfIl1xrf621koSgmm'
        self.database_id = '2041f77568ad816da67dd42b80997200' or os.getenv('NOTION_DATABASE_ID')
        self.notion_client = Client(auth=self.notion_token)
        
        # Redis configuration
        self.redis_client = redis.Redis(
            host=redis_host, 
            port=redis_port, 
            db=redis_db,
            decode_responses=False  # Keep as bytes for vector operations
        )
        
        # Vector embedding model
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        self.vector_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Redis vector set name
        self.vector_set_name = "research_papers_vectors"
        
        print(f"✅ Initialized with vector dimension: {self.vector_dim}")

    def get_notion_papers(self) -> List[Dict]:
        """
        Retrieve all papers from the Notion database.
        
        Returns:
            List of paper dictionaries with Paper ID and Abstract
        """
        papers = []
        has_more = True
        next_cursor = None
        
        print("🔍 Fetching papers from Notion database...")
        
        while has_more:
            try:
                # Query the database
                response = self.notion_client.databases.query(
                    database_id=self.database_id,
                    start_cursor=next_cursor,
                    # page_size=100  # Maximum page size
                )
                
                # Process each paper
                for page in response.get('results', []):
                    try:
                        properties = page.get('properties', {})
                        
                        # Extract Paper ID
                        paper_id_prop = properties.get('Paper ID', {})
                        paper_id = None
                        if paper_id_prop.get('rich_text'):
                            paper_id = paper_id_prop['rich_text'][0]['text']['content']
                        
                        # Extract Abstract
                        abstract_prop = properties.get('Abstract', {})
                        abstract = None
                        if abstract_prop.get('rich_text'):
                            abstract = abstract_prop['rich_text'][0]['text']['content']
                        
                        # Extract Title for logging
                        title_prop = properties.get('Title', {})
                        title = "Unknown Title"
                        if title_prop.get('title'):
                            title = title_prop['title'][0]['text']['content']
                        
                        
                        papers.append({
                            'paper_id': paper_id,
                            'abstract': abstract.strip(),
                            'title': title
                        })
            
                    except Exception as e:
                        print(f"❌ Error processing paper: {e}")
                        continue
                
                # Update pagination
                has_more = response.get('has_more', False)
                next_cursor = response.get('next_cursor')
                
            except Exception as e:
                print(f"❌ Error querying Notion database: {e}")
                break
        
        print(f"📚 Retrieved {len(papers)} papers from Notion")
        return papers

    def create_embeddings(self, abstracts: List[str]) -> np.ndarray:
        """
        Create vector embeddings for a list of abstracts.
        
        Args:
            abstracts: List of abstract texts
            
        Returns:
            NumPy array of embeddings
        """
        print(f"🧠 Creating embeddings for {len(abstracts)} abstracts...")
        embeddings = self.embedding_model.encode(abstracts, show_progress_bar=True)
        return embeddings

    def setup_redis_vector_set(self):
        """
        Set up Redis vector set with proper configuration.
        """
        try:
            # Check if vector set already exists
            if self.redis_client.exists(self.vector_set_name):
                print(f"📊 Vector set '{self.vector_set_name}' already exists")
                return
            
            # Create vector set using Redis commands
            #PS: assumes you have Redis with vector capabilities (Redis Stack or RedisInsight)
            print(f"🔧 Creating vector set '{self.vector_set_name}' with dimension {self.vector_dim}")
            
            # Create index for vector similarity search
            index_cmd = [
                "FT.CREATE", f"{self.vector_set_name}_idx",
                "ON", "HASH",
                "PREFIX", "1", f"{self.vector_set_name}:",
                "SCHEMA",
                "paper_id", "TEXT",
                "vector", "VECTOR", "FLAT", "6", 
                "TYPE", "FLOAT32", 
                "DIM", str(self.vector_dim), 
                "DISTANCE_METRIC", "COSINE"
            ]
            
            self.redis_client.execute_command(*index_cmd)
            print(f"✅ Created vector index for similarity search")
            
        except Exception as e:
            print(f"⚠️ Note: Vector index creation failed (this is normal if using basic Redis): {e}")
            print("📝 Continuing with basic Redis storage...")

    def store_vectors_in_redis(self, papers: List[Dict], embeddings: np.ndarray):
        """
        Store paper vectors in Redis.
        
        Args:
            papers: List of paper dictionaries
            embeddings: NumPy array of embeddings
        """
        print(f"💾 Storing {len(papers)} vectors in Redis...")
        
        # Setup vector set
        self.setup_redis_vector_set()
        
        # Store each paper's vector
        for i, paper in enumerate(papers):
            try:
                paper_id = paper['paper_id']
                vector = embeddings[i].astype(np.float32)
                
                # Create Redis key
                redis_key = f"{self.vector_set_name}:{paper_id}"
                
                # Store as hash with metadata
                self.redis_client.hset(redis_key, mapping={
                    'paper_id': paper_id,
                    'title': paper['title'][:200],  # Truncate long titles
                    'abstract': paper['abstract'][:1000],  # Truncate long abstracts
                    'vector': vector.tobytes(),
                    'vector_dim': self.vector_dim,
                    'updated_at': datetime.now().isoformat()
                })
                
                # Also store in a simple key-value format for easy access
                vector_key = f"vector:{paper_id}"
                self.redis_client.set(vector_key, vector.tobytes())
                
                if (i + 1) % 50 == 0:
                    print(f"  📦 Stored {i + 1}/{len(papers)} vectors...")
                    
            except Exception as e:
                print(f"❌ Error storing vector for paper {paper['paper_id']}: {e}")
                continue
        
        # Store metadata about the vector set
        metadata = {
            'total_papers': len(papers),
            'vector_dimension': self.vector_dim,
            'embedding_model': str(self.embedding_model),
            'last_updated': datetime.now().isoformat(),
            'paper_ids': json.dumps([p['paper_id'] for p in papers])
        }
        
        self.redis_client.hset(f"{self.vector_set_name}:metadata", mapping=metadata)
        print(f"✅ Stored {len(papers)} vectors in Redis")

    def get_existing_paper_ids(self) -> set:
        """
        Get set of paper IDs already stored in Redis.
        
        Returns:
            Set of existing paper IDs
        """
        try:
            # Get all keys matching our pattern
            pattern = f"{self.vector_set_name}:*"
            keys = self.redis_client.keys(pattern)
            
            paper_ids = set()
            for key in keys:
                key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
                if ':' in key_str and not key_str.endswith(':metadata'):
                    paper_id = key_str.split(':', 1)[1]
                    paper_ids.add(paper_id)
            
            return paper_ids
            
        except Exception as e:
            print(f"⚠️ Error getting existing paper IDs: {e}")
            return set()

    def sync_papers(self):
        """
        Main synchronization function that updates Redis with latest Notion data.
        """
        print("🚀 Starting Notion → Redis synchronization...")
        start_time = time.time()
        
        # Get papers from Notion
        papers = self.get_notion_papers()
        
        # Get existing paper IDs in Redis
        existing_ids = self.get_existing_paper_ids()
        print(f"📊 Found {len(existing_ids)} existing papers in Redis")
        
        # Filter papers that need to be added/updated
        new_papers = [p for p in papers if p['paper_id'] not in existing_ids]
        
        if not new_papers:
            print("✅ Redis database is already up to date!")
            return
        
        print(f"🔄 Found {len(new_papers)} new papers to process")
        
        # Create embeddings for new papers
        abstracts = [paper['abstract'] for paper in new_papers]
        embeddings = self.create_embeddings(abstracts)
        
        # Store in Redis
        self.store_vectors_in_redis(new_papers, embeddings)
        
        # Update total count
        total_papers = len(papers)
        print(f"\n📈 Synchronization complete!")
        print(f"⏱️ Total time: {time.time() - start_time:.2f} seconds")
        print(f"📚 Total papers in database: {total_papers}")
        print(f"🆕 New papers added: {len(new_papers)}")

    def search_similar_papers(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        Search for papers similar to the query text.
        
        Args:
            query_text: Text to search for
            top_k: Number of top results to return
            
        Returns:
            List of similar papers with similarity scores
        """
        print(f"🔍 Searching for papers similar to: '{query_text[:50]}...'")
        
        # Create embedding for query
        query_embedding = self.embedding_model.encode([query_text])[0]
        
        # Get all stored vectors
        pattern = f"{self.vector_set_name}:*"
        keys = self.redis_client.keys(pattern)
        
        similarities = []
        
        for key in keys:
            key_str = key.decode('utf-8') if isinstance(key, bytes) else str(key)
            if key_str.endswith(':metadata'):
                continue
                
            try:
                # Get paper data
                paper_data = self.redis_client.hgetall(key)
                # if not paper_data:
                #     continue
                
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
                
                # cosine similarity
                if 'vector' in paper_info:
                    similarity = np.dot(query_embedding, paper_info['vector']) / (
                        np.linalg.norm(query_embedding) * np.linalg.norm(paper_info['vector'])
                    )
                    
                    similarities.append({
                        'paper_id': paper_info.get('paper_id', 'Unknown'),
                        'title': paper_info.get('title', 'Unknown Title'),
                        'abstract': paper_info.get('abstract', 'No abstract'),
                        'similarity': float(similarity)
                    })
                    
            except Exception as e:
                print(f"⚠️ Error processing key {key_str}: {e}")
                continue
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        top_results = similarities[:top_k]
        
        print(f"📊 Found {len(top_results)} similar papers")
        return top_results

    def get_database_stats(self) -> Dict:
        """
        Get statistics about the Redis vector database.
        
        Returns:
            Dictionary with database statistics
        """
        try:
            # Get metadata
            metadata = self.redis_client.hgetall(f"{self.vector_set_name}:metadata")
            
            # Count total papers
            pattern = f"{self.vector_set_name}:*"
            keys = self.redis_client.keys(pattern)
            total_papers = len([k for k in keys if not k.decode('utf-8').endswith(':metadata')])
            
            stats = {
                'total_papers': total_papers,
                'vector_dimension': self.vector_dim,
                'last_updated': metadata.get(b'last_updated', b'Unknown').decode('utf-8') if metadata else 'Unknown',
                'embedding_model': metadata.get(b'embedding_model', b'Unknown').decode('utf-8') if metadata else 'Unknown'
            }
            
            return stats
            
        except Exception as e:
            print(f"⚠️ Error getting database stats: {e}")
            return {}


def main():
    """
    Main function to run the synchronization.
    """
    print("=" * 60)
    print("🔬 Research Papers Vector Database Synchronizer")
    print("=" * 60)
    os.system('brew services start redis')
    
    # Initialize the sync service
    sync_service = NotionRedisSync()
    
    # Sync papers from Notion to Redis
    sync_service.sync_papers()
    
    # Show database statistics
    stats = sync_service.get_database_stats()
    print(f"\n📊 Database Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print(f"🚀 Redis database location: {sync_service.redis_client.connection_pool.connection_kwargs['host']}:{sync_service.redis_client.connection_pool.connection_kwargs['port']}")
    
    # Example search 
    user_query = input("Enter your query to search the paper:")
    print(f"\n🔍 Example: Searching for papers about '{user_query}'")
    similar_papers = sync_service.search_similar_papers(user_query, top_k=3)
    
    for i, paper in enumerate(similar_papers, 1):
        print(f"\n{i}. {paper['title'][:80]}...")
        print(f"   Paper ID: {paper['paper_id']}")
        print(f"   Similarity: {paper['similarity']:.4f}")

    os.system("brew services stop redis")

if __name__ == "__main__":
    main()