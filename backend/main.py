import os
import requests
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from notion_client import Client
from textblob import TextBlob
from prefect import task, flow
from dotenv import load_dotenv
import xml.etree.ElementTree as ET
import time
import sys
from init_db import create_notion_database_schema

# Load environment variables from .env file
load_dotenv()

# AI-related arXiv categories
AI_CATEGORIES = ["cs.AI", "cs.LG", "stat.ML", "cs.CL", "cs.CV"]

@task(persist_result=False)
def fetch_arxiv_papers(start_date: str, end_date: str, categories: List[str] = AI_CATEGORIES, target_papers: int = 100) -> List[Dict]:
    """Fetch AI-related papers from arXiv for the specified date range."""
    papers = []
    base_url = "http://export.arxiv.org/api/query"
    
    # Build category query
    cat_query = "+OR+".join([f"cat:{cat}" for cat in categories])
    
    # Parse dates
    start_dt = datetime.strptime(start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)  # Include end date
    
    # Start with a larger max_results to account for date filtering
    max_results = min(target_papers * 3, 1000)  # ArXiv API limit is around 1000
    start_index = 0
    
    print(f"Fetching papers from arXiv for categories {categories} between {start_date} and {end_date}...")
    print(f"Target: {target_papers} papers")
    
    while len(papers) < target_papers and start_index < 2000:  # Safety limit
        query = f"search_query={cat_query}&sortBy=submittedDate&sortOrder=descending&max_results={min(500, max_results)}&start={start_index}"
        
        try:
            print(f"Fetching batch starting at index {start_index}...")
            response = requests.get(f"{base_url}?{query}")
            response.raise_for_status()
            
            root = ET.fromstring(response.content)
            namespace = {"atom": "http://www.w3.org/2005/Atom"}
            entries = root.findall("atom:entry", namespace)
            
            if not entries:
                print("No more entries available from arXiv API")
                break
            
            print(f"Retrieved {len(entries)} entries from arXiv API")
            
            batch_papers = []
            for entry in entries:
                try:
                    published = entry.find("atom:published", namespace).text
                    published_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
                    
                    # Check if paper is within date range
                    if not (start_dt <= published_date <= end_dt):
                        continue
                        
                    paper_id = entry.find("atom:id", namespace).text.split("/")[-1]
                    title = entry.find("atom:title", namespace).text.strip().replace('\n', ' ')
                    abstract = entry.find("atom:summary", namespace).text.strip().replace('\n', ' ')
                    authors = [author.find("atom:name", namespace).text for author in entry.findall("atom:author", namespace)]
                    link = f"https://arxiv.org/abs/{paper_id}"
                    
                    # Calculate sentiment
                    try:
                        sentiment = TextBlob(abstract).sentiment.polarity
                    except:
                        sentiment = 0.0
                    
                    paper = {
                        "paper_id": paper_id,
                        "title": title,
                        "authors": authors,
                        "abstract": abstract,
                        "published": published_date.strftime("%Y-%m-%d"),
                        "link": link,
                        "source": "arxiv",
                        "sentiment": sentiment,
                        "post_metrics": {"like_count": 0, "retweet_count": 0}
                    }
                    batch_papers.append(paper)
                    
                except Exception as e:
                    print(f"Error processing entry: {e}")
                    continue
            
            papers.extend(batch_papers)
            print(f"Added {len(batch_papers)} papers from this batch. Total so far: {len(papers)}")
            
            # If we have enough papers, break
            if len(papers) >= target_papers:
                papers = papers[:target_papers]  # Trim to exact target
                break
            
            # Move to next batch
            start_index += len(entries)
            
            # Add a small delay to be respectful to the API
            time.sleep(1)
            
        except Exception as e:
            print(f"Error fetching arXiv papers: {e}")
            break
    
    print(f"Final count: {len(papers)} papers retrieved")
    return papers

@task(persist_result=False)
def update_notion_async(paper: Dict, notion_token: str, database_id: str):
    """Add a paper to the Notion database."""
    if not notion_token or not database_id:
        print("Error: NOTION_TOKEN or NOTION_DATABASE_ID is not set.")
        return False
    
    notion_client = Client(auth=notion_token)
    try:
        # Log the loaded environment variables (masked)
        print(f"Attempting Notion update with token (masked): {notion_token[:4]}...{notion_token[-4:]}")
        print(f"Using Notion database ID: {database_id}")
        
        # Check if paper already exists
        existing = notion_client.databases.query(
            database_id,
            filter={"property": "Paper ID", "rich_text": {"equals": paper["paper_id"]}}
        )
        
        if existing["results"]:
            print(f"Paper already exists in Notion: {paper['title'][:50]}...")
            return False
        
        # Create new page
        response = notion_client.pages.create(
            parent={"database_id": database_id},
            properties={
                "Title": {"title": [{"text": {"content": paper["title"][:2000]}}]},  # Notion title limit
                "Authors": {"rich_text": [{"text": {"content": ", ".join(paper["authors"])[:2000]}}]},
                "Abstract": {"rich_text": [{"text": {"content": paper["abstract"][:2000]}}]},
                "Link": {"url": paper["link"]},
                "Publication Date": {"date": {"start": paper["published"]}},
                "Paper ID": {"rich_text": [{"text": {"content": paper["paper_id"]}}]},
                "Source": {"select": {"name": paper["source"]}},
                "Sentiment": {"number": paper["sentiment"]},
                "Status": {"select": {"name": "Unread"}}
            }
        )
        print(f"✅ Successfully added: {paper['title'][:50]}... (Page ID: {response['id']})")
        return True
        
    except Exception as e:
        print(f"❌ Error updating Notion for paper {paper['paper_id']}: {e}")
        return False


@flow(name="ResearchSync arXiv Ingestion", persist_result=False)
def sync_arxiv_to_notion(days_back: int = 30, target_papers: int = 100):
    """
    Sync recent arXiv papers to Notion database.
    
    Args:
        days_back: Number of days back from today to search
        target_papers: Target number of papers to retrieve
    """
    #Init DB on Notion
    db_id = create_notion_database_schema()

    # Calculate date range (recent papers)
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    start_date_str = start_date.strftime("%Y-%m-%d")
    end_date_str = end_date.strftime("%Y-%m-%d")
    
    print(f"🔍 Searching for papers from {start_date_str} to {end_date_str}")
    print(f"🎯 Target: {target_papers} papers")
    
    # Fetch AI papers from arXiv
    papers = fetch_arxiv_papers(
        start_date=start_date_str, 
        end_date=end_date_str, 
        categories=AI_CATEGORIES, 
        target_papers=target_papers
    )
    
    
    print(f"📚 Found {len(papers)} papers to process")
    
    # Add papers to Notion
    successful_adds = 0
    failed_adds = 0
    
    notion_token = 'ntn_I8446634157a83A61IttkJePdbRglYfIl1xrf621koSgmm'
    database_id = '2041f77568ad812e8e61e7c28efcf410' #hardcode to always stick to the same database
    # database_id = db_id #uncomment to always initialise a new database

    
    os.environ["NOTION_DATABASE_ID"] = db_id
    

    
    for i, paper in enumerate(papers, 1):
        print(f"📝 Processing paper {i}/{len(papers)}: {paper['title'][:50]}...")
        
        if update_notion_async(paper, notion_token, database_id):
            successful_adds += 1
        else:
            failed_adds += 1
        
        # Small delay to avoid rate limiting
        time.sleep(0.5)
    
    print(f"\n📊 Results:")
    print(f"✅ Successfully added: {successful_adds}")
    print(f"❌ Failed/Duplicate: {failed_adds}")
    print(f"📈 Total processed: {len(papers)}")

if __name__ == "__main__":
    sync_arxiv_to_notion(30,300)