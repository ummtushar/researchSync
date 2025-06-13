#!/usr/bin/env python3

import os
from datetime import datetime, timedelta
from typing import List, Dict
import xml.etree.ElementTree as ET
import requests
from textblob import TextBlob
from notion_client import Client
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

load_dotenv("../.env")

mcp = FastMCP("notion-arxiv")

notion = Client(auth=os.getenv("NOTION_TOKEN"))
parent_page_id = os.getenv("NOTION_PARENT_PAGE_ID")
current_database_id = None

def create_notion_database(title: str = "Research Papers") -> str: #same as init_db in ingestion_service
    global current_database_id
    
    payload = {
        "parent": {"type": "page_id", "page_id": parent_page_id},
        "title": [{"type": "text", "text": {"content": title}}],
        "properties": {
                        "Title": {"title": {}},
                        "Authors": {"rich_text": {}},
                        "Abstract": {"rich_text": {}},
                        "Link": {"url": {}},
                        "Publication Date": {"date": {}},
                        "Paper ID": {"rich_text": {}},
                        "Categories": {"multi_select": {"options": []}},
                        "Source": {"select": {"options": [
                            {"name": "arXiv", "color": "blue"},
                            {"name": "Journal", "color": "green"},
                            {"name": "Conference", "color": "orange"}
                        ]}},
                        "Sentiment": {"number": {"format": "number"}},
                        "Status": {"select": {"options": [
                            {"name": "Unread", "color": "default"},
                            {"name": "To Read", "color": "orange"},
                            {"name": "Reading", "color": "blue"},
                            {"name": "Completed", "color": "green"},
                            {"name": "Archived", "color": "gray"}
                        ]}}
                    }
    }
    
    response = notion.databases.create(**payload)
    current_database_id = response["id"]
    return current_database_id

def search_arxiv_papers(query: str, categories: List[str], max_results: int = 50, days_back: int = 30) -> List[Dict]:
    base_url = "http://export.arxiv.org/api/query"
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    if categories:
        cat_query = "+OR+".join([f"cat:{cat}" for cat in categories])
        search_query = f"({cat_query})+AND+({query})" if query else cat_query
    else:
        search_query = query
        
    query_params = f"search_query={search_query}&sortBy=submittedDate&sortOrder=descending&max_results={max_results}"
    
    response = requests.get(f"{base_url}?{query_params}")
    response.raise_for_status()
    
    root = ET.fromstring(response.content)
    namespace = {"atom": "http://www.w3.org/2005/Atom"}
    entries = root.findall("atom:entry", namespace)
    
    papers = []
    for entry in entries:
        published = entry.find("atom:published", namespace).text
        published_date = datetime.strptime(published, "%Y-%m-%dT%H:%M:%SZ")
        
        if published_date < start_date:
            continue
            
        paper_id = entry.find("atom:id", namespace).text.split("/")[-1]
        title = entry.find("atom:title", namespace).text.strip().replace('\n', ' ')
        abstract = entry.find("atom:summary", namespace).text.strip().replace('\n', ' ')
        authors = [author.find("atom:name", namespace).text for author in entry.findall("atom:author", namespace)]
        link = f"https://arxiv.org/abs/{paper_id}"
        
        paper_categories = []
        for category in entry.findall("atom:category", namespace):
            paper_categories.append(category.get('term'))
        
        sentiment = TextBlob(abstract).sentiment.polarity
        
        papers.append({
            "paper_id": paper_id,
            "title": title,
            "authors": authors,
            "abstract": abstract,
            "published": published_date.strftime("%Y-%m-%d"),
            "link": link,
            "categories": paper_categories,
            "sentiment": sentiment
        })
        
    return papers

def add_paper_to_notion(paper: Dict) -> bool:
    if not current_database_id:
        return False
        
    existing = notion.databases.query(
        current_database_id,
        filter={"property": "Paper ID", "rich_text": {"equals": paper["paper_id"]}}
    )
    
    if existing["results"]:
        return False
        
    category_options = [{"name": cat} for cat in paper["categories"]]
    
    notion.pages.create(
        parent={"database_id": current_database_id},
        properties={
            "Title": {"title": [{"text": {"content": paper["title"][:2000]}}]},
            "Authors": {"rich_text": [{"text": {"content": ", ".join(paper["authors"])[:2000]}}]},
            "Abstract": {"rich_text": [{"text": {"content": paper["abstract"][:2000]}}]},
            "Link": {"url": paper["link"]},
            "Publication Date": {"date": {"start": paper["published"]}},
            "Paper ID": {"rich_text": [{"text": {"content": paper["paper_id"]}}]},
            "Categories": {"multi_select": category_options},
            "Source": {"select": {"name": "arXiv"}},
            "Sentiment": {"number": paper["sentiment"]},
            "Status": {"select": {"name": "Unread"}}
        }
    )
    return True

# @mcp.tool()
# def create_database(title: str = "Research Papers") -> str:
#     """Create a new Notion database for research papers"""
#     database_id = create_notion_database(title)
#     return f"Database '{title}' created successfully with ID: {database_id}"

@mcp.tool()
def search_and_populate(
    query: str,
    categories: List[str],
    max_results: int = 50,
    days_back: int = 30
) -> str:
    """Search arXiv papers by categories and query, then populate Notion database
    
    Args:
        query: Search query for papers
        categories: arXiv category tags (e.g., cs.AI, cs.LG, stat.ML)
        max_results: Maximum number of papers to fetch
        days_back: Number of days back to search
    """
    global current_database_id
    
    current_database_id = create_notion_database()
    
    papers = search_arxiv_papers(query, categories, max_results, days_back)
    
    added_count = 0
    for paper in papers:
        if add_paper_to_notion(paper):
            added_count += 1
    
    return f"Found {len(papers)} papers, added {added_count} new papers to Notion database"

@mcp.tool()
def similarity_paper(user_query: str, notion_db_api: str, top_k: int = 5):
    """Find papers in Notion database most similar to user query"""
    
    headers = {
        "Authorization": f"Bearer {os.getenv('NOTION_TOKEN')}",
        "Content-Type": "application/json",
        "Notion-Version": "2022-06-28"
    }
    
    response = requests.post(
        f"https://api.notion.com/v1/databases/{notion_db_api}/query",
        headers=headers,
        json={"page_size": 100}
    )

    if response.status_code != 200:
        return f"Error querying database: {response.status_code} - {response.text}"
    
    papers_data = response.json()
    papers = []
    
    for page in papers_data.get("results", []):
        properties = page.get("properties", {})

        title = ""
        if properties.get("Title", {}).get("title"):
            title = properties["Title"]["title"][0]["text"]["content"]
        
        abstract = ""
        if properties.get("Abstract", {}).get("rich_text"):
            abstract = properties["Abstract"]["rich_text"][0]["text"]["content"]

        paper_id = ""
        if properties.get("Paper ID", {}).get("rich_text"):
            paper_id = properties["Paper ID"]["rich_text"][0]["text"]["content"]

        authors = ""
        if properties.get("Authors", {}).get("rich_text"):
            authors = properties["Authors"]["rich_text"][0]["text"]["content"]
        
        link = ""
        if properties.get("Link", {}).get("url"):
            link = properties["Link"]["url"]

        papers.append({
            "title": title,
            "abstract": abstract,
            "paper_id": paper_id,
            "authors": authors,
            "link": link,
            "notion_page_id": page["id"]
        })

    if not papers:
        return "No papers in the DB"
    
    # Simple version: just return titles for now
    if user_query.lower() == "list titles":
        titles_list = []
        for i, paper in enumerate(papers, 1):
            titles_list.append(f"{i}. {paper['title']}")
        return "\n".join(titles_list)
    
    # Full analysis version
    papers_text = f"User query: '{user_query}'\n\nPapers to analyze:\n\n"
    
    for i, paper in enumerate(papers):
        papers_text += f"Paper {i+1}:\n"
        papers_text += f"Title: {paper['title']}\n"
        papers_text += f"Abstract: {paper['abstract'][:500]}...\n"
        papers_text += f"Paper ID: {paper['paper_id']}\n"
        papers_text += f"Authors: {paper['authors']}\n"
        papers_text += f"Link: {paper['link']}\n"
        papers_text += f"---\n\n"
    
    papers_text += f"\nPlease analyze these papers and return the {top_k} most relevant papers for the query '{user_query}'. For each paper, provide:\n"
    papers_text += "1. Paper number from the list above\n"
    papers_text += "2. Title\n"
    papers_text += "3. Relevance score (1-10)\n"
    papers_text += "4. Brief explanation of why it's relevant\n"
    papers_text += "5. Paper ID\n"
    papers_text += "6. Authors\n"
    papers_text += "7. Link\n\n"
    papers_text += "Format as a numbered list with clear sections."
    
    return papers_text





if __name__ == "__main__":
    mcp.run()