import requests
import json
from prefect import task

# --- Configuration ---
# IMPORTANT: Replace with your actual Notion API Key and Parent Page ID
NOTION_API_KEY = "ntn_I8446634157a83A61IttkJePdbRglYfIl1xrf621koSgmm"
PARENT_PAGE_ID = "2031f77568ad8030abe5e68a3ad4aaee" 

DATABASE_TITLE = "Research Paper" # The title of the database to be created

# Define the properties (columns) for the new database
DATABASE_PROPERTIES = {
    "Title": {
        "title": {}
    },
    "Authors": {
        "rich_text": {}
    },
    "Abstract": {
        "rich_text": {}
    },
    "Link": {
        "url": {}
    },
    "Publication Date": {
        "date": {}
    },
    "Paper ID": {
        "rich_text": {}
    },
    "Source": {
        "select": {
            "options": [
                {"name": "Journal"},
                {"name": "Conference"},
                {"name": "Preprint"},
                {"name": "Book"},
                {"name": "Other"}
            ]
        }
    },
    "Sentiment": {
        "number": {
            "format": "number" # You can choose other formats like "percent" or "dollar"
        }
    },
    "Status": {
        "select": {
            "options": [
                {"name": "To Read", "color": "orange"},
                {"name": "Reading", "color": "blue"},
                {"name": "Completed", "color": "green"},
                {"name": "Archived", "color": "gray"}
            ]
        }
    }
}

# --- Notion API Endpoint ---
NOTION_API_URL = "https://api.notion.com/v1/databases"

# --- Headers for API Request ---
HEADERS = {
    "Authorization": f"Bearer {NOTION_API_KEY}",
    "Content-Type": "application/json",
    "Notion-Version": "2022-06-28" # Use a recent Notion API version
}

@task(persist_result=False)
def create_notion_database_schema():
    """
    Creates a new Notion database with only the specified properties (columns).
    No entries are added by this function.
    """
    payload = {
        "parent": {
            "type": "page_id",
            "page_id": PARENT_PAGE_ID
        },
        "title": [
            {
                "type": "text",
                "text": {
                    "content": DATABASE_TITLE
                }
            }
        ],
        "properties": DATABASE_PROPERTIES
    }

    print(f"Attempting to create database '{DATABASE_TITLE}' schema on page ID: {PARENT_PAGE_ID}...")
    response = requests.post(NOTION_API_URL, headers=HEADERS, data=json.dumps(payload))
    response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)

    database_data = response.json()
    database_id = database_data.get("id")
    database_url = database_data.get("url")

    print(f"\nDatabase '{DATABASE_TITLE}' schema created successfully!")
    print(f"Database ID: {database_id}")
    print(f"Database URL: {database_url}")
    return database_id

# if __name__ == "__main__":
#     create_notion_database_schema()
