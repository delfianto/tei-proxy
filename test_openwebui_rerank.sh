#!/bin/bash
PROXY_URL="http://localhost:4000/rerank"

echo "Sending mock OpenWebUI request to $PROXY_URL..."
echo "------------------------------------------------"

curl -s -X POST "$PROXY_URL" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "Elden Beast lore and significance",
    "documents": [
      {
        "page_content": "The Elden Beast is the vassal beast of the Greater Will and the living incarnation of the concept of Order.",
        "metadata": {
          "name": "Chapter 3.md",
          "created_by": "9c42ef38-5083-4d36-b3eb-5afee8b1227f",
          "file_id": "a1c915d9-de99-4e01-ba8e-e26b178ab5ce"
        }
      },
      {
        "page_content": "Radagon and Marika are the same entity, serving as the vessel for the Elden Ring.",
        "metadata": {
          "name": "Chapter 1.md",
          "created_by": "9c42ef38-5083-4d36-b3eb-5afee8b1227f",
          "file_id": "34857b3f-f731-4837-95c5-9e74acde0254"
        }
      },
      {
        "page_content": "Just a plain string to ensure our proxy still handles legacy fallback strings without breaking."
      }
    ]
  }' | jq .

echo ""
echo "------------------------------------------------"
echo "If the proxy is running and TEI is reachable, you should see a successful JSON response with ranked indices and scores."
