#!/bin/bash
# Exercises the rerank endpoints against a running proxy.
# Override with: PROXY_BASE=http://localhost:8000 ./test_openwebui_rerank.sh
PROXY_BASE="${PROXY_BASE:-http://localhost:4000}"

echo "=== 1. OpenWebUI shim: exact ExternalReranker payload ==="
echo "POST $PROXY_BASE/openwebui/rerank"
echo "------------------------------------------------"
# This is byte-for-byte what OpenWebUI's ExternalReranker.predict() sends:
# {model, query, documents: [strings], top_n: len(documents)} + Bearer auth.
# Expect: results for ALL documents, ordered by index.
curl -s -X POST "$PROXY_BASE/openwebui/rerank" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer dummy-owui-key" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "Elden Beast lore and significance",
    "documents": [
      "The Elden Beast is the vassal beast of the Greater Will and the living incarnation of the concept of Order.",
      "Radagon and Marika are the same entity, serving as the vessel for the Elden Ring.",
      "Just a plain string about something unrelated entirely."
    ],
    "top_n": 3
  }' | jq .

echo ""
echo "=== 2. OpenWebUI shim: legacy dict documents (page_content) ==="
echo "------------------------------------------------"
curl -s -X POST "$PROXY_BASE/openwebui/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Elden Beast lore and significance",
    "documents": [
      {
        "page_content": "The Elden Beast is the vassal beast of the Greater Will.",
        "metadata": { "name": "Chapter 3.md" }
      },
      {
        "page_content": "Radagon and Marika are the same entity.",
        "metadata": { "name": "Chapter 1.md" }
      },
      "Just a plain string to ensure legacy fallback strings still work."
    ]
  }' | jq .

echo ""
echo "=== 3. Jina dialect /jina/rerank with return_documents (alias: /v1/rerank) ==="
echo "------------------------------------------------"
curl -s -X POST "$PROXY_BASE/jina/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "BAAI/bge-reranker-v2-m3",
    "query": "Elden Beast lore and significance",
    "documents": [
      "The Elden Beast is the vassal beast of the Greater Will.",
      "Radagon and Marika are the same entity.",
      "Unrelated filler text."
    ],
    "top_n": 2,
    "return_documents": true
  }' | jq .

echo ""
echo "=== 4. Cohere dialect /cohere/rerank (alias: /v2/rerank) ==="
echo "------------------------------------------------"
curl -s -X POST "$PROXY_BASE/cohere/rerank" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "rerank-v3.5",
    "query": "Elden Beast lore and significance",
    "documents": [
      "The Elden Beast is the vassal beast of the Greater Will.",
      "Radagon and Marika are the same entity.",
      "Unrelated filler text."
    ],
    "top_n": 2
  }' | jq .

echo ""
echo "------------------------------------------------"
echo "Shim (1,2) must return one result per document, index-ordered."
echo "Jina (3) must include object/model/usage and document text."
echo "Cohere (4) must include id and meta.api_version/billed_units."
