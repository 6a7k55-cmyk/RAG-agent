# api.py

from fastapi import FastAPI
from pydantic import BaseModel
import requests
import openai
import json
import os
from fastapi.concurrency import run_in_threadpool
from dotenv import load_dotenv

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()
openai.api_key = OPENAI_API_KEY
if not openai.api_key:
    raise ValueError("OPENAI_API_KEY is not set in environment variables.")

# -----------------------------
# Configuration
# -----------------------------
KB_SEARCH_URL = "https://squid-app-7q77b.ondigitalocean.app/api/api/kb/factcheck/search"
TOP_K = 5

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="RAG Fact-Checking API")

# -----------------------------
# Request model
# -----------------------------
class SolveRequest(BaseModel):
    claim: str

# -----------------------------
# KB search function
# -----------------------------
def kb_search(query: str, top_k=TOP_K):
    payload = {"query": query, "top_k": top_k}
    try:
        resp = requests.post(KB_SEARCH_URL, json=payload, timeout=10)
        resp.raise_for_status()
        results = resp.json().get("results", [])
        print(f"KB search returned {len(results)} documents.")
        return results
    except Exception as e:
        print("KB Search error:", e)
        return []

# -----------------------------
# GPT reasoning function
# -----------------------------
def analyze_with_gpt(claim: str, docs: list):
    retrieved_context_ids = [doc["doc_id"] for doc in docs]
    context_text = "\n\n".join([f"{doc['doc_id']}: {doc['content']}" for doc in docs])

    prompt = f"""
You are a fact-checking assistant. Analyze the claim below using the context provided.
Return a JSON with the following keys:
- thought_process: your reasoning about the claim
- final_answer: True, False, or Partially True
- citation: relevant document IDs and quotes supporting your verdict

Claim: "{claim}"

Context:
{context_text}
"""

    try:
        # --- NEW OPENAI API SYNTAX ---
        response = openai.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a careful fact-checker."},
                {"role": "user", "content": prompt}
            ],
            temperature=0
        )

        gpt_output = response.choices[0].message.content
        print("GPT output received.")

        try:
            result = json.loads(gpt_output)
        except json.JSONDecodeError:
            print("GPT output is not valid JSON, using fallback.")
            result = {
                "thought_process": gpt_output,
                "final_answer": "Partially True",
                "citation": "; ".join([f"{doc['doc_id']}: {doc['content']}" for doc in docs])
            }

        result["retrieved_context_ids"] = retrieved_context_ids
        return result

    except Exception as e:
        print("GPT error:", e)
        return {
            "thought_process": "",
            "final_answer": "Partially True",
            "citation": "; ".join([f"{doc['doc_id']}: {doc['content']}" for doc in docs]),
            "retrieved_context_ids": retrieved_context_ids,
            "error": str(e)
        }

# -----------------------------
# /solve endpoint
# -----------------------------
@app.post("/solve")
async def solve_endpoint(request: SolveRequest):
    claim = request.claim
    print("Received claim:", claim)
    try:
        # Run blocking calls in threadpool
        docs = await run_in_threadpool(kb_search, claim)
        if not docs:
            print("No KB documents found, returning fallback.")
            return {
                "thought_process": "No documents found in KB.",
                "final_answer": "Partially True",
                "citation": "",
                "retrieved_context_ids": []
            }

        result = await run_in_threadpool(analyze_with_gpt, claim, docs)
        return result

    except Exception as e:
        print("Endpoint error:", e)
        return {
            "thought_process": "",
            "final_answer": "Partially True",
            "citation": "",
            "retrieved_context_ids": [],
            "error": str(e)
        }
