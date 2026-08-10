import os
import re
import json
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import requests
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/api/youtube-search")
def youtube_search():
    """Find a real, currently-embeddable video ID for a query by reading
    YouTube's own search results page server-side (no API key needed, no
    CORS issue since this runs on the backend, not in the browser).
    Much more reliable than the old listType=search embed trick, which
    YouTube has made increasingly unreliable."""
    query = request.args.get("q", "").strip()
    if not query:
        return jsonify({"error": "No search query given."}), 400

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        resp = requests.get(
            "https://www.youtube.com/results",
            params={"search_query": query, "hl": "en", "gl": "US"},
            headers=headers,
            timeout=10,
        )
        resp.raise_for_status()
        match = re.search(r'"videoId":"([a-zA-Z0-9_-]{11})"', resp.text)
        if not match:
            return jsonify({"error": "No video found for that search."}), 404
        video_id = match.group(1)
        return jsonify({
            "videoId": video_id,
            "searchUrl": "https://www.youtube.com/results?search_query=" + requests.utils.quote(query),
        })
    except requests.RequestException as exc:
        print(f"\n[YouTube search error] {exc}\n")
        return jsonify({"error": f"Could not search YouTube: {exc}"}), 502

def to_groq_messages(system, claude_messages):
    """Convert the frontend's Claude-format message history into Groq's
    OpenAI-style messages list."""
    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    for msg in claude_messages:
        role = msg.get("role")
        content = msg.get("content")

        if isinstance(content, str):
            messages.append({"role": role, "content": content})
            continue

        if not isinstance(content, list):
            continue

        if role == "assistant":
            text_parts = [b.get("text", "") for b in content if b.get("type") == "text"]
            tool_calls = []
            for b in content:
                if b.get("type") == "tool_use":
                    tool_calls.append({
                        "id": b.get("id"),
                        "type": "function",
                        "function": {
                            "name": b.get("name"),
                            "arguments": json.dumps(b.get("input", {})),
                        },
                    })
            entry = {"role": "assistant", "content": " ".join(text_parts)}
            if tool_calls:
                entry["tool_calls"] = tool_calls
            messages.append(entry)

        elif role == "user":
            for b in content:
                if b.get("type") == "tool_result":
                    messages.append({
                        "role": "tool",
                        "tool_call_id": b.get("tool_use_id"),
                        "content": str(b.get("content", "")),
                    })

    return messages


def to_groq_tools(claude_tools):
    if not claude_tools:
        return None
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t.get("description", ""),
                "parameters": t.get("input_schema", {"type": "object", "properties": {}}),
            },
        }
        for t in claude_tools
    ]


def from_groq_response(groq_json):
    """Convert Groq's OpenAI-style response back into the Claude-style shape
    the frontend expects: {content: [...blocks], stop_reason: "..."}"""
    choice = (groq_json.get("choices") or [{}])[0]
    message = choice.get("message", {})
    content_blocks = []

    if message.get("content"):
        content_blocks.append({"type": "text", "text": message["content"]})

    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        fn = tc.get("function", {})
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        content_blocks.append({
            "type": "tool_use",
            "id": tc.get("id"),
            "name": fn.get("name"),
            "input": args,
        })

    stop_reason = "tool_use" if tool_calls else "end_turn"
    return {"content": content_blocks, "stop_reason": stop_reason}


@app.route("/api/chat", methods=["POST"])
def chat():
    if not GROQ_API_KEY:
        return jsonify({
            "error": "Missing GROQ_API_KEY. Copy .env.example to .env, "
                     "add your free key from console.groq.com, and restart the server."
        }), 500

    body = request.get_json(force=True, silent=True) or {}

    payload = {
        "model": GROQ_MODEL,
        "messages": to_groq_messages(body.get("system", ""), body.get("messages", [])),
        "max_tokens": body.get("max_tokens", 1000),
        "temperature": 0.3,
    }
    groq_tools = to_groq_tools(body.get("tools"))
    if groq_tools:
        payload["tools"] = groq_tools
        payload["tool_choice"] = "auto"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    last_error_detail = None
    for attempt in range(3):
        try:
            resp = requests.post(GROQ_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            return jsonify(from_groq_response(resp.json()))
        except requests.HTTPError as exc:
            detail = exc.response.text if exc.response is not None else str(exc)
            status = exc.response.status_code if exc.response is not None else 502
            print(f"\n[Groq API error {status}, attempt {attempt + 1}/3] {detail}\n")
            last_error_detail = detail
            if status == 400 and "tool_use_failed" in detail:
                continue  
            break  
        except requests.RequestException as exc:
            print(f"\n[Could not reach Groq, attempt {attempt + 1}/3] {exc}\n")
            last_error_detail = str(exc)
            continue
        except Exception as exc:  # noqa: BLE001 - last-resort safety net
            print(f"\n[Unexpected server error, attempt {attempt + 1}/3] {exc}\n")
            last_error_detail = str(exc)
            continue

    print(f"\n[Falling back to friendly message after failure] {last_error_detail}\n")
    return jsonify({
        "content": [{
            "type": "text",
            "text": "I had a little trouble with that one — could you try saying it a bit more simply? "
                    "For example, \"play believer on youtube\" or \"open instagram.com\"."
        }],
        "stop_reason": "end_turn",
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\nEcho is running → http://localhost:{port}\n")
    app.run(host="0.0.0.0", port=port, debug=False)