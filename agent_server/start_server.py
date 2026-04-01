import json
import logging
import os
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv
from mlflow.genai.agent_server import AgentServer, setup_mlflow_git_based_version_tracking

# Load env vars from .env before importing the agent for proper auth
load_dotenv(dotenv_path=Path(__file__).parent.parent / ".env", override=True)

# Need to import the agent to register the functions with the server
import agent_server.agent  # noqa: E402

agent_server = AgentServer("ResponsesAgent", enable_chat_proxy=False)
# Define the app as a module level variable to enable multiple workers
app = agent_server.app  # noqa: F841

try:
    setup_mlflow_git_based_version_tracking()
except Exception:
    pass  # Not a git repo in deployed app — safe to skip

# ---------------------------------------------------------------------------
# Rate limiting — lightweight ASGI middleware (no starlette dependency)
# ---------------------------------------------------------------------------

from fastapi.responses import HTMLResponse

logger = logging.getLogger(__name__)

RATE_LIMIT_MAX = int(os.environ.get("RATE_LIMIT_MAX_REQUESTS", "30"))
RATE_LIMIT_WINDOW = int(os.environ.get("RATE_LIMIT_WINDOW_SECONDS", "60"))

_rate_limit_store: dict[str, list[float]] = defaultdict(list)


@app.middleware("http")
async def rate_limit_middleware(request, call_next):
    """Per-user rate limiting on /invocations endpoint."""
    if request.url.path != "/invocations":
        return await call_next(request)

    # Identify user by forwarded email (from Databricks App proxy) or IP
    user_key = (
        request.headers.get("x-forwarded-user-email", "")
        or (request.client.host if request.client else "unknown")
    )
    now = time.time()
    cutoff = now - RATE_LIMIT_WINDOW

    # Clean old entries
    _rate_limit_store[user_key] = [t for t in _rate_limit_store[user_key] if t > cutoff]

    if len(_rate_limit_store[user_key]) >= RATE_LIMIT_MAX:
        logger.warning("Rate limit exceeded for user: %s", user_key)
        from starlette.responses import JSONResponse
        return JSONResponse(
            status_code=429,
            content={"detail": f"Rate limit exceeded. Max {RATE_LIMIT_MAX} requests per {RATE_LIMIT_WINDOW}s."},
        )

    _rate_limit_store[user_key].append(now)
    return await call_next(request)

# ---------------------------------------------------------------------------
# Built-in Chat UI — served at GET /
# ---------------------------------------------------------------------------

COMPANY_NAME = os.environ.get("COMPANY_NAME", "Multi-Agent Assistant")

CHAT_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/><meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>""" + COMPANY_NAME + """</title>
<script src="https://cdn.jsdelivr.net/npm/marked/marked.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/dompurify@3.2.4/dist/purify.min.js"></script>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;background:#0e0e10;color:#e8e8e8;height:100vh;display:flex;flex-direction:column}
header{padding:14px 24px;background:#1a1a1f;border-bottom:1px solid #2a2a30;display:flex;align-items:center;gap:10px}
header .logo{font-size:20px;color:#e84d31}
header h1{font-size:15px;font-weight:700}
header .badge{font-size:9px;font-weight:700;color:#e84d31;background:rgba(232,77,49,.12);padding:3px 8px;border-radius:4px;letter-spacing:.08em}
#chat{flex:1;overflow-y:auto;padding:20px 24px;display:flex;flex-direction:column;gap:14px}
.msg{max-width:82%;padding:14px 18px;border-radius:12px;font-size:14px;line-height:1.7}
.msg.user{align-self:flex-end;background:#2563eb;color:#fff;border-bottom-right-radius:4px;white-space:pre-wrap}
.msg.assistant{align-self:flex-start;background:#1e1e26;border:1px solid #2a2a30;border-bottom-left-radius:4px}
.msg.error{background:#7f1d1d;border:1px solid #991b1b;white-space:pre-wrap}
.msg.assistant h2{font-size:15px;font-weight:700;color:#e84d31;margin:16px 0 8px;padding-bottom:4px;border-bottom:1px solid #2a2a30}
.msg.assistant h2:first-child{margin-top:0}
.msg.assistant h3{font-size:14px;font-weight:600;color:#ccc;margin:12px 0 6px}
.msg.assistant p{margin:6px 0}
.msg.assistant strong{color:#f0a070}
.msg.assistant code{background:#16161e;padding:2px 6px;border-radius:4px;font-size:12px;color:#7dd3fc;font-family:'SF Mono',Consolas,monospace}
.msg.assistant pre{background:#16161e;padding:12px;border-radius:6px;overflow-x:auto;margin:8px 0}
.msg.assistant pre code{padding:0;background:none}
.msg.assistant ul,.msg.assistant ol{margin:6px 0 6px 20px}
.msg.assistant li{margin:3px 0}
.msg.assistant hr{border:none;border-top:1px solid #2a2a30;margin:12px 0}
.msg.assistant table{border-collapse:collapse;margin:8px 0;width:100%}
.msg.assistant th{background:#16161e;padding:8px 12px;text-align:left;font-size:12px;color:#aaa;border:1px solid #2a2a30}
.msg.assistant td{padding:8px 12px;border:1px solid #2a2a30;font-size:13px}
.msg.assistant a{color:#60a5fa;text-decoration:none}
#input-area{padding:16px 24px;background:#1a1a1f;border-top:1px solid #2a2a30;display:flex;gap:10px}
#input-area textarea{flex:1;padding:12px;background:#0e0e10;border:1px solid #333;border-radius:8px;color:#e8e8e8;font-size:14px;resize:none;font-family:inherit;outline:none;min-height:44px;max-height:120px}
#input-area textarea:focus{border-color:#e84d31}
#input-area button{padding:0 20px;background:#e84d31;color:#fff;border:none;border-radius:8px;font-weight:600;font-size:14px;cursor:pointer;white-space:nowrap}
#input-area button:disabled{opacity:.5;cursor:not-allowed}
.typing{color:#888;font-style:italic;font-size:13px}
</style>
</head>
<body>
<header>
  <span class="logo">&#x2B21;</span>
  <h1>""" + COMPANY_NAME + """</h1>
  <span class="badge">MULTI-AGENT</span>
</header>
<div id="chat"></div>
<div id="input-area">
  <textarea id="input" placeholder="Ask me anything..." rows="1"
    onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send()}"></textarea>
  <button onclick="send()" id="btn">Send</button>
</div>
<script>
marked.setOptions({breaks:true,gfm:true});
const chat=document.getElementById('chat'),input=document.getElementById('input'),btn=document.getElementById('btn');
function addMsg(role,content,isHtml){
  const d=document.createElement('div');d.className='msg '+role;
  if(isHtml)d.innerHTML=DOMPurify.sanitize(content);else d.textContent=content;
  chat.appendChild(d);chat.scrollTop=chat.scrollHeight;return d;
}
async function send(){
  const q=input.value.trim();if(!q)return;
  input.value='';btn.disabled=true;
  addMsg('user',q,false);
  const t=addMsg('assistant','Thinking...',false);t.classList.add('typing');
  try{
    const r=await fetch('/invocations',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({input:[{role:'user',content:q}]})});
    const d=await r.json();t.classList.remove('typing');
    if(!r.ok){t.textContent=d.detail||'Error: '+r.status;t.classList.add('error');btn.disabled=false;return;}
    let text='';
    for(const item of (d.output||[])){
      const c=item.content;
      if(typeof c==='string'&&c)text+=c;
      else if(Array.isArray(c))for(const x of c){if(x.text)text+=x.text;}
    }
    t.innerHTML=DOMPurify.sanitize(marked.parse(text||JSON.stringify(d.output,null,2)));
  }catch(e){t.classList.remove('typing');t.textContent='Error: '+e.message;t.classList.add('error');}
  btn.disabled=false;input.focus();
}
input.focus();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def root():
    return CHAT_HTML


def main():
    agent_server.run(app_import_string="agent_server.start_server:app")


if __name__ == "__main__":
    main()
