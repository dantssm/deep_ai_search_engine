import os
import uuid
import asyncio
import tempfile
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from src.pipeline import DeepResearchPipeline
from src.utils.logger import log_pipeline, add_log_callback, remove_log_callback
from src.services import get_session_manager, set_current_session, get_memory_stats

app = FastAPI(title="AI Open Deep Research Engine", version="4.0")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(BASE_DIR, "static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_pipelines = {}

log_pipeline("Open Deep Research Engine Ready (Refactored)")

@app.get("/")
async def home():
    """Main HTML page endpoint"""
    return FileResponse(os.path.join(BASE_DIR, "index.html"))

@app.get("/api/health")
async def health():
    """Health check endpoint providing system stats"""
    stats = get_memory_stats()
    is_available = stats.get("available", False)
    
    return {
        "status": "ok",
        "timestamp": datetime.now().isoformat(),
        "active_sessions": stats.get("active_sessions", 0),
        "system_usage": {
            "ram_used_mb": round(stats.get("rss_mb", 0), 2) if is_available else "N/A",
            "ram_usage_percent": round(stats.get("percent", 0), 2) if is_available else "N/A"
        }
    }

@app.websocket("/ws/search")
async def websocket_search(ws: WebSocket):
    """WebSocket endpoint for research operations"""
    await ws.accept()
    session_id = str(uuid.uuid4())
    log_pipeline(f"WebSocket connected (session: {session_id[:8]})")

    set_current_session(session_id)
    pipeline = DeepResearchPipeline(session_id)
    _pipelines[session_id] = pipeline
    
    async def log_to_ws(message: str):
        try:
            await ws.send_json({"type": "status", "message": message})
        except Exception:
            pass
    
    add_log_callback(log_to_ws)
    
    try:
        while True:

            data = await ws.receive_json()
            message_type = data.get("type")
            
            if message_type == "create_plan":
                query = data.get("query", "").strip()
                depth = data.get("depth", "standard")
                
                if not query:
                    await ws.send_json({"type": "error", "message": "Query required"})
                    continue
                
                try:
                    plan = await pipeline.create_plan(query, depth)
                    await ws.send_json({"type": "plan_generated", "plan": plan})
                except Exception as e:
                    await ws.send_json({"type": "error", "message": str(e)})
            
            elif message_type == "refine_plan":
                try:
                    refined = await pipeline.refine_plan(
                        data.get("query", ""),
                        data.get("depth", "standard"),
                        data.get("current_plan", {}),
                        data.get("feedback", "")
                    )
                    await ws.send_json({"type": "plan_refined", "plan": refined})
                except Exception as e:
                    await ws.send_json({"type": "error", "message": str(e)})
            
            elif message_type == "execute_research":
                plan = data.get("plan")
                if not plan: continue
                
                try:
                    if data.get("enable_streaming", True):
                        await pipeline.execute_research_streaming(plan, ws=ws)
                    else:
                        result = await pipeline.execute_research(plan)
                        await ws.send_json({"type": "complete", "result": result})

                except Exception as e:
                    log_pipeline(f"Research error: {e}", level="error")
                    await ws.send_json({"type": "error", "message": str(e)})
            
            elif message_type == "clear":
                pipeline.clear()
                await ws.send_json({"type": "cleared"})
                
    except WebSocketDisconnect:
        log_pipeline(f"WebSocket disconnected (session: {session_id[:8]})")
        
    finally:
        remove_log_callback(log_to_ws)
        get_session_manager().cleanup_session(session_id)
        _pipelines.pop(session_id, None)

class ExportRequest(BaseModel):
    session_id: str

@app.post("/api/export")
async def export_report(request: ExportRequest, background_tasks: BackgroundTasks):
    """Export research report as a Markdown file"""
    session_id = request.session_id
    pipeline = _pipelines.get(session_id)
    
    if not pipeline or not getattr(pipeline, 'last_result', None):
        raise HTTPException(status_code=404, detail="No report available")
    
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.md') as tmp:
        tmp_path = tmp.name
        
    try:
        from src.utils.export import export_to_markdown_from_json
        
        if export_to_markdown_from_json(pipeline.last_result, tmp_path):
            background_tasks.add_task(os.unlink, tmp_path)
            
            return FileResponse(
                tmp_path, 
                media_type="text/markdown", 
                filename="research_report.md"
            )
            
        raise HTTPException(status_code=500, detail="Export failed")
        
    except Exception as e:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    """Startup event to initialize periodic cleanup task"""
    async def periodic_cleanup():
        while True:
            await asyncio.sleep(600)
            get_session_manager().cleanup_old_sessions(max_age_seconds=3600)
            
    asyncio.create_task(periodic_cleanup())