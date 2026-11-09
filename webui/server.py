from __future__ import annotations
import os, sys, uuid, json, time, hmac, base64, asyncio, shutil, mimetypes
from pathlib import Path
from urllib.parse import urlparse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import FileResponse, Response, HTMLResponse
from fastapi.staticfiles import StaticFiles

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
HOME=Path(os.getenv("REA_VIDEO_WEBUI_HOME","/data" if os.getenv("RAILWAY_ENVIRONMENT") else str(Path.home()/".rea-video-webui")))
JOBS=HOME/"jobs"; JOBS.mkdir(parents=True,exist_ok=True)
STATIC=Path(__file__).parent/"static"
app=FastAPI(title="rea-video WebUI",docs_url=None,redoc_url=None)
app.mount("/static",StaticFiles(directory=STATIC),name="static")
tasks=set()

@app.middleware("http")
async def auth(req:Request, call_next):
    if req.url.path=="/api/health": return await call_next(req)
    u,p=os.getenv("REA_WEBUI_USERNAME"),os.getenv("REA_WEBUI_PASSWORD")
    if not u or not p:
        if os.getenv("RAILWAY_ENVIRONMENT"): return Response("Set REA_WEBUI_USERNAME and REA_WEBUI_PASSWORD",503)
        return await call_next(req)
    from fastapi.security.utils import get_authorization_scheme_param
    scheme,cred=get_authorization_scheme_param(req.headers.get("Authorization"))
    try:
        user,pw=base64.b64decode(cred).decode().split(":",1) if scheme.lower()=="basic" else ("","")
        ok=hmac.compare_digest(user,u) and hmac.compare_digest(pw,p)
    except Exception: ok=False
    if not ok: return Response("Authentication required",401,headers={"WWW-Authenticate":'Basic realm="rea-video"'})
    return await call_next(req)

def jobdir(j):
    try: uuid.UUID(j)
    except Exception: raise HTTPException(404,"Job not found")
    p=JOBS/j
    if not p.is_dir(): raise HTTPException(404,"Job not found")
    return p
def read(p): return json.loads(p.read_text())
def write(p,x): p.write_text(json.dumps(x,ensure_ascii=False,indent=2))

@app.get("/")
async def index(): return FileResponse(STATIC/"index.html")
@app.get("/api/health")
async def health(): return {"ok":True,"ffmpeg":shutil.which("ffmpeg") is not None,"python":sys.version.split()[0]}
@app.get("/api/jobs")
async def jobs():
    rows=[]
    for p in JOBS.iterdir():
        if (p/"job.json").is_file():
            try: rows.append(read(p/"job.json"))
            except Exception: pass
    return {"jobs":sorted(rows,key=lambda x:x.get("created_at",0),reverse=True)}
@app.post("/api/analyze")
async def analyze(file:UploadFile=File(...), use_vlm:bool=Form(False), base_url:str=Form(""), api_key:str=Form(""), model:str=Form(""), media:str=Form("frames"), lang:str=Form("zh")):
    suffix=Path(file.filename or "video.mp4").suffix.lower()
    if suffix not in {".mp4",".mov",".m4v",".mkv",".webm",".avi"}: raise HTTPException(415,"Unsupported video format")
    if use_vlm and not all([base_url,api_key,model]): raise HTTPException(422,"Complete BYOK settings first")
    if use_vlm and (urlparse(base_url).scheme!="https" and urlparse(base_url).hostname not in {"localhost","127.0.0.1"}): raise HTTPException(422,"Use HTTPS for remote model endpoints")
    jid=str(uuid.uuid4()); d=JOBS/jid; d.mkdir()
    src=d/("source"+suffix)
    size=0
    with src.open("wb") as f:
        while chunk:=await file.read(1024*1024):
            size+=len(chunk)
            if size>int(os.getenv("REA_VIDEO_WEBUI_MAX_UPLOAD_MB","1024"))*1024*1024: raise HTTPException(413,"Upload too large")
            f.write(chunk)
    meta={"id":jid,"filename":file.filename,"size":size,"status":"queued","progress":0,"stage":"Queued","created_at":time.time(),"use_vlm":use_vlm,"model":model if use_vlm else ""}
    write(d/"job.json",meta)
    async def runner():
        out=d/"output"; out.mkdir()
        args=[sys.executable,"-m","rea_video.cli","-v","decompose",str(src),"-o",str(out),"--lang",lang]
        args += ["--vlm"] if use_vlm else ["--no-vlm"]
        env=os.environ.copy()
        if use_vlm:
            env.update(REA_VLM_BASE_URL=base_url.rstrip("/"),REA_VLM_API_KEY=api_key,REA_VLM_MODEL=model,REA_VLM_MEDIA=media)
        meta.update(status="running",stage="Analyzing",progress=3); write(d/"job.json",meta)
        proc=await asyncio.create_subprocess_exec(*args,cwd=ROOT,env=env,stdout=asyncio.subprocess.PIPE,stderr=asyncio.subprocess.STDOUT)
        lines=[]
        while line:=await proc.stdout.readline():
            text=line.decode(errors="replace").rstrip()
            if use_vlm: text=text.replace(api_key,"***")
            lines.append(text)
            (d/"job.log").write_text("\n".join(lines[-200:]))
            meta.update(progress=min(95,meta["progress"]+1),stage=text[-120:]); write(d/"job.json",meta)
        rc=await proc.wait()
        if rc==0 and (out/"breakdown.json").exists():
            b=read(out/"breakdown.json"); meta.update(status="completed",progress=100,stage="Complete",shots=b.get("summary",{}).get("shots"),verdict=b.get("summary",{}).get("verdict"),confidence=b.get("summary",{}).get("confidence"))
        else: meta.update(status="failed",progress=100,stage="Failed",error="\n".join(lines[-10:]))
        write(d/"job.json",meta)
    task=asyncio.create_task(runner()); tasks.add(task); task.add_done_callback(tasks.discard)
    return meta
@app.get("/api/jobs/{jid}")
async def get_job(jid): return read(jobdir(jid)/"job.json")
@app.get("/api/jobs/{jid}/logs")
async def logs(jid):
    p=jobdir(jid)/"job.log"; return {"lines":p.read_text().splitlines()[-160:] if p.exists() else []}
@app.get("/api/jobs/{jid}/breakdown")
async def breakdown(jid):
    p=jobdir(jid)/"output"/"breakdown.json"
    if not p.exists(): raise HTTPException(409,"Analysis not ready")
    return read(p)
@app.get("/api/jobs/{jid}/source")
async def source(jid):
    p=next(jobdir(jid).glob("source.*"),None)
    if not p: raise HTTPException(404,"Source missing")
    return FileResponse(p,media_type=mimetypes.guess_type(p.name)[0] or "video/mp4")
@app.get("/api/jobs/{jid}/assets/{asset:path}")
async def asset(jid,asset):
    root=(jobdir(jid)/"output").resolve(); p=(root/asset).resolve()
    if not p.is_relative_to(root) or not p.is_file(): raise HTTPException(404,"Asset missing")
    return FileResponse(p,media_type=mimetypes.guess_type(p.name)[0] or "application/octet-stream")
@app.get("/api/jobs/{jid}/report")
async def report(jid):
    p=jobdir(jid)/"output"/"report.html"
    if not p.exists(): raise HTTPException(404,"Report missing")
    html=p.read_text(errors="replace").replace("<head>",f'<head><base href="/api/jobs/{jid}/assets/">',1)
    return HTMLResponse(html)
