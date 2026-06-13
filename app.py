import asyncio, sqlite3, time, json
import joblib, numpy as np, pandas as pd
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

# ── Load models ────────────────────────────────────────────────────────────────
_lgb        = joblib.load("notebooks/lgb_finals.pkl")
_xgb        = joblib.load("notebooks/xgb_finals.pkl")
_feats      = joblib.load("notebooks/features_finals.pkl")
with open("notebooks/train_medians.json") as f:
    _meds = pd.Series(json.load(f))
with open("cat_cols.json") as f:
    _cat_cols = json.load(f)

W_LGB, W_XGB = 0.47, 0.53
DB = "flagged.db"

def init_db():
    con = sqlite3.connect(DB)
    con.execute("CREATE TABLE IF NOT EXISTS flagged (id INTEGER PRIMARY KEY AUTOINCREMENT, txn_id TEXT, score REAL, threshold REAL, reasons TEXT, latency_ms REAL, reviewed INTEGER DEFAULT 0, label TEXT, ts REAL)")
    con.commit();
    con.close()
init_db()

from collections import deque

class _ThresholdEngine:
    def __init__(self):
        self._buf = deque(maxlen=1000)
    def update(self, v): self._buf.append(v)
    @property
    def threshold(self):
        if len(self._buf) < 50: return 0.50
        r = sum(self._buf)/len(self._buf)/0.035
        if r > 2.0:   return 0.35
        elif r > 1.5: return 0.42
        elif r < 0.5: return 0.58
        return 0.50

class _Monitor:
    def __init__(self):
        self.total=0; self.flagged=0
        self._lats=deque(maxlen=500); self._t0=time.perf_counter()
    def record(self, lat, flag):
        self.total+=1; self.flagged+=flag; self._lats.append(lat)
    @property
    def tps(self): return self.total/max(time.perf_counter()-self._t0,1e-6)
    @property
    def p95(self): return float(np.percentile(list(self._lats),95)) if self._lats else 0

_eng = _ThresholdEngine()
_mon = _Monitor()
_ws_clients = []

HIGH_RISK = {'protonmail.com','mail.com','guerrillamail.com'}

app = FastAPI(title="Fraud Detection Engine", version="1.0")

class TxnIn(BaseModel):
    model_config = {"extra":"allow"}

class ReviewIn(BaseModel):
    label: str

def _score(row):

    # ------------------------------
    # DataFrame creation
    # ------------------------------
    t = time.perf_counter()

    df = pd.DataFrame([row])

    df_time = (time.perf_counter() - t) * 1000

    # ------------------------------
    # Category alignment
    # ------------------------------
    t = time.perf_counter()

    for col, cats in _cat_cols.items():
        if col in df.columns:
            df[col] = pd.Categorical(
                df[col],
                categories=cats
            )

    cat_time = (time.perf_counter() - t) * 1000

    # ------------------------------
    # LightGBM
    # ------------------------------
    t = time.perf_counter()

    lgb_s = float(
        _lgb.predict_proba(df[_feats])[:, 1][0]
    )

    lgb_time = (time.perf_counter() - t) * 1000

    # ------------------------------
    # XGBoost
    # ------------------------------
    t = time.perf_counter()

    xgb_s = float(
        _xgb.predict_proba(df[_feats])[:, 1][0]
    )

    xgb_time = (time.perf_counter() - t) * 1000

    # ------------------------------
    # Ensemble
    # ------------------------------

    ens = 0.47 * lgb_s + 0.53 * xgb_s

    latency = df_time + cat_time + lgb_time + xgb_time

    print()
    print(f"DataFrame : {df_time:.2f} ms")
    print(f"Categories: {cat_time:.2f} ms")
    print(f"LightGBM  : {lgb_time:.2f} ms")
    print(f"XGBoost   : {xgb_time:.2f} ms")
    print(f"Total     : {latency:.2f} ms")

    return {
        "txn_id": row.get("TransactionID", "?"),
        "lgb": lgb_s,
        "xgb": xgb_s,
        "score": ens,
        "threshold": 0.50,
        "flagged": False,
        "reasons": [],
        "latency_ms": latency
    }

@app.post("/score")
async def score(txn: TxnIn):
    r = _score(txn.model_dump())
    _mon.record(r["latency_ms"], r["flagged"])
    if r["flagged"]:
        con=sqlite3.connect(DB)
        con.execute("INSERT INTO flagged(txn_id,score,threshold,reasons,latency_ms,ts) VALUES(?,?,?,?,?,?)",
                    (r["txn_id"],r["score"],r["threshold"],str(r["reasons"]),r["latency_ms"],time.time()))
        con.commit(); con.close()
        for ws in _ws_clients:
            try: await ws.send_json(r)
            except: pass
    return r

@app.get("/transactions/flagged")
async def get_flagged(limit:int=50, offset:int=0):
    con=sqlite3.connect(DB)
    rows=con.execute("SELECT * FROM flagged ORDER BY ts DESC LIMIT ? OFFSET ?",(limit,offset)).fetchall()
    con.close()
    cols=["id","txn_id","score","threshold","reasons","latency_ms","reviewed","label","ts"]
    return [dict(zip(cols,r)) for r in rows]

@app.post("/transactions/{txn_id}/review")
async def review(txn_id:str, body:ReviewIn):
    con=sqlite3.connect(DB)
    con.execute("UPDATE flagged SET reviewed=1,label=? WHERE txn_id=?",(body.label,txn_id))
    con.commit(); con.close()
    _eng.update(1 if body.label=="fraud" else 0)
    return {"status":"ok"}

@app.get("/metrics/live")
async def metrics():
    return {"tps":round(_mon.tps,1),"total":_mon.total,
            "flagged":_mon.flagged,"p95_ms":round(_mon.p95,2),
            "threshold":round(_eng.threshold,4)}

@app.websocket("/ws/live-flags")
async def ws_live(ws:WebSocket):
    await ws.accept(); _ws_clients.append(ws)
    try:
        while True: await ws.receive_text()
    except WebSocketDisconnect: _ws_clients.remove(ws)

if __name__=="__main__":
    uvicorn.run("app:app",host="0.0.0.0",port=8000,reload=False)

