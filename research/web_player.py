"""Play chess against the trained model in a browser, with pieces and sound.

A zero-new-dependency local web app: Python's stdlib HTTP server serves a single
self-contained HTML page (CSS-grid board, Unicode pieces, WebAudio-synthesised
sounds - no image or audio assets) and answers moves by reusing the project's
usable player, `arena.PolicySearchAgent` (supervised policy + material negamax
and capture-quiescence), via `play.load_agent`.

Usage (from the repo root):
    python research/web_player.py                     # you play White, opens browser
    python research/web_player.py --color black       # you play Black
    python research/web_player.py --depth 3           # stronger, slower agent
    python research/web_player.py --weights path.pth  # explicit supervised weights
    python research/web_player.py --port 8000 --no-open

Then open the printed URL (default http://127.0.0.1:8765).
"""
import argparse
import json
import os
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))

from play import load_agent  # noqa: E402

STATE = {
    "board": chess.Board(),
    "lock": threading.Lock(),
    "agent": None,
    "human_white": True,
}


def _side_to_move_is_human(board):
    return (board.turn == chess.WHITE) == STATE["human_white"]


def _snapshot(board, human_move=None, ai_move=None):
    return {
        "fen": board.fen(),
        "legal": [m.uci() for m in board.legal_moves],
        "turn": "w" if board.turn == chess.WHITE else "b",
        "human_white": STATE["human_white"],
        "human_move": human_move,
        "ai_move": ai_move,
        "check": board.is_check(),
        "gameover": board.is_game_over(),
        "result": board.result() if board.is_game_over() else "*",
    }


def _move_info(board, move):
    return {
        "uci": move.uci(),
        "san": board.san(move),
        "capture": board.is_capture(move),
    }


def _agent_reply_if_needed(board):
    """If it is the agent's turn and the game is live, make its move and report it."""
    if board.is_game_over() or _side_to_move_is_human(board):
        return None
    move = STATE["agent"].choose(board)
    info = _move_info(board, move)
    board.push(move)
    return info


def api_new():
    with STATE["lock"]:
        STATE["board"] = chess.Board()
        board = STATE["board"]
        ai_move = _agent_reply_if_needed(board)
        return _snapshot(board, human_move=None, ai_move=ai_move)


def api_move(uci):
    with STATE["lock"]:
        board = STATE["board"]
        if board.is_game_over():
            return {"ok": False, "error": "game is over; start a new game"}
        if not _side_to_move_is_human(board):
            return {"ok": False, "error": "not your turn"}
        try:
            move = chess.Move.from_uci(uci)
        except ValueError:
            return {"ok": False, "error": f"unparseable move: {uci}"}
        if move not in board.legal_moves:
            return {"ok": False, "error": f"illegal move: {uci}"}
        human_move = _move_info(board, move)
        board.push(move)
        ai_move = _agent_reply_if_needed(board)
        return dict(ok=True, **_snapshot(board, human_move=human_move, ai_move=ai_move))


HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Chess vs the trained model</title>
<style>
  :root { --light:#f0d9b5; --dark:#b58863; --sel:#f7ec7a; --last:#cdd26a; --check:#e0574a; }
  * { box-sizing:border-box; }
  body { margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
         gap:28px; flex-wrap:wrap; background:#262421; color:#eee;
         font:15px/1.4 system-ui,Segoe UI,Roboto,sans-serif; padding:24px; }
  #wrap { display:flex; flex-direction:column; align-items:center; gap:12px; }
  #board { display:grid; grid-template-columns:repeat(8,64px); grid-template-rows:repeat(8,64px);
           border:4px solid #17161a; border-radius:4px; box-shadow:0 10px 40px rgba(0,0,0,.5); }
  .sq { width:64px; height:64px; display:flex; align-items:center; justify-content:center;
        font-size:46px; line-height:1; cursor:pointer; user-select:none; position:relative; }
  .sq.light{background:var(--light);} .sq.dark{background:var(--dark);}
  .sq.last{background:var(--last);} .sq.sel{background:var(--sel);}
  .sq.check{background:var(--check);}
  .sq .pc.w{color:#fff; -webkit-text-stroke:2px #141414; text-shadow:0 2px 4px rgba(0,0,0,.55);}
  .sq .pc.b{color:#141414; -webkit-text-stroke:2px #f5f5f5; text-shadow:0 2px 4px rgba(0,0,0,.45);}
  .sq.target::after{content:""; position:absolute; width:18px; height:18px; border-radius:50%;
        background:rgba(20,20,20,.28);}
  .sq.target.cap::after{width:52px;height:52px;background:transparent;border:4px solid rgba(20,20,20,.30);}
  #panel { width:300px; display:flex; flex-direction:column; gap:12px; }
  h1 { font-size:18px; margin:0; }
  #status { background:#33312e; border-radius:6px; padding:10px 12px; min-height:44px; }
  #hist { background:#33312e; border-radius:6px; padding:10px 12px; max-height:220px; overflow:auto;
          font-family:ui-monospace,Consolas,monospace; font-size:13px; white-space:pre-wrap; }
  button { background:#5d8a4e; color:#fff; border:0; border-radius:6px; padding:10px 14px;
           font-size:14px; cursor:pointer; }
  button:hover{background:#6ba05a;}
  .muted{color:#9a978f;font-size:12px;}
</style>
</head>
<body>
<div id="wrap">
  <div id="board"></div>
  <div class="muted">click a piece, then a highlighted square</div>
</div>
<div id="panel">
  <h1>Chess vs the trained model</h1>
  <div id="status">loading…</div>
  <div id="hist"></div>
  <button id="new">New game</button>
  <div class="muted">you are <span id="mycolor">White</span> · agent = supervised policy + search</div>
</div>
<script>
const GLYPH = {p:"\u265F", r:"\u265C", n:"\u265E", b:"\u265D", q:"\u265B", k:"\u265A"};
let S = null, selected = null;

// ---- WebAudio-synthesised sounds (no audio files) ----
let AC = null;
function ac(){ if(!AC) AC = new (window.AudioContext||window.webkitAudioContext)(); return AC; }
function tone(freq, dur, type, gain, when){
  const c = ac(), o = c.createOscillator(), g = c.createGain();
  o.type = type||"triangle"; o.frequency.value = freq;
  const t = c.currentTime + (when||0);
  g.gain.setValueAtTime(0.0001, t);
  g.gain.exponentialRampToValueAtTime(gain||0.25, t+0.01);
  g.gain.exponentialRampToValueAtTime(0.0001, t+dur);
  o.connect(g); g.connect(c.destination); o.start(t); o.stop(t+dur+0.02);
}
const sMove    = () => tone(196, 0.07, "triangle", 0.22);
const sCapture = () => { tone(110, 0.10, "square", 0.20); tone(70, 0.12, "sine", 0.25, 0.02); };
const sCheck   = () => { tone(880, 0.10, "sine", 0.20); tone(1320, 0.14, "sine", 0.18, 0.10); };
const sEnd     = (win) => win ? [523,659,784,1046].forEach((f,i)=>tone(f,0.18,"triangle",0.22,i*0.12))
                              : [392,330,262].forEach((f,i)=>tone(f,0.20,"sine",0.20,i*0.14));

function sqName(i){ return "abcdefgh"[i%8] + (8-Math.floor(i/8)); }

function render(){
  const board = document.getElementById("board");
  board.innerHTML = "";
  const rows = S.fen.split(" ")[0].split("/");
  const legalFrom = {};
  for (const u of S.legal){ (legalFrom[u.slice(0,2)] = legalFrom[u.slice(0,2)]||[]).push(u.slice(2,4)); }
  const lastSq = new Set();
  if (S.ai_move){ lastSq.add(S.ai_move.uci.slice(0,2)); lastSq.add(S.ai_move.uci.slice(2,4)); }
  if (S.human_move){ lastSq.add(S.human_move.uci.slice(0,2)); lastSq.add(S.human_move.uci.slice(2,4)); }
  let kingCheckSq = null;
  if (S.check){
    let r=0;
    outer: for (const row of rows){
      let c=0;
      for (const ch of row){
        if (ch===(S.turn==="w"?"K":"k")){ kingCheckSq = r*8+c; break outer; }
        c += (ch>="1"&&ch<="8") ? parseInt(ch) : 1;
      }
      r++;
    }
  }
  for (let r=0; r<8; r++){
    let c=0;
    for (const ch of rows[r]){
      const span = (ch>="1"&&ch<="8") ? parseInt(ch) : 1;
      for (let k=0; k<span; k++){
        const i = r*8 + (c+k), name = sqName(i);
        const d = document.createElement("div");
        d.className = "sq " + (((r+(c+k))%2===0)?"light":"dark");
        if (lastSq.has(name)) d.classList.add("last");
        if (selected===name) d.classList.add("sel");
        if (kingCheckSq===i) d.classList.add("check");
        const isPiece = !(ch>="1"&&ch<="8");
        if (selected && (legalFrom[selected]||[]).includes(name)){
          d.classList.add("target");
          if (isPiece) d.classList.add("cap");
        }
        if (isPiece){
          const p = document.createElement("span");
          p.className = "pc " + (ch===ch.toUpperCase() ? "w":"b");
          p.textContent = GLYPH[ch.toLowerCase()];
          d.appendChild(p);
        }
        d.dataset.sq = name;
        d.onclick = () => click(name);
        board.appendChild(d);
      }
      c += span;
    }
  }
  document.getElementById("mycolor").textContent = S.human_white ? "White":"Black";
  status();
}

function status(){
  const el = document.getElementById("status");
  if (S.gameover){ el.textContent = "Game over — result " + S.result; return; }
  const mine = (S.turn==="w") === S.human_white;
  el.textContent = (mine ? "Your move" : "Agent thinking…") + (S.check ? " — CHECK!" : "");
}

async function post(path, body){
  const res = await fetch(path, {method:"POST", headers:{"Content-Type":"application/json"},
                                 body: JSON.stringify(body||{})});
  return res.json();
}

async function click(name){
  if (S.gameover) return;
  const mine = (S.turn==="w") === S.human_white;
  if (!mine) return;
  if (selected && name!==selected){
    const uci = selected + name;
    if (S.legal.includes(uci)){
      selected = null;
      const r = await post("/api/move", {uci});
      if (!r.ok){ alert(r.error); return; }
      S = r; sounds(); render(); history();
      return;
    }
  }
  const has = S.legal.some(u => u.slice(0,2)===name);
  selected = has ? name : null;
  render();
}

function sounds(){
  if (S.human_move) (S.human_move.capture ? sCapture : sMove)();
  if (S.ai_move) setTimeout(()=> (S.ai_move.capture ? sCapture : sMove)(), 120);
  if (S.check) setTimeout(sCheck, 240);
  if (S.gameover){
    const humanWon = (S.result==="1-0") === S.human_white;
    const drew = S.result==="1/2-1/2";
    setTimeout(()=> sEnd(drew ? false : humanWon), 380);
  }
}

function history(){
  const el = document.getElementById("hist");
  const lines = [];
  if (S.human_move) lines.push("you: " + S.human_move.san);
  if (S.ai_move) lines.push("agent: " + S.ai_move.san);
  el.textContent = (el.textContent ? el.textContent + "\n" : "") + lines.join("  ");
  el.scrollTop = el.scrollHeight;
}

async function newGame(){
  document.getElementById("hist").textContent = "";
  selected = null;
  S = await post("/api/new");
  if (S.ai_move) setTimeout(()=> (S.ai_move.capture ? sCapture : sMove)(), 120);
  render();
}
document.getElementById("new").onclick = newGame;
newGame();
</script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype):
        data = body if isinstance(body, bytes) else body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store, max-age=0")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self._send(200, HTML, "text/html; charset=utf-8")

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError:
            payload = {}
        if self.path == "/api/new":
            out = api_new()
        elif self.path == "/api/move":
            out = api_move(payload.get("uci", ""))
        else:
            self._send(404, json.dumps({"ok": False, "error": "not found"}), "application/json")
            return
        self._send(200, json.dumps(out), "application/json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--color", choices=["white", "black"], default="white")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--qdepth", type=int, default=4)
    ap.add_argument("--weights", default=None)
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--no-open", action="store_true")
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    STATE["agent"] = load_agent(device, args.depth, args.qdepth, weights=args.weights)
    STATE["human_white"] = args.color == "white"

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://127.0.0.1:{args.port}"
    print(f"serving {url}  (you play {args.color}, agent depth {args.depth}+q{args.qdepth} on {device})")
    print("Ctrl-C to stop.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")


if __name__ == "__main__":
    main()
