"""
Fix: el Agentflow de Flowise Cloud falla en el nodo End con
"Cannot read properties of undefined (reading 'filePath')".

Causa raiz (confirmada): el registro de componentes de Flowise Cloud NO
contiene el componente `endAgentflow`. Al cerrar el flujo, el motor hace
componentNodes['endAgentflow'].filePath -> undefined.

Solucion: eliminar el nodo End y su arista entrante, dejando
`llmAgentflow_5` (Sintesis y Consenso) como nodo terminal. Su output.content
(el JSON de sintesis) pasa a ser la respuesta de la prediccion, que es
exactamente lo que el backend Python ya lee de flowise_response['text'].

NOTA: Flowise Cloud esta detras de Cloudflare, que bloquea el fingerprint TLS
de urllib (error 1010). Por eso toda la I/O HTTP se hace con curl via subprocess.

Uso:
    python scripts/fix_end_node.py            # aplica (GET -> edita -> PUT -> verifica -> predice)
    python scripts/fix_end_node.py --dry-run  # solo muestra que haria, sin PUT
"""
import json
import sys
import io
import os
import time
import subprocess

# Credenciales y endpoint desde la config del proyecto (lee .env / secrets.toml).
# NUNCA hardcodear la API key aquí: el repo es público.
from app.config import settings

BASE = settings.FLOWISE_URL or "https://cloud.flowiseai.com"
FLOW_ID = settings.FLOWISE_CHATFLOW_ID
API_KEY = settings.FLOWISE_API_KEY or ""
if not API_KEY:
    print("ERROR: FLOWISE_API_KEY no está configurada (.env / .streamlit/secrets.toml).")
    sys.exit(1)

SCR = os.path.dirname(os.path.abspath(__file__))
AUTH = "Authorization: Bearer " + API_KEY
CT = "Content-Type: application/json"

DRY = "--dry-run" in sys.argv
LOG = []


def log(msg):
    print(msg)
    LOG.append(str(msg))


def curl(method, path, body_path=None, out_path=None, timeout=300):
    """Returns (http_status:int, body:str). Uses curl to bypass Cloudflare."""
    out_path = out_path or os.path.join(SCR, "_curl_out.bin")
    cmd = [
        "curl", "-s", "-o", out_path, "-w", "%{http_code}",
        "-X", method, BASE + path,
        "-H", AUTH, "-H", CT,
        "--max-time", str(timeout),
    ]
    if body_path:
        cmd += ["--data-binary", "@" + body_path]
    p = subprocess.run(cmd, capture_output=True, text=True)
    status = int(p.stdout.strip() or "0")
    body = ""
    if os.path.exists(out_path):
        body = io.open(out_path, "r", encoding="utf-8").read()
    return status, body


def main():
    status, raw = curl("GET", "/api/v1/chatflows/" + FLOW_ID,
                       out_path=os.path.join(SCR, "_flow_get.json"), timeout=60)
    log("GET chatflow -> HTTP %s (%d bytes)" % (status, len(raw)))
    if status != 200:
        log("ABORT: no se pudo leer el flujo")
        log(raw[:500])
        _dump_log()
        return
    cf = json.loads(raw)

    bpath = r"C:\FLOWISE\flowise_cloud_backup_prefix.json"
    io.open(bpath, "w", encoding="utf-8").write(raw)
    log("backup -> " + bpath)

    fd = json.loads(cf["flowData"])
    names = [n["data"].get("name") for n in fd["nodes"]]
    log("nodes antes: " + ", ".join(names))

    end_ids = [n["id"] for n in fd["nodes"] if n["data"].get("name") == "endAgentflow"]
    log("End node ids: " + str(end_ids))
    fd["nodes"] = [n for n in fd["nodes"] if n["id"] not in end_ids]
    before_edges = len(fd["edges"])
    fd["edges"] = [
        e for e in fd["edges"]
        if e["source"] not in end_ids and e["target"] not in end_ids
    ]
    log("edges: %d -> %d" % (before_edges, len(fd["edges"])))

    ins = {e["source"] for e in fd["edges"]}
    terminals = [n["id"] for n in fd["nodes"] if n["id"] not in ins]
    log("terminales ahora: " + str(terminals))

    new_flowdata = json.dumps(fd, ensure_ascii=False)
    cf_put = {"flowData": new_flowdata}
    put_path = os.path.join(SCR, "_put_body.json")
    io.open(put_path, "w", encoding="utf-8").write(json.dumps(cf_put, ensure_ascii=False))

    if DRY:
        log("DRY-RUN: no se hace PUT.")
        _dump_log()
        return

    status, raw = curl("PUT", "/api/v1/chatflows/" + FLOW_ID, body_path=put_path,
                       out_path=os.path.join(SCR, "_put_resp.json"), timeout=120)
    log("PUT chatflow -> HTTP %s" % status)
    if status >= 300:
        log("ABORT PUT fallo:")
        log(raw[:800])
        _dump_log()
        return

    status, raw = curl("GET", "/api/v1/chatflows/" + FLOW_ID,
                       out_path=os.path.join(SCR, "_flow_verify.json"), timeout=60)
    vfd = json.loads(json.loads(raw)["flowData"])
    vnames = [n["data"].get("name") for n in vfd["nodes"]]
    log("nodes persistidos: " + ", ".join(vnames))
    log("End sigue presente: " + str(any(x == "endAgentflow" for x in vnames)))

    question = json.dumps({
        "section_type": "rag_query",
        "section_text": "Evalua brevemente la formulacion del problema sobre la escasa atencion individualizada del docente en tesis de ingenieria.",
        "retrieved_context": "[Fragmento 1 | Pagina 6] La formulacion del problema describe la escasa atencion individualizada del docente y el exceso de carga academica.",
        "reference_context": "",
        "previous_iteration": "",
        "research_line": "",
        "match_type": "semantic_similarity",
    }, ensure_ascii=False)
    body = {"question": question, "streaming": False}
    pred_body_path = os.path.join(SCR, "_pred_body.json")
    io.open(pred_body_path, "w", encoding="utf-8").write(json.dumps(body, ensure_ascii=False))
    t0 = time.time()
    status, raw = curl("POST", "/api/v1/prediction/" + FLOW_ID, body_path=pred_body_path,
                       out_path=os.path.join(SCR, "_pred_resp.json"), timeout=300)
    log("PREDICT -> HTTP %s en %.1fs (%d bytes)" % (status, time.time() - t0, len(raw)))
    if status == 200:
        try:
            d = json.loads(raw)
            txt = d.get("text", "")
            log("RESULTADO OK. text len=%d" % len(txt))
            log("text head: " + txt[:400].replace("\n", " "))
        except Exception:
            log("respuesta 200 no-JSON: " + raw[:400])
    else:
        log("PREDICT fallo: " + raw[:500])

    _dump_log()


def _dump_log():
    io.open(os.path.join(SCR, "_fix_report.txt"), "w", encoding="utf-8").write("\n".join(LOG) + "\n")


if __name__ == "__main__":
    main()
