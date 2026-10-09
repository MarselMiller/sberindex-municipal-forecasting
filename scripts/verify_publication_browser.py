"""Local, stdlib-only browser QA. This script never edits the page or research files."""
from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import struct
import subprocess
import sys
import time
import shutil
import tempfile
import urllib.parse
import urllib.request
import uuid


class WebSocket:
    def __init__(self, url: str):
        address = urllib.parse.urlsplit(url)
        if address.scheme != "ws" or address.hostname not in {"127.0.0.1", "localhost"}:
            raise ValueError("Only a local browser WebSocket is allowed")
        self.socket = socket.create_connection((address.hostname, address.port), timeout=15)
        self.buffer = bytearray()
        key = base64.b64encode(os.urandom(16)).decode()
        path = address.path + ("?" + address.query if address.query else "")
        request = (
            f"GET {path} HTTP/1.1\r\nHost: {address.hostname}:{address.port}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n\r\n"
        )
        self.socket.sendall(request.encode("ascii"))
        while b"\r\n\r\n" not in self.buffer:
            chunk = self.socket.recv(8192)
            if not chunk:
                raise EOFError("Browser closed during WebSocket handshake")
            self.buffer.extend(chunk)
        raw_headers, rest = bytes(self.buffer).split(b"\r\n\r\n", 1)
        self.buffer = bytearray(rest)
        if b" 101 " not in raw_headers.split(b"\r\n", 1)[0]:
            raise RuntimeError("WebSocket handshake rejected: " + raw_headers.decode(errors="replace"))
        expected = base64.b64encode(hashlib.sha1((key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode()).digest()).decode()
        headers = dict(line.decode().split(":", 1) for line in raw_headers.split(b"\r\n")[1:])
        received = next((value.strip() for name, value in headers.items() if name.lower() == "sec-websocket-accept"), "")
        if received != expected:
            raise RuntimeError("Invalid browser WebSocket acceptance key")

    def _read(self, length: int) -> bytes:
        while len(self.buffer) < length:
            chunk = self.socket.recv(max(8192, min(length - len(self.buffer), 1048576)))
            if not chunk:
                raise EOFError("Browser WebSocket closed")
            self.buffer.extend(chunk)
        result = bytes(self.buffer[:length])
        del self.buffer[:length]
        return result

    def send_frame(self, payload: bytes, opcode: int = 1):
        length = len(payload)
        header = bytes([0x80 | opcode])
        if length < 126:
            header += bytes([0x80 | length])
        elif length <= 65535:
            header += bytes([0x80 | 126]) + struct.pack(">H", length)
        else:
            header += bytes([0x80 | 127]) + struct.pack(">Q", length)
        mask = os.urandom(4)
        masked = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
        self.socket.sendall(header + mask + masked)

    def receive(self) -> dict:
        parts = []
        while True:
            first, second = self._read(2)
            final, opcode = bool(first & 0x80), first & 0x0F
            length = second & 0x7F
            if length == 126:
                length = struct.unpack(">H", self._read(2))[0]
            elif length == 127:
                length = struct.unpack(">Q", self._read(8))[0]
            if length > 128 * 1024 * 1024:
                raise RuntimeError("Unexpectedly large browser response")
            mask = self._read(4) if second & 0x80 else None
            payload = self._read(length)
            if mask:
                payload = bytes(value ^ mask[index % 4] for index, value in enumerate(payload))
            if opcode == 9:
                self.send_frame(payload, 10)
                continue
            if opcode == 10:
                continue
            if opcode == 8:
                raise EOFError("Browser closed the WebSocket")
            if opcode not in {0, 1, 2}:
                raise RuntimeError("Unsupported browser frame")
            parts.append(payload)
            if final:
                return json.loads(b"".join(parts).decode("utf-8"))

    def close(self):
        self.socket.close()


class CDP:
    def __init__(self, url: str):
        self.ws = WebSocket(url)
        self.next_id = 0
        self.events = []

    def call(self, method: str, params: dict | None = None, timeout: float = 20):
        self.next_id += 1
        identifier = self.next_id
        self.ws.send_frame(json.dumps({"id": identifier, "method": method, "params": params or {}}).encode())
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(method)
            self.ws.socket.settimeout(remaining)
            message = self.ws.receive()
            if message.get("id") == identifier:
                if "error" in message:
                    raise RuntimeError(method + ": " + json.dumps(message["error"]))
                return message.get("result", {})
            self.events.append(message)

    def evaluate(self, expression: str, await_promise: bool = False):
        response = self.call("Runtime.evaluate", {
            "expression": expression, "returnByValue": True, "awaitPromise": await_promise,
        })
        if response.get("exceptionDetails"):
            raise RuntimeError("QA expression failed: " + json.dumps(response["exceptionDetails"], ensure_ascii=False))
        return response.get("result", {}).get("value")


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/publication_site_checks/browser"
PAGE = ROOT / "dist/submission-site/index.html"
EDGE = Path(os.environ.get("ProgramFiles(x86)", "C:/Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe"
SOURCE_FILES = ["index.html", "assets/project-report.css", "assets/project-report.js", "assets/project-report-data.js"]
checks: list[dict] = []
screenshots: list[str] = []


def check(name: str, passed: bool, details=None):
    checks.append({"name": name, "passed": bool(passed), "details": details})


def wait_ready(client: CDP, interactive: bool = True):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        ready = client.evaluate("document.readyState === 'complete'" +
                                (" && document.documentElement.classList.contains('is-interactive')" if interactive else ""))
        if ready:
            return
        time.sleep(0.05)
    raise TimeoutError("Page readiness")


def settle(client: CDP):
    client.evaluate("new Promise(resolve => requestAnimationFrame(() => requestAnimationFrame(resolve)))", True)


def screenshot(client: CDP, name: str, target: str | None = None, selector: str | None = None):
    if target or selector:
        query = "document.querySelector(" + json.dumps(selector) + ")" if selector else "document.getElementById(" + json.dumps(target) + ")"
        client.evaluate("(() => {const item=" + query +
                        "; const header=document.querySelector('.site-header').getBoundingClientRect().height;"
                        "window.scrollTo({top:item.getBoundingClientRect().top+window.scrollY-header-20,behavior:'instant'});})()")
    else:
        client.evaluate("window.scrollTo({top:0,behavior:'instant'})")
    settle(client)
    response = client.call("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False, "fromSurface": True})
    (OUTPUT / name).write_bytes(base64.b64decode(response["data"]))
    screenshots.append("outputs/publication_site_checks/browser/" + name)


def viewport(client: CDP, width: int, height: int, mobile: bool):
    client.call("Emulation.setDeviceMetricsOverride", {
        "width": width, "height": height, "deviceScaleFactor": 1, "mobile": mobile,
        "screenWidth": width, "screenHeight": height,
    })
    settle(client)
    return client.evaluate("(() => ({width:innerWidth,clientWidth:document.documentElement.clientWidth,scrollWidth:document.documentElement.scrollWidth,"
                           "bodyWidth:document.body.scrollWidth,"
                           "offenders:[...document.querySelectorAll('body *')].filter(x=>!x.closest('.table-scroll'))"
                           ".filter(x=>{const r=x.getBoundingClientRect();return r.width>0&&(r.right>" + str(width) + "+1||r.left < -1||x.scrollWidth>x.clientWidth+1);})"
                           ".slice(0,15).map(x=>({tag:x.tagName,id:x.id,class:x.className.toString(),scrollWidth:x.scrollWidth,clientWidth:x.clientWidth,font:getComputedStyle(x).fontSize}))}))()")


def key(client: CDP, name: str, code: str, virtual: int, text: str = "", scripts_enabled: bool = True):
    params = {"key": name, "code": code, "windowsVirtualKeyCode": virtual}
    if text:
        params["text"] = text
    client.call("Input.dispatchKeyEvent", {"type": "keyDown", **params})
    client.call("Input.dispatchKeyEvent", {"type": "keyUp", "key": name, "code": code, "windowsVirtualKeyCode": virtual})
    if scripts_enabled:
        settle(client)


def research_details(client: CDP, scripts_enabled: bool):
    found = client.evaluate("(() => {const specs=[{section:'#detection',marker:'detection',name:'detection-counts',numbers:['360','2880']},"
                            "{section:'#warning',marker:'warning',name:'warning-split',numbers:['14','4']}];return specs.map(spec=>{"
                            "const item=[...document.querySelectorAll(spec.section+' details')].find(x=>x.dataset.protocolDetail===spec.marker);"
                            "if(!item)return {name:spec.name,found:false};item.dataset.qaDisclosure=spec.name;return {name:spec.name,found:true,open:item.open,text:item.textContent};});})()")
    if scripts_enabled:
        check("research counts moved into two closed disclosures", len(found) == 2 and all(item.get("found") and not item.get("open") for item in found), found)
    for item in found:
        label = item["name"]
        if not item.get("found"):
            check(("JavaScript disabled: " if not scripts_enabled else "") + "keyboard opens research disclosure " + label, False, item)
            continue
        selector = "[data-qa-disclosure='" + label + "']"
        client.evaluate("(() => {const detail=document.querySelector(" + json.dumps(selector) + ");detail.open=false;detail.querySelector('summary').focus();})()")
        key(client, " ", "Space", 32, " ", scripts_enabled=scripts_enabled)
        opened = client.evaluate("(() => {const detail=document.querySelector(" + json.dumps(selector) + ");return {open:detail.open,text:detail.innerText,focus:document.activeElement===detail.querySelector('summary')};})()")
        compact = re.sub(r"\s", "", opened["text"])
        numbers = ["360", "2880"] if label == "detection-counts" else ["14", "4"]
        observed_numbers = re.findall(r"\d+", compact)
        check(("JavaScript disabled: " if not scripts_enabled else "") + "keyboard opens research disclosure " + label,
              opened["open"] and opened["focus"] and all(number in observed_numbers for number in numbers), opened)
        if scripts_enabled:
            key(client, " ", "Space", 32, " ")


def run_qa(client: CDP):
    client.call("Page.enable")
    client.call("Runtime.enable")
    client.call("Log.enable")
    client.call("Network.enable")
    client.call("Network.setBlockedURLs", {"urls": ["http://*", "https://*", "ws://*", "wss://*"]})
    client.call("Emulation.setEmulatedMedia", {"features": [{"name": "prefers-reduced-motion", "value": "reduce"}]})
    client.call("Emulation.setFocusEmulationEnabled", {"enabled": True})
    client.call("Emulation.setDeviceMetricsOverride", {"width": 1440, "height": 1000, "deviceScaleFactor": 1, "mobile": False})
    client.call("Page.navigate", {"url": PAGE.as_uri()})
    wait_ready(client)
    client.evaluate("Promise.all([...document.images].map(img=>{img.loading='eager';return img.complete?Promise.resolve():new Promise(r=>{img.addEventListener('load',r,{once:true});img.addEventListener('error',r,{once:true});});}))", True)
    initial = client.evaluate("(() => {const ids=[...document.querySelectorAll('[id]')].map(x=>x.id);return {"
                              "terms:document.querySelectorAll('.glossary-term').length,dataTerms:PROJECT_REPORT_DATA.glossary.terms.length,"
                              "inlineDefinitions:document.querySelectorAll('[data-first-definition]').length,"
                              "duplicates:ids.filter((id,index)=>ids.indexOf(id)!==index),"
                              "images:[...document.images].map(x=>({src:x.getAttribute('src'),loaded:x.complete&&x.naturalWidth>0})),"
                              "financialSVGs:document.querySelectorAll('.line-chart svg').length,"
                              "financialPoints:[...document.querySelectorAll('.line-chart svg')].map(x=>x.querySelectorAll('circle.point').length)}})()")
    check("75 glossary terms rendered", initial["terms"] == initial["dataTerms"] == 75, initial)
    check("first-use definitions in requested 15-25 range", 15 <= initial["inlineDefinitions"] <= 25, initial["inlineDefinitions"])
    check("unique HTML ids", not initial["duplicates"], initial["duplicates"])
    check("local images loaded", all(image["loaded"] for image in initial["images"]), initial["images"])
    check("two financial charts with 12 observations each", initial["financialSVGs"] == 2 and initial["financialPoints"] == [12, 12], initial["financialPoints"])
    ablation = client.evaluate("(() => ({models:PROJECT_REPORT_DATA.forecasting.models.length,rows:PROJECT_REPORT_DATA.forecasting.rows.length,"
                              "metrics:PROJECT_REPORT_DATA.forecasting.ablation.metrics.length,gaps:PROJECT_REPORT_DATA.forecasting.ablation.gap_analysis.length,"
                              "included:PROJECT_REPORT_DATA.forecasting.ablation.included_in_main_benchmark,"
                              "shares:[...document.querySelectorAll('[data-ablation-horizon]')].map(x=>({h:Number(x.dataset.ablationHorizon),text:x.textContent})),"
                              "text:document.getElementById('forecasting-ablation').textContent,"
                              "links:[...document.querySelectorAll('#forecasting-ablation a')].map(x=>x.getAttribute('href'))}))()")
    check("Persistence remains separate from nine main strategies", ablation["models"] == 9 and ablation["rows"] == 72 and ablation["included"] is False, ablation)
    check("Persistence source aggregate counts preserved", ablation["metrics"] == 35 and ablation["gaps"] == 28, {"metrics":ablation["metrics"],"gaps":ablation["gaps"]})
    expected_shares = [{"h":1,"text":"68%"},{"h":3,"text":"69%"},{"h":6,"text":"76%"}]
    check("three rounded descriptive holdout gap shares", ablation["shares"] == expected_shares, ablation["shares"])
    check("Persistence caveats and exact comparison visible", all(s in ablation["text"] for s in ["LightGBMDirect", "National/Local + LightGBM", "SeasonalNaiveYoY", "validation", "Holdout уже просмотрен", "h = 6", "h = 12", "не полностью устойчивы"])
          and "causal contribution" not in ablation["text"], None)
    check("Persistence public source links are relative and local", ablation["links"] == ["references/national-local-persistence.html", "references/persistence/gap_analysis.csv"]
          and all((PAGE.parent / path).resolve().is_file() for path in ablation["links"]), ablation["links"])
    robustness = client.evaluate("(() => {const b=PROJECT_REPORT_DATA.forecasting.robustness;const d=document.getElementById('forecasting-robustness');return {"
        "keys:b.primary_pair.records.map(r=>r.split+':'+r.horizon),"
        "rows:[...d.querySelectorAll('tbody tr')].map(x=>[...x.children].map(t=>t.textContent)),"
        "text:d.textContent,includesZero:b.primary_pair.records.filter(r=>r.split==='holdout').map(r=>r.origin_bootstrap.includes_zero),"
        "validationCI:b.primary_pair.records.find(r=>r.split==='validation'&&r.horizon===6).origin_bootstrap}})()")
    check("Robustness has separate six split/horizon rows", sorted(robustness['keys']) == sorted(s+':'+str(h) for s in ['validation','holdout'] for h in [1,3,6]), robustness['keys'])
    check("Three saved holdout diagnostics rendered", [r[:4] for r in robustness['rows']] == [['h = 1','121,15','55,56%','5/6'],['h = 3','27,68','52,38%','4/6'],['h = 6','8,79','57,14%','3/6']], robustness['rows'])
    check("Robustness interval text and zero inclusion", robustness['includesZero']==[False,True,True] and [r[5] for r in robustness['rows']]==['Интервал выше нуля','Интервал включает ноль','Интервал включает ноль'], robustness['rows'])
    check("One-origin validation CI remains null", robustness['validationCI']['lower'] is None and robustness['validationCI']['upper'] is None, robustness['validationCI'])
    check("Robustness caution visible", 'На validation преимущество не воспроизводится' in robustness['text'] and 'описательную оценку' in robustness['text'], None)
    client.evaluate("location.hash='forecasting-robustness'")
    settle(client)
    check("Robustness anchor navigation", client.evaluate("location.hash==='#forecasting-robustness'&&!!document.getElementById('forecasting-robustness')"))
    visible = client.evaluate("(() => {const walk=document.createTreeWalker(document.body,NodeFilter.SHOW_TEXT);let t='',n;while(n=walk.nextNode()){if(!n.parentElement.closest('script,style'))t+=' '+n.textContent;}const attrs=[...document.querySelectorAll('[alt],[title],[aria-label]')].flatMap(x=>['alt','title','aria-label'].map(a=>x.getAttribute(a)||''));return {text:t,attrs:attrs.join(' '),svgTitles:[...document.querySelectorAll('.line-chart svg title')].map(x=>x.textContent)};})()")
    leaked = re.findall(r'\b(?:E\d{1,2}[a-d]?|F[02345678]|S[0-3]|K[0-2]|M[0-2]|C0|L0|CN|LN)\b', visible['text']+' '+visible['attrs'])
    check('no internal experiment or bare variant IDs in HTML text and accessibility labels', not leaked, {'leaked': leaked, 'svgTitles': visible['svgTitles']})
    check('synthetic table uses four meaningful strategy names', all(label in visible['text'] for label in ['Constant risk','History only','History + detector state','History + detector state + external precursors']), None)
    from build_publication_site import Inventory
    original = (ROOT / 'docs/index.html').read_text(encoding='utf-8-sig')
    original = re.sub(r'<!-- forecasting-robustness:begin -->.*?<!-- forecasting-robustness:end -->', '', original, flags=re.DOTALL)
    baseline = {'cells': [re.sub(r'\s+', ' ', x).strip() for x in Inventory(original).cells]}
    table_cells = client.evaluate("[...document.querySelectorAll('table th,table td')].filter(x=>!x.closest('#forecasting-robustness')).map(x=>x.textContent.replace(/\\s+/g,' ').trim())")
    check("246 original table cells unchanged by Persistence integration", len(table_cells) == 246 and table_cells == baseline["cells"], {"cells":len(table_cells)})
    main_prose = client.evaluate("(() => {const main=document.querySelector('main').cloneNode(true);main.querySelectorAll('#glossary,table,details,script,style').forEach(x=>x.remove());return main.textContent;})()")
    phrases = ["Future observations", "Offline segmentation", "Fixed ablation", "forecasting uplift", "predictive effect",
               "incremental predictive value", "financial indicators не оценивается", "Permutation inference",
               "train positives", "test positives", "News-архив", "vintages расходов", "полностью используют fallback",
               "Detection methods", "Early-warning evaluation"]
    bad_phrases = [phrase for phrase in phrases if phrase.casefold() in main_prose.casefold()]
    check("no identified English/Russian calques in main prose", not bad_phrases, bad_phrases)
    client.evaluate("(() => {const item=[...document.querySelectorAll('#financial p')].find(x=>x.textContent.includes('фиксированн'));if(item)item.dataset.qaFinancialProse='true';})()")


    forecast = []
    for split in ["holdout", "validation"]:
        for horizon in [1, 3, 6, 12]:
            row = client.evaluate("(() => {const s=document.getElementById('forecast-split');s.value=" + json.dumps(split) +
                                  ";s.dispatchEvent(new Event('change',{bubbles:true}));document.querySelector('[data-horizon=\"" + str(horizon) +
                                  "\"]').click();const source=PROJECT_REPORT_DATA.forecasting.rows.filter(x=>x.split===" + json.dumps(split) +
                                  "&&x.horizon===" + str(horizon) + ");const finite=source.filter(x=>Number.isFinite(x.mae_macro));return {"
                                  "split:s.value,horizon:" + str(horizon) + ",bars:document.querySelectorAll('#forecast-chart .bar-row').length,"
                                  "expected:finite.length,values:[...document.querySelectorAll('#forecast-chart .bar-value')].map(x=>x.textContent),"
                                  "exact:finite.map(x=>x.mae_macro),status:document.getElementById('forecast-status').textContent,"
                                  "emptyText:document.getElementById('forecast-chart').textContent,"
                                  "pressed:document.querySelector('[data-horizon=\"" + str(horizon) + "\"]').getAttribute('aria-pressed'),"
                                  "fallbackLabels:document.querySelectorAll('#forecast-chart small').length,expectedFallbacks:finite.filter(x=>x.fallback_count>0).length};})()")
            matches = all(abs(float(re.sub(r"[^0-9,.-]", "", text).replace(",", ".")) - exact) <= 0.0051
                          for text, exact in zip(row["values"], row["exact"]))
            expected = 0 if split == "validation" and horizon == 12 else 9
            good = row["bars"] == row["expected"] == expected and row["pressed"] == "true" and matches
            if horizon == 12 and split == "holdout":
                good = good and row["fallbackLabels"] == row["expectedFallbacks"] == 4
            if horizon == 12 and split == "validation":
                good = good and "не заменяется нулём" in row["emptyText"]
            check(f"forecast {split} h={horizon}", good, row)
            forecast.append(row)

    detection = []
    for metric in ["f1", "precision", "recall", "false_alarms_per_12"]:
        result = client.evaluate("(() => {const s=document.getElementById('detection-metric');s.value=" + json.dumps(metric) +
                                 ";s.dispatchEvent(new Event('change',{bubbles:true}));return {metric:s.value,"
                                 "bars:document.querySelectorAll('#detection-chart .bar-row').length,"
                                 "values:[...document.querySelectorAll('#detection-chart .bar-value')].map(x=>x.textContent),"
                                 "exact:PROJECT_REPORT_DATA.detection.rows.map(x=>x[" + json.dumps("false_positives_per_12_months" if metric == "false_alarms_per_12" else metric) + "]),"
                                 "title:document.getElementById('detection-chart-title').textContent};})()")
        matches = all(abs(float(text.replace(",", ".")) - exact) <= 0.00051 for text, exact in zip(result["values"], result["exact"]))
        check("detection " + metric, result["bars"] == 5 and matches, result)
        detection.append(result)

    for k, positive, unknown in [(1, 6, 6), (3, 4, 8)]:
        result = client.evaluate("(() => {document.querySelector('[data-warning-window=\"" + str(k) +
                                 "\"]').click();const rows=[...document.querySelectorAll('#warning-cohort .cohort-date')];return {"
                                 "total:rows.length,positive:rows.filter(x=>x.dataset.status==='positive').length,"
                                 "negative:rows.filter(x=>x.dataset.status==='negative').length,unknown:rows.filter(x=>x.dataset.status==='unknown').length,"
                                 "status:document.getElementById('cohort-status').textContent};})()")
        check(f"early-warning cohort k={k}", result["total"] == 12 and result["positive"] == positive and result["negative"] == 0 and result["unknown"] == unknown, result)

    def search(query: str):
        return client.evaluate("(() => {const s=document.getElementById('glossary-search');s.value=" + json.dumps(query) +
                               ";s.dispatchEvent(new Event('input',{bubbles:true}));return {visible:[...document.querySelectorAll('.glossary-term')].filter(x=>!x.hidden).map(x=>x.id),"
                               "count:document.getElementById('glossary-count').textContent,empty:!document.getElementById('glossary-empty').hidden};})()")

    client.evaluate("document.getElementById('glossary-reset').click()")
    result = search("forecast horizon")
    check("glossary English search", "forecast-horizon" in result["visible"] and len(result["visible"]) < 75, result)
    result = search("прогнозирование")
    check("glossary Russian search", "forecasting" in result["visible"] and 0 < len(result["visible"]) < 75, result)
    result = search("zzzzzzz-no-term")
    check("glossary empty search", not result["visible"] and result["empty"], result)
    client.evaluate("document.getElementById('glossary-reset').click()")
    group = client.evaluate("(() => {const s=document.getElementById('glossary-group');s.value=PROJECT_REPORT_DATA.glossary.groups[0].id;s.dispatchEvent(new Event('change',{bubbles:true}));return {"
                            "visible:[...document.querySelectorAll('.glossary-term')].filter(x=>!x.hidden).map(x=>x.id),expected:PROJECT_REPORT_DATA.glossary.groups[0].term_ids,"
                            "groups:[...document.querySelectorAll('.glossary-group')].filter(x=>!x.hidden).length};})()")
    check("glossary thematic group filter", group["visible"] == group["expected"] and group["groups"] == 1, group)
    reset = client.evaluate("(() => {document.getElementById('glossary-reset').click();return {visible:[...document.querySelectorAll('.glossary-term')].filter(x=>!x.hidden).length,"
                            "query:document.getElementById('glossary-search').value,group:document.getElementById('glossary-group').value,focus:document.activeElement.id};})()")
    check("glossary reset and input focus", reset == {"visible": 75, "query": "", "group": "", "focus": "glossary-search"}, reset)
    search("zzzzzzz-no-term")
    for term in ["pipeline", "forecast-origin", "mae"]:
        client.evaluate("location.hash=" + json.dumps(term))
        settle(client)
        result = client.evaluate("(() => {const target=document.getElementById(" + json.dumps(term) +
                                 ");return {id:target.id,open:target.open,hidden:target.hidden,focus:document.activeElement===target.querySelector('summary'),"
                                 "query:document.getElementById('glossary-search').value,visible:[...document.querySelectorAll('.glossary-term')].filter(x=>!x.hidden).length};})()")
        check("glossary hash reveal " + term, result["open"] is True and not result["hidden"] and result["focus"] and result["query"] == "" and result["visible"] == 75, result)

    client.evaluate("document.querySelector('[data-horizon=\"3\"]').focus()")
    key(client, "Enter", "Enter", 13, "\r")
    pressed = client.evaluate("document.querySelector('[data-horizon=\"3\"]').getAttribute('aria-pressed')")
    check("keyboard Enter activates horizon control", pressed == "true", pressed)
    key(client, "Tab", "Tab", 9)
    focus = client.evaluate("({horizon:document.activeElement.dataset.horizon,visible:document.activeElement.matches(':focus-visible')})")
    check("keyboard Tab order and visible focus", focus.get("horizon") == "6" and focus.get("visible"), focus)
    client.evaluate("document.getElementById('national-local-decomposition').querySelector('summary').focus()")
    was_open = client.evaluate("document.getElementById('national-local-decomposition').open")
    key(client, " ", "Space", 32, " ")
    now_open = client.evaluate("document.getElementById('national-local-decomposition').open")
    check("keyboard Space toggles long glossary summary", was_open != now_open, {"before": was_open, "after": now_open})
    research_details(client, True)

    themes = []
    for _ in range(2):
        result = client.evaluate("(() => {document.getElementById('theme-toggle').click();return {theme:document.documentElement.dataset.theme,"
                                 "pressed:document.getElementById('theme-toggle').getAttribute('aria-pressed'),background:getComputedStyle(document.body).backgroundColor,color:getComputedStyle(document.body).color};})()")
        themes.append(result)
    check("light/dark toggle", {x["theme"] for x in themes} == {"light", "dark"} and all(x["pressed"] == str(x["theme"] == "dark").lower() for x in themes), themes)
    client.evaluate("document.documentElement.dataset.theme='light';document.getElementById('glossary-reset').click();location.hash='';document.querySelector('[data-horizon=\"1\"]').click();document.getElementById('forecast-split').value='holdout';document.getElementById('forecast-split').dispatchEvent(new Event('change',{bubbles:true}));")
    desktop = viewport(client, 1440, 1000, False)
    check("desktop 1440 page overflow", desktop["scrollWidth"] <= 1441 and desktop["bodyWidth"] <= 1441, desktop)
    screenshot(client, "desktop_hero.png")
    screenshot(client, "desktop_results.png", "results")
    screenshot(client, "desktop_persistence_ablation.png", "forecasting-ablation")
    screenshot(client, "desktop_robustness.png", "forecasting-robustness")
    screenshot(client, "desktop_forecast_chart.png", selector="#forecast-chart-title")
    screenshot(client, "desktop_financial_heading.png", "financial")
    screenshot(client, "desktop_financial.png", selector=".financial-chart-grid")
    screenshot(client, "desktop_financial_prose.png", selector="[data-qa-financial-prose]")
    client.evaluate("document.querySelector('[data-warning-window=\"1\"]').click()")
    screenshot(client, "desktop_warning.png", selector="#warning .chart-toolbar")
    screenshot(client, "desktop_early_warning_prose.png", selector="#warning .margin-note")
    client.evaluate("document.querySelector('#warning .synthetic-subsection').open=true")
    screenshot(client, "desktop_synthetic_details.png", selector="#warning .synthetic-subsection")
    client.evaluate("document.querySelector('#warning .synthetic-subsection').open=false")
    screenshot(client, "desktop_conclusions.png", "conclusions")
    client.evaluate("document.getElementById('theme-toggle').click()")
    screenshot(client, "desktop_dark_hero.png")
    screenshot(client, "desktop_dark_robustness.png", "forecasting-robustness")
    client.evaluate("document.getElementById('theme-toggle').click()")
    mobile = viewport(client, 390, 844, True)
    check("mobile 390 page overflow", mobile["width"] == 390 and mobile["scrollWidth"] <= 391 and mobile["bodyWidth"] <= 391, mobile)
    screenshot(client, "mobile_hero.png")
    screenshot(client, "mobile_results.png", "results")
    screenshot(client, "mobile_persistence_ablation.png", "forecasting-ablation")
    screenshot(client, "mobile_robustness.png", "forecasting-robustness")
    screenshot(client, "mobile_financial_heading.png", "financial")
    screenshot(client, "mobile_financial_prose.png", selector="[data-qa-financial-prose]")
    screenshot(client, "mobile_warning.png", "warning")
    screenshot(client, "mobile_early_warning_prose.png", selector="#warning .margin-note")
    client.evaluate("document.querySelector('#warning .synthetic-subsection').open=true")
    screenshot(client, "mobile_synthetic_details.png", selector="#warning .synthetic-subsection")
    screenshot(client, "mobile_synthetic_table.png", selector="#warning .synthetic-subsection .table-scroll")
    client.evaluate("document.querySelector('#warning .synthetic-subsection').open=false")
    screenshot(client, "mobile_conclusions.png", "conclusions")
    screenshot(client, "mobile_glossary.png", "glossary")
    narrow = viewport(client, 320, 844, True)
    check("mobile 320 page overflow", narrow["width"] == 320 and narrow["scrollWidth"] <= 321 and narrow["bodyWidth"] <= 321, narrow)
    screenshot(client, "mobile_320_hero.png")
    screenshot(client, "mobile_320_persistence_ablation.png", "forecasting-ablation")
    screenshot(client, "mobile_320_robustness.png", "forecasting-robustness")
    viewport(client, 390, 844, True)

    client.call("Emulation.setScriptExecutionDisabled", {"value": True})
    client.call("Page.navigate", {"url": PAGE.as_uri() + "?no-js-qa=1"})
    wait_ready(client, False)
    nojs = client.evaluate("(() => ({interactive:document.documentElement.classList.contains('is-interactive'),"
                          "tables:document.querySelectorAll('table').length,tableRows:[...document.querySelectorAll('table')].map(x=>x.querySelectorAll('tbody tr').length),"
                          "forecastTableOpen:document.querySelector('.data-table-disclosure').open,glossary:document.querySelectorAll('.glossary-term').length,"
                          "controls:[...document.querySelectorAll('.enhanced-control')].filter(x=>getComputedStyle(x).display!=='none').length,"
                          "scrollWidth:document.documentElement.scrollWidth,width:innerWidth}))()")
    check("JavaScript disabled: static tables and glossary", not nojs["interactive"] and nojs["tables"] >= 4 and nojs["forecastTableOpen"] and nojs["glossary"] == 75 and nojs["controls"] == 0 and nojs["scrollWidth"] <= nojs["width"] + 1, nojs)
    static_shares = client.evaluate("[...document.querySelectorAll('[data-ablation-horizon]')].map(x=>({h:Number(x.dataset.ablationHorizon),text:x.textContent}))")
    check("JavaScript disabled: Persistence shares remain readable", static_shares == expected_shares, static_shares)
    regions = client.evaluate("[...document.querySelectorAll('.table-scroll')].map(x=>({tabIndex:x.tabIndex,role:x.getAttribute('role'),label:x.getAttribute('aria-label')}))")
    check("table regions keyboard focus and accessible names", len(regions) == 6 and all(x["tabIndex"] == 0 and x["role"] == "region" and x["label"] for x in regions), regions)
    client.evaluate("(() => {const region=document.querySelector('.table-scroll');region.scrollLeft=0;region.focus();})()")
    key(client, "ArrowRight", "ArrowRight", 39, scripts_enabled=False)
    time.sleep(0.2)
    table_scroll = client.evaluate("(() => {const region=document.querySelector('.table-scroll');return {focused:document.activeElement===region,left:region.scrollLeft,client:region.clientWidth,scroll:region.scrollWidth};})()")
    check("JavaScript disabled: keyboard scrolls mobile table", table_scroll["focused"] and table_scroll["scroll"] > table_scroll["client"] and table_scroll["left"] > 0, table_scroll)
    client.evaluate("document.getElementById('national-local-decomposition').querySelector('summary').focus()")
    key(client, " ", "Space", 32, " ", scripts_enabled=False)
    native = client.evaluate("document.getElementById('national-local-decomposition').open")
    check("JavaScript disabled: keyboard opens long native glossary summary", native is True, native)
    research_details(client, False)
    client.evaluate("(() => {const detail=document.querySelector('#warning .synthetic-subsection');detail.open=false;detail.querySelector('summary').focus();})()")
    key(client, " ", "Space", 32, " ", scripts_enabled=False)
    client.evaluate("(() => {const region=document.querySelector('#warning .synthetic-subsection .table-scroll');region.scrollLeft=0;region.focus();})()")
    key(client, "ArrowRight", "ArrowRight", 39, scripts_enabled=False)
    time.sleep(0.2)
    synthetic_scroll = client.evaluate("(() => {const detail=document.querySelector('#warning .synthetic-subsection');const region=detail.querySelector('.table-scroll');return {open:detail.open,focused:document.activeElement===region,left:region.scrollLeft,client:region.clientWidth,scroll:region.scrollWidth};})()")
    check("JavaScript disabled: keyboard opens and scrolls long-name synthetic table", synthetic_scroll["open"] and synthetic_scroll["focused"] and synthetic_scroll["scroll"] > synthetic_scroll["client"] and synthetic_scroll["left"] > 0, synthetic_scroll)
    client.call("Emulation.setScriptExecutionDisabled", {"value": False})
    screenshot(client, "mobile_nojs_glossary.png", "glossary")
    screenshot(client, "mobile_nojs_detection_counts.png", selector="[data-qa-disclosure='detection-counts']")
    screenshot(client, "mobile_nojs_warning_split.png", selector="[data-qa-disclosure='warning-split']")
    screenshot(client, "mobile_nojs_persistence_ablation.png", "forecasting-ablation")
    screenshot(client, "mobile_nojs_robustness.png", "forecasting-robustness")
    screenshot(client, "mobile_nojs_synthetic_table.png", selector="#warning .synthetic-subsection .table-scroll")
    client.evaluate("true")


def reference_qa(client: CDP):
    """Open the isolated copies, with all external requests still blocked."""
    client.call('Emulation.setScriptExecutionDisabled', {'value': False})
    pages = sorted((PAGE.parent / 'references').glob('*.html'))
    for path in pages:
        client.call('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
        client.call('Page.navigate', {'url': path.as_uri()})
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            if client.evaluate("document.readyState==='complete'&&!!document.querySelector('main h1')"):
                break
            time.sleep(.05)
        client.evaluate("Promise.all([...document.images].map(img=>{img.loading='eager';return img.complete?Promise.resolve():new Promise(r=>{img.onload=r;img.onerror=r;});}))", True)
        result = client.evaluate("({heading:!!document.querySelector('main h1'),images:[...document.images].map(i=>i.complete&&i.naturalWidth>0),anchors:[...document.querySelectorAll('a[href^=\"#\"]')].every(a=>!!document.getElementById(decodeURIComponent(a.hash.slice(1)))),width:document.documentElement.scrollWidth})")
        check('anonymous local document ' + path.name, result['heading'] and all(result['images']) and result['anchors'] and result['width'] <= 1441, result)
        if path.name in {'methodology.html', 'glossary.html'}:
            screenshot(client, 'reference_' + path.stem + '_desktop.png')
            for width in [390, 320]:
                for theme in ['light', 'dark']:
                    client.call('Emulation.setDeviceMetricsOverride', {'width': width, 'height': 844, 'deviceScaleFactor': 1, 'mobile': True})
                    client.call('Emulation.setEmulatedMedia', {'features': [{'name': 'prefers-color-scheme', 'value': theme}]})
                    settle(client)
                    dimensions = client.evaluate("({width:innerWidth,scroll:document.documentElement.scrollWidth,background:getComputedStyle(document.body).backgroundColor})")
                    check(f'{path.stem} mobile {width} {theme}', dimensions['width'] == width and dimensions['scroll'] <= width + 1, dimensions)
            if path.name == 'methodology.html':
                counts = client.evaluate("({tables:document.querySelectorAll('main table').length,formulas:document.querySelectorAll('main .math-block').length,figures:document.querySelectorAll('main img:not([data-latex])').length})")
                check('methodology retains 12 tables, six formulas, nine figures', counts == {'tables': 12, 'formulas': 6, 'figures': 9}, counts)
                screenshot(client, 'reference_methodology_mobile_320_dark.png')
                client.evaluate("(() => {let d=document.querySelector('.math-source');d.open=false;d.querySelector('summary').focus();})()")
                key(client, ' ', 'Space', 32, ' ')
                check('keyboard reveals original LaTeX', client.evaluate("({open:document.querySelector('.math-source').open,focus:document.activeElement===document.querySelector('.math-source summary')})").get('open'))
            else:
                count = client.evaluate("document.querySelectorAll('main a[id]').length")
                check('local canonical glossary has 75 stable term anchors', count == 75, count)
    client.call('Emulation.setDeviceMetricsOverride', {'width': 1440, 'height': 1000, 'deviceScaleFactor': 1, 'mobile': False})
    pdf = PAGE.parent / 'presentation/presentation.pdf'
    navigation = client.call('Page.navigate', {'url': pdf.as_uri()})
    time.sleep(1)
    pdf_state = client.evaluate("({url:location.href,type:document.contentType,viewer:!!document.querySelector('embed[type=\"application/pdf\"]')})")
    check('local PDF opens without GitHub', not navigation.get('errorText') and pdf_state['url'] == pdf.as_uri() and (pdf_state['type'] == 'application/pdf' or pdf_state['viewer']), pdf_state)
    screenshot(client, 'local_presentation_pdf.png')


def main():
    global PAGE
    sys.stdout.reconfigure(encoding="utf-8")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not EDGE.is_file() or not PAGE.is_file():
        raise RuntimeError("Existing Edge and completed local HTML are required")
    isolation = tempfile.TemporaryDirectory(prefix='sberindex-publication-qa-')
    isolated_site = Path(isolation.name) / 'site'
    shutil.copytree(PAGE.parent, isolated_site)
    PAGE = isolated_site / 'index.html'
    profile = OUTPUT / ("browser_profile_" + uuid.uuid4().hex[:12])
    process = None
    client = None
    started = time.monotonic()
    initial_sha256 = {name: hashlib.sha256((PAGE.parent / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
    failure = None
    flags = [
        str(EDGE), "--headless", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
        "--disable-extensions", "--disable-background-networking", "--disable-component-update",
        "--disable-sync", "--metrics-recording-only", "--disable-breakpad", "--disable-crash-reporter",
        "--no-sandbox", "--disable-features=RendererCodeIntegrity", "--proxy-server=127.0.0.1:9",
        "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=0", "--user-data-dir=" + str(profile), "about:blank",
    ]
    try:
        with (OUTPUT / "browser_stdout.log").open("wb") as stdout, (OUTPUT / "browser_stderr.log").open("wb") as stderr:
            process = subprocess.Popen(flags, stdout=stdout, stderr=stderr, creationflags=subprocess.CREATE_NO_WINDOW)
        active = profile / "DevToolsActivePort"
        deadline = time.monotonic() + 20
        while not active.is_file() and time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError("Headless browser exited before local debugger became ready")
            time.sleep(0.05)
        if not active.is_file():
            raise TimeoutError("Local Edge debugger startup")
        port = int(active.read_text().splitlines()[0])
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open("http://127.0.0.1:" + str(port) + "/json/list", timeout=10) as response:
            pages = json.load(response)
        page = next(item for item in pages if item["type"] == "page")
        client = CDP(page["webSocketDebuggerUrl"])
        run_qa(client)
        reference_qa(client)
    except Exception as error:
        failure = type(error).__name__ + ": " + str(error)
        check("browser QA execution", False, failure)
    finally:
        events = client.events if client else []
        exceptions = [event for event in events if event.get("method") == "Runtime.exceptionThrown"]
        console_errors = [event for event in events if event.get("method") == "Runtime.consoleAPICalled" and event.get("params", {}).get("type") == "error"]
        log_errors = [event for event in events if event.get("method") == "Log.entryAdded" and event.get("params", {}).get("entry", {}).get("level") == "error"]
        external = [event.get("params", {}).get("request", {}).get("url") for event in events
                    if event.get("method") == "Network.requestWillBeSent"
                    and urllib.parse.urlsplit(event.get("params", {}).get("request", {}).get("url", "")).scheme in {"http", "https", "ws", "wss"}]
        check("no uncaught JavaScript exceptions", not exceptions, exceptions)
        check("no console.error messages", not console_errors, console_errors)
        check("no browser page error logs", not log_errors, log_errors)
        check("no external resource requests", not external, external)
        if client:
            try:
                client.call("Browser.close", timeout=3)
            except (EOFError, OSError, TimeoutError):
                pass
            client.ws.close()
        if process:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait(timeout=5)
        final_sha256 = {name: hashlib.sha256((PAGE.parent / name).read_bytes()).hexdigest() for name in SOURCE_FILES}
        check("source files unchanged throughout browser QA", initial_sha256 == final_sha256)
        proof = {
            "task": "Isolated publication package browser QA", "status": "PASS" if all(item["passed"] for item in checks) else "FAIL",
            "runtime_seconds": round(time.monotonic() - started, 3), "source": "isolated publication package/index.html",
            "source_sha256": final_sha256, "checks": checks, "screenshots": screenshots,
            "connection": "Local Edge CDP over 127.0.0.1, stdlib WebSocket; HTTP/HTTPS/WS/WSS page requests blocked",
            "page_edits": 0, "new_model_fits": 0, "new_experiments": 0, "package_installs": 0,
            "presentation_export": False, "commit_push": False,
        }
        (OUTPUT / "frontend_browser_check.json").write_text(json.dumps(proof, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": proof["status"], "runtime_seconds": proof["runtime_seconds"], "checks": len(checks),
                          "failures": [item for item in checks if not item["passed"]], "screenshots": screenshots}, ensure_ascii=False, indent=2))
        isolation.cleanup()
    return 0 if all(item["passed"] for item in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
