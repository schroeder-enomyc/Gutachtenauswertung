# Datei: gutachten_responses.py
import os, sys, time, re, json
from pathlib import Path
from typing import Any, Dict, List, Set
from dotenv import load_dotenv
from openai import OpenAI

# --------------------------
# Einstellungen
# --------------------------
VS_ID_FILE = Path(".gutachten_vs_id")
DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")
ENABLE_SCHEMA = True  # JSON-Schema-Ausgabe erzwingen (empfohlen)

SYSTEM = (
    "Du bist Unternehmensberater. Antworte NUR auf Basis der File-Search-Wissensbasis.\n\n"
    "Lieferformat (intern): Du erzeugst zunächst eine strukturierte JSON-Antwort gemäss response_format.\n"
    "Inhaltlich:\n"
    "- Gliedere nach Abteilungen (aus den Daten). Für JEDE Maßnahme liefere: title (Kurzname), rationale (1 Satz), "
    "und source {unternehmen, jahr, abteilung} – NUR aus den zulässigen Werten (Whitelist).\n"
    "- Wenn eine Info nicht in den Dateien steht oder die Quelle unklar ist, gib die Maßnahme NICHT aus.\n"
    "- Keine Dateinamen/Pfade als Quelle. Nur die Felder unternehmen/jahr/abteilung aus der Wissensbasis.\n"
)

# --------------------------
# Hilfsfunktionen: Vector Store
# --------------------------
def die(msg: str, code: int = 1):
    print(msg)
    sys.exit(code)

def load_or_create_vector_store(client: OpenAI, vs_id_path: Path) -> str:
    if vs_id_path.exists():
        vs_id = vs_id_path.read_text().strip()
        if vs_id:
            try:
                client.vector_stores.retrieve(vs_id)
                print(f"♻️  Reuse Vector Store: {vs_id}")
                return vs_id
            except Exception:
                print("ℹ️  gespeicherter Vector Store nicht auffindbar – erstelle neu.")
    vs = client.vector_stores.create(name="gutachten-vs")
    vs_id_path.write_text(vs.id)
    print(f"🆕 Vector Store erstellt: {vs.id}")
    return vs.id

def upload_files(client: OpenAI, paths: List[str]) -> List[str]:
    file_ids = []
    for raw in paths:
        p = Path(raw)
        if not p.exists():
            print(f"⚠️  Datei nicht gefunden: {p}")
            continue
        with p.open("rb") as f:
            up = client.files.create(file=f, purpose="assistants")
        file_ids.append(up.id)
    return file_ids

def attach_and_index_files(client: OpenAI, vs_id: str, file_ids: List[str]) -> None:
    if not file_ids:
        die("Keine Datei erfolgreich hochgeladen – bitte Pfade prüfen.")
    batch = client.vector_stores.file_batches.create(
        vector_store_id=vs_id,
        file_ids=file_ids
    )
    while True:
        b = client.vector_stores.file_batches.retrieve(
            vector_store_id=vs_id,
            batch_id=batch.id
        )
        status = getattr(b, "status", None)
        counts = getattr(b, "file_counts", None)
        completed = getattr(counts, "completed", None)
        failed = getattr(counts, "failed", None)
        in_progress = getattr(counts, "in_progress", None)
        total = getattr(counts, "total", None)
        if completed is None and hasattr(counts, "model_dump"):
            d = counts.model_dump()
            completed = d.get("completed")
            failed = d.get("failed")
            in_progress = d.get("in_progress")
            total = d.get("total")
        if status:
            print(f"Indexierung: {status}"
                  + (f" | total={total}, completed={completed}, in_progress={in_progress}, failed={failed}"
                     if total is not None else ""))
        if status in ("completed", "failed", "canceled"):
            if status != "completed" or (failed and failed > 0):
                die("Indexierung nicht erfolgreich – bitte Dateien prüfen.")
            break
        time.sleep(1)

# --------------------------
# Whitelist aus JSON-Dateien extrahieren
# --------------------------
def extract_whitelists_from_json(paths: List[str]) -> Dict[str, Set[Any]]:
    unternehmen: Set[str] = set()
    jahre: Set[int] = set()
    abteilungen: Set[str] = set()
    # Wir versuchen nur lokale JSONs zu parsen, die du sowieso an den Vector Store anhängst
    for raw in paths:
        if not raw.lower().endswith(".json"):
            continue
        p = Path(raw)
        if not p.exists():
            continue
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        # datenbasis kann Liste von Objekten sein
        if isinstance(data, list):
            for entry in data:
                if not isinstance(entry, dict):
                    continue
                u = entry.get("unternehmen")
                if isinstance(u, str) and u.strip():
                    unternehmen.add(u.strip())
                j = entry.get("jahr")
                if isinstance(j, int):
                    jahre.add(j)
                # abteilungen tief drin
                details = entry.get("details", {})
                if isinstance(details, dict):
                    haupt = details.get("haupt- und unterstützungsprozesse")
                    if isinstance(haupt, list):
                        for dep in haupt:
                            if isinstance(dep, dict):
                                abt = dep.get("abteilung")
                                if isinstance(abt, str) and abt.strip():
                                    abteilungen.add(abt.strip())
        # branchen.json interessiert uns hier nicht für Quellen
    return {
        "unternehmen": unternehmen,
        "jahre": jahre,
        "abteilungen": abteilungen,
    }

def build_json_schema(whitelist: Dict[str, Set[Any]]) -> Dict[str, Any]:
    # Enums aus Whitelist bauen; Leere vermeiden (falls leer, kein enum setzen)
    u_list = sorted(list(whitelist.get("unternehmen", [])))
    j_list = sorted(list(whitelist.get("jahre", [])))
    a_list = sorted(list(whitelist.get("abteilungen", [])))

    def enum_or_type(tp, enum_values):
        # Hilfskonstruktor: Wenn enum_values leer, kein enum vorgeben
        base = {"type": tp}
        if enum_values:
            base["enum"] = enum_values
        return base

    schema = {
        "name": "MeasuresResponse",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "departments": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "department": enum_or_type("string", a_list),
                            "measures": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "title": {"type": "string"},
                                        "rationale": {"type": "string"},
                                        "source": {
                                            "type": "object",
                                            "properties": {
                                                "unternehmen": enum_or_type("string", u_list),
                                                "jahr": enum_or_type("integer", j_list),
                                                "abteilung": enum_or_type("string", a_list),
                                            },
                                            "required": ["unternehmen", "jahr", "abteilung"],
                                        },
                                    },
                                    "required": ["title", "rationale", "source"],
                                },
                            },
                        },
                        "required": ["department", "measures"],
                    },
                }
            },
            "required": ["departments"],
        },
    }
    return schema

# --------------------------
# Rendering / Validierung
# --------------------------
INLINE_SRC_RE = re.compile(r"\[(Quelle:\s*[^\]]+|Allgemeinwissen)\]")

def render_structured_json_to_markdown(data: Dict[str, Any]) -> str:
    # Erwartet das JSON gemäss Schema und rendert Markdown
    lines: List[str] = []
    for dep in data.get("departments", []):
        department = dep.get("department")
        if not department:
            # fallback
            department = "Abteilung"
        lines.append(f"### {department}")
        for m in dep.get("measures", []):
            title = m.get("title", "").strip()
            rationale = m.get("rationale", "").strip()
            src = m.get("source", {}) or {}
            u = src.get("unternehmen", "")
            j = src.get("jahr", "")
            a = src.get("abteilung", department)
            lines.append(f"- **{title}** – {rationale} [Quelle: {u}/{j}/{a}].")
        lines.append("")  # Leerzeile
    return "\n".join(lines).strip()

def parse_json_output_text(text: str) -> Dict[str, Any]:
    # Robust gegen Extra-Whitespace / Markdown-Fences
    s = text.strip()
    # Entferne evtl. Codefences
    if s.startswith("```"):
        s = s.strip("`")
        # nach erster Zeile evtl. 'json'
        s = "\n".join(s.splitlines()[1:])
    return json.loads(s)

def warn_if_missing_inline_sources(text: str) -> None:
    offending = []
    for line in text.splitlines():
        if line.strip().startswith(("-", "*")) or re.match(r"^\s*\d+\.", line):
            if not INLINE_SRC_RE.search(line):
                offending.append(line.strip())
    if offending:
        print("\n⚠️  Einige Maßnahmen haben KEINE Inline-Quelle/Allgemeinwissen-Tag:")
        for l in offending[:5]:
            print(f"   · {l}")
        if len(offending) > 5:
            print(f"   … und {len(offending)-5} weitere.")

FILENAME_IN_SRC_RE = re.compile(r"\[Quelle:[^\]]*\.(json|csv|txt|pdf|docx)\b", re.IGNORECASE)
def warn_if_filename_in_source(text: str) -> None:
    bad = []
    for line in text.splitlines():
        m = FILENAME_IN_SRC_RE.search(line)
        if m:
            bad.append(line.strip())
    if bad:
        print("\n⚠️  Quellen-Format fehlerhaft (Dateiname erkannt). Nur {unternehmen}/{jahr}/{abteilung} verwenden:")
        for l in bad[:5]:
            print(f"   · {l}")
        if len(bad) > 5:
            print(f"   … und {len(bad)-5} weitere.")

def render_with_citations(resp: object, client: OpenAI) -> str:
    # Fallback-Renderer (falls schema disabled / parsing fehlgeschlagen)
    out_text = getattr(resp, "output_text", None)
    if not out_text:
        parts = []
        try:
            for item in getattr(resp, "output", []) or []:
                if getattr(item, "type", "") == "message":
                    for c in getattr(item, "content", []) or []:
                        if getattr(c, "type", "") in ("output_text", "text"):
                            txt = getattr(c, "text", "")
                            if not txt and hasattr(c, "text") and hasattr(c.text, "value"):
                                txt = c.text.value
                            if txt:
                                parts.append(txt)
        except Exception:
            pass
        out_text = "\n".join(parts) if parts else "(keine Ausgabe)"

    print("\n— Antwort —\n")
    print(out_text)

    # (Optional) echte API-Citations auf Dateiebene anzeigen – nicht identisch mit unseren Inline-Quellen!
    file_ids = set()
    try:
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", "") == "message":
                for c in getattr(item, "content", []) or []:
                    txt = getattr(c, "text", None)
                    anns = getattr(txt, "annotations", None) if txt else None
                    if anns:
                        for a in anns:
                            if getattr(a, "type", "") in ("file_citation", "file_path"):
                                fid = getattr(a, "file_id", None)
                                if fid:
                                    file_ids.add(fid)
    except Exception:
        pass

    if file_ids:
        print("\n— Quellen (Dateien aus File Search) —")
        for fid in file_ids:
            try:
                f = client.files.retrieve(fid)
                name = getattr(f, "filename", None) or fid
                print(f"- {name}")
            except Exception:
                print(f"- file_id: {fid}")

    return out_text

# --------------------------
# Main
# --------------------------
def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        die("Bitte OPENAI_API_KEY in .env oder als Umgebungsvariable setzen.")
    if len(sys.argv) < 2:
        die("Usage: python gutachten_responses.py <datei1> [datei2 ...]")

    client = OpenAI()
    model = DEFAULT_MODEL

    # 1) Whitelist aus lokalen JSONs bauen
    whitelist = extract_whitelists_from_json(sys.argv[1:])
    if not whitelist["unternehmen"] or not whitelist["jahre"]:
        print("ℹ️  Hinweis: Konnte Unternehmen/Jahr nicht aus JSON extrahieren – Quellen-Whitelist evtl. leer.")
    else:
        print(f"Whitelist Unternehmen: {sorted(whitelist['unternehmen'])}")
        print(f"Whitelist Jahre: {sorted(whitelist['jahre'])}")
        print(f"Whitelist Abteilungen (Auszug): {sorted(list(whitelist['abteilungen']))[:10]}{' …' if len(whitelist['abteilungen'])>10 else ''}")

    # 2) Dateien hochladen
    file_ids = upload_files(client, sys.argv[1:])
    if not file_ids:
        die("Keine Datei erfolgreich hochgeladen – bitte Pfade prüfen.")
    print(f"✅ Hochgeladen: {len(file_ids)} Datei(en)")

    # 3) Vector Store reusen/erstellen
    vs_id = load_or_create_vector_store(client, VS_ID_FILE)

    # 4) Dateien anhängen + Indexierung abwarten
    attach_and_index_files(client, vs_id, file_ids)

    # 5) Interaktiver Loop
    print("\nFrage eingeben (z. B. Unternehmensbeschreibung); Ende mit :quit:")
    while True:
        user_q = input("\n> ").strip()
        if user_q.lower() in (":q", ":quit", "exit"):
            print("Bye!")
            break
        if not user_q:
            continue

        # Response-Format vorbereiten (Schema nur, wenn aktiviert)
        kwargs = {
            "model": model,
            "input": user_q,
            "instructions": SYSTEM,
            "tools": [{
                "type": "file_search",
                "vector_store_ids": [vs_id],
            }],
        }
        if ENABLE_SCHEMA:
            schema = build_json_schema(whitelist)
            kwargs["response_format"] = {"type": "json_schema", "json_schema": schema}

        try:
            resp = client.responses.create(**kwargs)
        except Exception as e:
            print(f"❌ Anfrage fehlgeschlagen: {e}")
            continue

        if ENABLE_SCHEMA:
            # Versuch, JSON zu parsen und schön zu rendern
            raw = getattr(resp, "output_text", "") or ""
            try:
                data = parse_json_output_text(raw)
                md = render_structured_json_to_markdown(data)
                print("\n— Antwort —\n")
                print(md)
                warn_if_missing_inline_sources(md)
                warn_if_filename_in_source(md)
                continue
            except Exception as e:
                print(f"ℹ️  Konnte JSON-Schema-Ausgabe nicht parsen ({e}) – zeige Rohtext.")
                # Fallback auf generischen Renderer
                text = render_with_citations(resp, client)
                warn_if_missing_inline_sources(text)
                warn_if_filename_in_source(text)
        else:
            text = render_with_citations(resp, client)
            warn_if_missing_inline_sources(text)
            warn_if_filename_in_source(text)

if __name__ == "__main__":
    main()
