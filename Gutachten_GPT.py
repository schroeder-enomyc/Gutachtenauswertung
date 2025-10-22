import os, sys, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

SYSTEM = (
    "Du bist Unternehmensberater. Antworte NUR auf Basis der File-Search-Wissensbasis. "
    "Gib konkrete Handlungsempfehlungen, gruppiert nach Abteilungen, mit kurzer Begründung "
    "und Quellenhinweisen. Nutze für die Quellenangabe die Felder aus der Wissensbasis: /unternehmen und /jahr. Wenn Information nicht in den Dateien "
    "auffindbar ist: Antworte mit 'Keine Quelle gefunden' und stelle eine präzise Rückfrage."
)

def die(msg: str, code: int = 1):
    print(msg); sys.exit(code)

def main():
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        die("Bitte OPENAI_API_KEY in .env oder als Umgebungsvariable setzen.")
    if len(sys.argv) < 2:
        die("Usage: python gutachten_responses.py <datei1> [datei2 ...]")

    client = OpenAI()
    model = os.getenv("OPENAI_MODEL", "gpt-4o")  # setz hier dein Wunschmodell per ENV

    # 1) Dateien hochladen
    file_ids = []
    for raw in sys.argv[1:]:
        p = Path(raw)
        if not p.exists():
            print(f"⚠️ Datei nicht gefunden: {p}")
            continue
        with p.open("rb") as f:
            up = client.files.create(file=f, purpose="assistants")
        file_ids.append(up.id)

    if not file_ids:
        die("Keine Datei erfolgreich hochgeladen – bitte Pfade prüfen.")
    print(f"✅ Hochgeladen: {len(file_ids)} Datei(en)")

    # 2) Vector Store erstellen
    vs = client.vector_stores.create(name="gutachten-vs")

    # 3) Dateien im Batch anhängen + Indexierung abwarten
    batch = client.vector_stores.file_batches.create(
        vector_store_id=vs.id,
        file_ids=file_ids
    )

    while True:
        b = client.vector_stores.file_batches.retrieve(
            vector_store_id=vs.id,
            batch_id=batch.id
        )

        status = getattr(b, "status", None)
        counts = getattr(b, "file_counts", None)

        # counts kann ein Pydantic-Modell sein -> Felder direkt lesen
        completed = getattr(counts, "completed", None)
        failed = getattr(counts, "failed", None)
        in_progress = getattr(counts, "in_progress", None)
        total = getattr(counts, "total", None)

        # Fallback, falls das Modell anders heißt / sich ändert:
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


    # 4) Interaktive Fragen via Responses API + File Search
    print("\nFrage eingeben (z. B. Unternehmensbeschreibung); Ende mit :quit:")
    while True:
        user_q = input("\n> ").strip()
        if user_q.lower() in (":q", ":quit", "exit"):
            print("Bye!"); break
        if not user_q:
            continue

        try:
            resp = client.responses.create(
                model=model,
                input=user_q,
                instructions=SYSTEM,
                tools=[{
                    "type": "file_search",
                    "vector_store_ids": [vs.id],
                }]
            )
        except Exception as e:
            print(f"❌ Anfrage fehlgeschlagen: {e}")
            continue

        print("\n— Antwort —")
        print(getattr(resp, "output_text", None) or "(keine Ausgabe)")

if __name__ == "__main__":
    main()
