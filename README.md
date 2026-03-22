# Presidio Detector

Prototypische Implementierung zur lokalen Erkennung personenbezogener Daten in Texten mit Microsoft Presidio.
- Lokale Analyse ohne OpenAI API
- Erkennung über Microsoft Presidio + spaCy
- Konsolenbasierte Ausgabe gefundener Treffer

---

## BEFEHLE

### Virtuelle Umgebung anlegen
```bash
python3.12 -m venv .venv
```

### Virtuelle Umgebung aktivieren
**macOS / Linux (bash/zsh)**
```bash
source .venv/bin/activate
```

### Abhängigkeiten installieren
```bash
pip install --upgrade pip
pip install presidio-analyzer spacy
```

### spaCy Modell installieren
```bash
python -m spacy download en_core_web_lg
```

### requirements.txt verwenden
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```

### Script mit direktem Text ausführen
```bash
python presidio_detect.py --text "John Doe, john@example.com, +1 202 555 0101"
```

### Script mit Datei ausführen
```bash
python presidio_detect.py --file input.txt
```# presidio_evaluation
