# Head Hunter Groups - System RAG

System RAG (Retrieval-Augmented Generation) do przeszukiwania bazy wiedzy o rekrutacji, karierze i rynku pracy.

## Funkcjonalności

- 🔍 Przeszukiwanie dokumentów PDF z użyciem RAG
- 💬 Interaktywny interfejs do zadawania pytań
- 📊 System FAQ z najczęściej zadawanymi pytaniami
- 🤖 Wykorzystanie OpenAI GPT i embeddings
- 💾 ChromaDB jako baza wektorowa

## Wymagania

- Python 3.11+
- OpenAI API Key

## Instalacja lokalna

```bash
# Utwórz wirtualne środowisko
python -m venv .venv

# Aktywuj
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Zainstaluj zależności
pip install -r requirements.txt

# Utwórz plik .env
OPENAI_API_KEY=twój-klucz-api
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small
PDF_FOLDER=documents/
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=100
MAX_SEARCH_RESULTS=8
CHAIN_TYPE=stuff

# Dodaj dokumenty PDF do folderu documents/

# Uruchom serwer
python main.py
```

Aplikacja będzie dostępna pod adresem: http://localhost:8000

## Wdrożenie na Render

1. Zarejestruj się na [Render.com](https://render.com)
2. Połącz repozytorium GitHub
3. Utwórz nowy Web Service
4. Wybierz repozytorium i branch
5. Ustaw:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn main:app --host 0.0.0.0 --port $PORT`
6. Dodaj zmienne środowiskowe (zobacz `render.yaml`)
7. Deploy!

## Struktura projektu

```
rag-system/
├── main.py              # FastAPI aplikacja
├── index.html           # Frontend
├── documents/           # Dokumenty PDF (baza wiedzy)
├── chroma_db/          # Baza wektorowa (generowana)
├── faq_data.json       # Dane FAQ (generowane)
└── requirements.txt     # Zależności
```

## API Endpoints

- `GET /` - Strona główna
- `POST /api/question` - Zadaj pytanie
- `GET /api/faq` - Pobierz FAQ
- `GET /health` - Status systemu
- `POST /api/reset` - Reset bazy wektorowej
- `POST /api/reset-faq` - Reset FAQ

## Zmienne środowiskowe

- `OPENAI_API_KEY` - klucz API OpenAI (wymagane)
- `OPENAI_MODEL` - model LLM (domyślnie: gpt-3.5-turbo)
- `EMBEDDING_MODEL` - model embeddings (domyślnie: text-embedding-3-small)
- `PDF_FOLDER` - folder z dokumentami (domyślnie: documents/)
- `CHROMA_PERSIST_DIRECTORY` - katalog bazy wektorowej (domyślnie: ./chroma_db)
- `CHUNK_SIZE` - rozmiar fragmentów (domyślnie: 500)
- `CHUNK_OVERLAP` - nakładanie fragmentów (domyślnie: 100)
- `MAX_SEARCH_RESULTS` - maksymalna liczba wyników (domyślnie: 8)
- `CHAIN_TYPE` - typ łańcucha RAG (domyślnie: stuff)

## Licencja

MIT

