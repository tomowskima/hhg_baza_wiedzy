# Notatki z konfiguracji projektu - hhg_baza_wiedzy

## Data: 2025-01-XX

## Problemy napotkane i rozwiązania

### 1. Problem z brakującym pakietem `langchain-chroma`
**Błąd:** `ModuleNotFoundError: No module named 'langchain_chroma'`

**Rozwiązanie:**
```bash
.venv/bin/python3 -m pip install langchain-chroma==0.1.3
```
Dodano do `requirements.txt`:
```
langchain-chroma==0.1.3
```

### 2. Konflikt wersji `openai` i `langchain-openai`
**Błąd:** `Client.__init__() got an unexpected keyword argument 'proxies'`

**Rozwiązanie:**
- Zmieniono inicjalizację `OpenAIEmbeddings` i `ChatOpenAI` - używają zmiennej środowiskowej zamiast przekazywać klucz bezpośrednio
- Zainstalowano kompatybilną wersję: `openai>=1.40.0,<1.50.0`

**Zmiana w kodzie:**
```python
# Przed:
self.embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    openai_api_key=os.getenv("OPENAI_API_KEY")
)

# Po:
os.environ["OPENAI_API_KEY"] = api_key
self.embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
)
```

### 3. Niekompatybilna baza ChromaDB
**Błąd:** `'_type'` podczas ładowania bazy

**Rozwiązanie:**
```bash
rm -rf chroma_db
```
Baza została utworzona od nowa przy następnym uruchomieniu.

### 4. Deprecation Warning LangChain
**Ostrzeżenie:** `The method Chain.__call__ was deprecated in langchain 0.1.0`

**Rozwiązanie:**
```python
# Przed:
result = self.qa_chain({"query": question})

# Po:
result = self.qa_chain.invoke({"query": question})
```

### 5. Konfiguracja klucza OpenAI API
**Lokalizacja:** https://platform.openai.com/api-keys

**Plik `.env`:**
```
OPENAI_API_KEY=sk-proj-...
OPENAI_MODEL=gpt-3.5-turbo
EMBEDDING_MODEL=text-embedding-3-small
PDF_FOLDER=documents/
CHROMA_PERSIST_DIRECTORY=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=100
MAX_SEARCH_RESULTS=8
CHAIN_TYPE=stuff
```

## Finalne wersje pakietów

- `langchain==0.2.16`
- `langchain-community==0.2.16`
- `langchain-openai==0.1.25`
- `langchain-chroma==0.1.3`
- `openai>=1.40.0,<1.50.0`
- `chromadb==0.5.23` (zależność od langchain-chroma)

## Komendy do uruchomienia

```bash
# Aktywacja środowiska wirtualnego
cd /Users/tomowski/PycharmProjects/PythonProject/hhg_baza_wiedzy
source .venv/bin/activate  # lub .venv/bin/python3

# Uruchomienie aplikacji
.venv/bin/python main.py

# Aplikacja dostępna pod:
# - Frontend: http://localhost:8000
# - API Docs: http://localhost:8000/docs
```

## Git - konfiguracja i push

```bash
# Utworzenie .gitignore (już istnieje)
# Dodanie plików
git add main.py index.html requirements.txt README.md Procfile render.yaml runtime.txt .gitignore

# Pierwszy commit
git commit -m "Initial commit: RAG System for Head Hunter Groups"

# Konfiguracja remote
git remote add origin https://github.com/tomowskima/hhg_baza_wiedzy.git

# Push
git push -u origin main
```

## Repozytorium GitHub

https://github.com/tomowskima/hhg_baza_wiedzy

## Uwagi

- Plik `.env` NIE jest w repozytorium (jest w `.gitignore`) - to dobrze!
- Dokumenty PDF też nie są w repo (folder `documents/` jest ignorowany)
- Na produkcji (Render) trzeba dodać zmienne środowiskowe w panelu
- Błędy telemetrii ChromaDB są niegroźne i nie wpływają na działanie

## Status końcowy

✅ Wszystkie problemy rozwiązane
✅ Aplikacja działa poprawnie
✅ Kod na GitHubie
✅ Gotowe do dalszego rozwoju

