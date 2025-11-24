# Architektura Systemu hhg_baza_wiedzy - Low-Level Opis

## Przegląd Systemu

`hhg_baza_wiedzy` to system RAG (Retrieval-Augmented Generation) oparty na FastAPI, który umożliwia zadawanie pytań w języku naturalnym i otrzymywanie odpowiedzi na podstawie dokumentów PDF z dziedziny HR.

## Komponenty Systemu

### 1. Frontend (index.html)
- **Technologia**: Vanilla JavaScript, HTML5, CSS3
- **Funkcjonalność**:
  - Interfejs użytkownika do zadawania pytań
  - Wyświetlanie odpowiedzi z systemu RAG
  - Wyświetlanie kafelków z najczęściej zadawanymi pytaniami (FAQ)
  - Automatyczne ładowanie FAQ przy starcie strony
  - Odświeżanie FAQ po zadaniu pytania

### 2. Backend API (main.py - FastAPI)
- **Framework**: FastAPI 0.109.0
- **Serwer**: Uvicorn (ASGI)
- **Port**: 8000 (lokalnie) / $PORT (Render)

#### Endpointy:
- `GET /` - Serwuje frontend (index.html) lub zwraca info o API
- `GET /health` - Health check systemu
- `POST /api/question` - Główny endpoint do zadawania pytań
- `GET /api/faq` - Pobiera listę najczęściej zadawanych pytań
- `GET /api/faq/debug` - Debug endpoint do sprawdzenia stanu FAQ
- `POST /api/reset` - Resetuje bazę wektorową
- `POST /api/reset-faq` - Resetuje system FAQ

### 3. System RAG (RAGSystem)

#### 3.1 Inicjalizacja
```
1. Sprawdzenie zmiennych środowiskowych (OPENAI_API_KEY)
2. Inicjalizacja OpenAIEmbeddings (text-embedding-3-small)
3. Inicjalizacja ChatOpenAI (gpt-3.5-turbo)
4. Sprawdzenie czy istnieje baza wektorowa (chroma_db/)
   - Jeśli TAK: load_vectorstore()
   - Jeśli NIE: create_vectorstore()
5. Utworzenie łańcucha QA (create_qa_chain())
```

#### 3.2 Tworzenie Bazy Wektorowej (create_vectorstore)
```
1. Skanowanie folderu documents/ w poszukiwaniu plików .pdf
2. Dla każdego PDF:
   a. Załadowanie dokumentu (PyPDFLoader)
   b. Podział na strony
   c. Podział na chunki (RecursiveCharacterTextSplitter):
      - chunk_size: 500 znaków
      - chunk_overlap: 100 znaków
      - separators: ["\n\n", "\n", ". ", ";", " "]
   d. Dodanie chunków do listy all_chunks
3. Utworzenie bazy wektorowej ChromaDB:
   - documents: all_chunks
   - embedding: OpenAIEmbeddings
   - persist_directory: ./chroma_db
4. Automatyczne zapisanie do dysku (persist)
```

#### 3.3 Wyszukiwanie Odpowiedzi (search)
```
1. Sprawdzenie czy qa_chain istnieje
2. Wywołanie qa_chain.invoke({"query": question})
3. Proces wewnętrzny:
   a. Embedding pytania (OpenAIEmbeddings)
   b. Wyszukiwanie podobnych chunków w ChromaDB (MMR - Maximum Marginal Relevance)
      - search_type: "mmr"
      - k: MAX_SEARCH_RESULTS (8)
      - fetch_k: MAX_SEARCH_RESULTS * 2 (16)
   c. Pobranie top 8 najbardziej podobnych fragmentów
   d. Przekazanie kontekstu do LLM (ChatOpenAI)
   e. Generowanie odpowiedzi przez LLM z użyciem prompt template
4. Wyodrębnienie źródeł (nazwy plików PDF)
5. Zwrócenie odpowiedzi i źródeł
```

#### 3.4 Prompt Template
```
Jesteś ekspertem HR analizującym dokumenty o procesach rekrutacji...
[12 zasad dla kompendium wiedzy]
Kontekst: {context}  # Top 8 fragmentów z dokumentów
Pytanie: {question}
KOMPENDIUM CAŁEJ WIEDZY: [odpowiedź LLM]
```

### 4. System FAQ (FAQ Management)

#### 4.1 Struktura Danych
```json
{
  "hash_pytania": {
    "question": "Tekst pytania",
    "answer": "Odpowiedź (max 200 znaków)...",
    "count": 5,
    "last_asked": "2025-11-23T14:23:15.329641311",
    "question_hash": "4dc0f1f38feaf667ef6928d2817b47e6"
  }
}
```

#### 4.2 Hash'owanie Pytań (get_question_hash)
```
1. Normalizacja pytania:
   - lowercase
   - strip (usunięcie białych znaków na początku/końcu)
   - Usunięcie znaków specjalnych (tylko alphanumeric + spacje)
   - Normalizacja spacji (wiele spacji -> jedna)
2. MD5 hash znormalizowanego pytania
3. Zwrócenie 32-znakowego hex string
```

#### 4.3 Aktualizacja FAQ (update_faq)
```
1. Załadowanie danych z pliku (load_faq_data) z file locking:
   - fcntl.LOCK_SH (shared lock - read)
   - Retry mechanism (5 prób, exponential backoff)
2. Obliczenie hash pytania
3. Sprawdzenie czy hash istnieje w danych:
   - Jeśli TAK: 
     * count += 1
     * last_asked = now()
     * old_position = get_faq_position()
   - Jeśli NIE:
     * Dodanie nowego FAQ z count=1
     * old_position = None
4. Sortowanie po count (malejąco)
5. Ograniczenie do MAX_FAQ_ITEMS (9):
   - Jeśli więcej niż 9: zostaw tylko top 9
   - Jeśli nowe pytanie ma count=1 i jest poza top 9: zostanie usunięte
6. Zapis do pliku (save_faq_data) z file locking:
   - fcntl.LOCK_EX (exclusive lock - write)
   - json.dump()
   - f.flush() + os.fsync() - wymuszenie zapisu na dysk
7. Zwrócenie True jeśli pozycja się zmieniła
```

#### 4.4 File Locking (Race Condition Prevention)
```
Problem: Wiele requestów jednocześnie może próbować zapisać FAQ
Rozwiązanie: File locking z fcntl
- LOCK_SH: Shared lock (read) - wiele procesów może czytać jednocześnie
- LOCK_EX: Exclusive lock (write) - tylko jeden proces może pisać
- Retry mechanism: 5 prób z exponential backoff (0.1s, 0.2s, 0.4s, 0.8s, 1.6s)
```

### 5. ChromaDB (Vector Store)

#### 5.1 Struktura
```
chroma_db/
├── chroma.sqlite3          # SQLite database z metadanymi
└── [collection_id]/
    ├── data_level0.bin     # Dane wektorowe (HNSW index)
    ├── header.bin          # Nagłówek kolekcji
    ├── length.bin          # Długości wektorów
    └── link_lists.bin      # Listy połączeń HNSW
```

#### 5.2 HNSW (Hierarchical Navigable Small World)
- Algorytm aproksymacyjnego wyszukiwania najbliższych sąsiadów
- Szybkie wyszukiwanie podobnych wektorów (embeddingów)
- Używany przez ChromaDB do wyszukiwania podobnych fragmentów dokumentów

### 6. OpenAI Integration

#### 6.1 Embeddings (text-embedding-3-small)
- Model: text-embedding-3-small
- Wektor: 1536 wymiarów
- Użycie: Embedding pytania i chunków dokumentów

#### 6.2 Chat Model (gpt-3.5-turbo)
- Model: gpt-3.5-turbo
- Temperature: 0.1 (niska - dokładność)
- Max tokens: 1000
- Użycie: Generowanie odpowiedzi na podstawie kontekstu

#### 6.3 Workaround dla proxies
- Problem: langchain-openai 0.2.0+ przekazuje `proxies` do OpenAI 1.40.0, który go nie akceptuje
- Rozwiązanie: Monkey patch dla `SyncHttpxClientWrapper` i `AsyncHttpxClientWrapper`
- Usunięcie `proxies` z kwargs przed inicjalizacją klienta

### 7. Lifecycle Management

#### 7.1 Startup (lifespan context manager)
```
1. FastAPI lifespan startup:
   - Inicjalizacja RAGSystem()
   - Jeśli błąd: logowanie, ale aplikacja startuje (rag_system = None)
2. Aplikacja gotowa do przyjmowania requestów
```

#### 7.2 Shutdown
```
- Opcjonalny cleanup (obecnie pusta implementacja)
- ChromaDB automatycznie zapisuje dane
```

## Przepływ Danych

### Przepływ 1: Inicjalizacja Systemu
```
Uvicorn Start
  → FastAPI lifespan startup
    → RAGSystem.__init__()
      → initialize()
        → Sprawdzenie OPENAI_API_KEY
        → OpenAIEmbeddings(model="text-embedding-3-small")
        → ChatOpenAI(model="gpt-3.5-turbo")
        → Sprawdzenie chroma_db/
          → Jeśli istnieje: load_vectorstore()
          → Jeśli nie: create_vectorstore()
            → Skanowanie documents/
            → Ładowanie PDF (PyPDFLoader)
            → Podział na chunki (RecursiveCharacterTextSplitter)
            → Chroma.from_documents() → zapis do chroma_db/
        → create_qa_chain()
          → RetrievalQA.from_chain_type()
            → Retriever z MMR search
            → Prompt template
      → rag_system gotowy
  → Aplikacja gotowa
```

### Przepływ 2: Zadawanie Pytania
```
User (Frontend)
  → POST /api/question
    → FastAPI endpoint (ask_question)
      → Walidacja pytania (3-1000 znaków)
      → Sprawdzenie rag_system
      → rag_system.search(question)
        → qa_chain.invoke({"query": question})
          → Embedding pytania (OpenAIEmbeddings)
          → Wyszukiwanie w ChromaDB (MMR, k=8)
          → Pobranie top 8 fragmentów
          → Prompt: kontekst + pytanie
          → LLM generuje odpowiedź (ChatOpenAI)
          → Wyodrębnienie źródeł (nazwy PDF)
        → Zwrócenie {answer, sources}
      → update_faq(question, answer)
        → load_faq_data() [LOCK_SH]
        → get_question_hash(question)
        → Sprawdzenie czy hash istnieje
        → Aktualizacja lub dodanie FAQ
        → Sortowanie po count
        → Ograniczenie do top 9
        → save_faq_data() [LOCK_EX]
      → Zwrócenie Answer(answer, updated_faq, sources)
  → Frontend wyświetla odpowiedź
  → Jeśli updated_faq: loadFAQ() → GET /api/faq
```

### Przepływ 3: Ładowanie FAQ
```
User (Frontend) - DOMContentLoaded
  → GET /api/faq
    → FastAPI endpoint (get_faq)
      → load_faq_data() [LOCK_SH]
      → Sortowanie po count (malejąco)
      → Ograniczenie do MAX_FAQ_ITEMS (9)
      → Zwrócenie List[FAQItem]
  → Frontend renderFAQ()
    → Wyświetlenie kafelków FAQ
```

### Przepływ 4: Kliknięcie w Kafelek FAQ
```
User kliknie kafelek FAQ
  → loadFAQAnswer(encodedQuestion)
    → questionInput.value = question
    → sendQuestion()
      → POST /api/question
        → [jak w Przepływie 2]
```

## Bezpieczeństwo i Obsługa Błędów

### File Locking
- Zapobiega race conditions przy jednoczesnych zapytaniach
- Retry mechanism z exponential backoff
- Graceful degradation przy błędach I/O

### Error Handling
- Try-catch w każdym endpoint
- Logowanie błędów z exc_info=True
- HTTPException z odpowiednimi kodami statusu
- Graceful degradation: aplikacja działa nawet jeśli RAG nie jest gotowy

### CORS
- Allow all origins (w produkcji zmienić na konkretną domenę)
- Allow all methods i headers

## Optymalizacje

### 1. Chunking Strategy
- chunk_size: 500 znaków (optymalne dla kompendium)
- chunk_overlap: 100 znaków (zachowanie kontekstu)
- RecursiveCharacterTextSplitter (inteligentny podział)

### 2. MMR Search
- Maximum Marginal Relevance zamiast prostego similarity search
- fetch_k: 16 (więcej kandydatów do wyboru)
- k: 8 (finalne fragmenty)
- Lepsza różnorodność wyników

### 3. Prompt Engineering
- Szczegółowe zasady dla różnych typów pytań
- Wymuszenie użycia tylko kontekstu z dokumentów
- Wyczerpujące odpowiedzi (kompendium)

### 4. FAQ Management
- Hash'owanie dla identyfikacji podobnych pytań
- Top 9 najpopularniejszych pytań
- Automatyczne sortowanie po count

## Deployment (Render.com)

### Build Process
```
1. Clone repository
2. Install Python 3.11.9 (runtime.txt)
3. pip install -r requirements.txt
4. Start: uvicorn main:app --host 0.0.0.0 --port $PORT
```

### Environment Variables
- OPENAI_API_KEY (required)
- OPENAI_MODEL (default: gpt-3.5-turbo)
- EMBEDDING_MODEL (default: text-embedding-3-small)
- PDF_FOLDER (default: documents/)
- CHROMA_PERSIST_DIRECTORY (default: ./chroma_db)
- CHUNK_SIZE (default: 500)
- CHUNK_OVERLAP (default: 100)
- MAX_SEARCH_RESULTS (default: 8)
- CHAIN_TYPE (default: stuff)

### Filesystem Considerations
- ChromaDB: persistent directory (chroma_db/)
- FAQ: faq_data.json (w repo dla wszystkich użytkowników)
- Documents: documents/ (PDF w repo)

## Limity i Ograniczenia

1. **MAX_FAQ_ITEMS**: 9 (tylko top 9 najpopularniejszych pytań)
2. **Question length**: 3-1000 znaków
3. **Answer truncation**: 200 znaków w FAQ
4. **Chunk size**: 500 znaków
5. **Search results**: 8 fragmentów
6. **File locking retries**: 5 prób

## Przyszłe Ulepszenia

1. **Baza danych dla FAQ**: PostgreSQL zamiast JSON (trwałość między restartami)
2. **Cache**: Redis dla często zadawanych pytań
3. **Rate limiting**: Ograniczenie liczby requestów na użytkownika
4. **Authentication**: Autoryzacja użytkowników
5. **Analytics**: Śledzenie popularności pytań
6. **Multi-language**: Wsparcie dla wielu języków

