# Diagramy Przepływu Systemu hhg_baza_wiedzy - Mermaid

## 1. Inicjalizacja Systemu

```mermaid
sequenceDiagram
    participant Uvicorn
    participant FastAPI
    participant Lifespan
    participant RAGSystem
    participant OpenAI
    participant ChromaDB
    participant FileSystem

    Uvicorn->>FastAPI: Start aplikacji
    FastAPI->>Lifespan: lifespan startup
    Lifespan->>RAGSystem: __init__()
    RAGSystem->>RAGSystem: initialize()
    
    RAGSystem->>FileSystem: Sprawdź OPENAI_API_KEY
    FileSystem-->>RAGSystem: Klucz OK
    
    RAGSystem->>OpenAI: OpenAIEmbeddings(model="text-embedding-3-small")
    OpenAI-->>RAGSystem: Embeddings gotowe
    
    RAGSystem->>OpenAI: ChatOpenAI(model="gpt-3.5-turbo")
    OpenAI-->>RAGSystem: LLM gotowe
    
    RAGSystem->>FileSystem: Sprawdź chroma_db/
    alt Baza istnieje
        FileSystem-->>RAGSystem: Baza istnieje
        RAGSystem->>ChromaDB: load_vectorstore()
        ChromaDB-->>RAGSystem: Baza załadowana
    else Baza nie istnieje
        FileSystem-->>RAGSystem: Baza nie istnieje
        RAGSystem->>FileSystem: Skanuj documents/*.pdf
        FileSystem-->>RAGSystem: Lista PDF
        loop Dla każdego PDF
            RAGSystem->>FileSystem: PyPDFLoader(pdf_path)
            FileSystem-->>RAGSystem: Strony dokumentu
            RAGSystem->>RAGSystem: RecursiveCharacterTextSplitter
            RAGSystem->>RAGSystem: Chunki (500 znaków, overlap 100)
        end
        RAGSystem->>ChromaDB: Chroma.from_documents(chunks, embeddings)
        ChromaDB->>ChromaDB: Tworzenie indeksu HNSW
        ChromaDB->>FileSystem: Zapis do chroma_db/
        ChromaDB-->>RAGSystem: Baza utworzona
    end
    
    RAGSystem->>RAGSystem: create_qa_chain()
    RAGSystem->>RAGSystem: RetrievalQA z MMR search
    RAGSystem-->>Lifespan: System gotowy
    Lifespan-->>FastAPI: Startup complete
    FastAPI-->>Uvicorn: Aplikacja gotowa
```

## 2. Zadawanie Pytania (Główny Przepływ)

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant FastAPI
    participant RAGSystem
    participant OpenAI
    participant ChromaDB
    participant FAQSystem
    participant FileSystem

    User->>Frontend: Wpisz pytanie + Enter/Click
    Frontend->>Frontend: Walidacja (niepuste)
    Frontend->>FastAPI: POST /api/question<br/>{question: "..."}
    
    FastAPI->>FastAPI: Walidacja (3-1000 znaków)
    FastAPI->>FastAPI: Sprawdź rag_system
    
    FastAPI->>RAGSystem: search(question)
    RAGSystem->>RAGSystem: qa_chain.invoke({"query": question})
    
    RAGSystem->>OpenAI: Embedding pytania<br/>(text-embedding-3-small)
    OpenAI-->>RAGSystem: Wektor pytania (1536 dim)
    
    RAGSystem->>ChromaDB: MMR Search<br/>(k=8, fetch_k=16)
    ChromaDB->>ChromaDB: Wyszukiwanie podobnych wektorów<br/>(HNSW index)
    ChromaDB-->>RAGSystem: Top 8 fragmentów dokumentów
    
    RAGSystem->>RAGSystem: Prompt template<br/>(kontekst + pytanie)
    RAGSystem->>OpenAI: ChatOpenAI.generate()<br/>(prompt z kontekstem)
    OpenAI-->>RAGSystem: Odpowiedź (kompendium wiedzy)
    
    RAGSystem->>RAGSystem: Wyodrębnij źródła<br/>(nazwy PDF)
    RAGSystem-->>FastAPI: {answer, sources}
    
    FastAPI->>FAQSystem: update_faq(question, answer)
    
    FAQSystem->>FileSystem: load_faq_data()<br/>[LOCK_SH - read lock]
    FileSystem-->>FAQSystem: faq_data (dict)
    
    FAQSystem->>FAQSystem: get_question_hash(question)
    Note over FAQSystem: Normalizacja + MD5 hash
    
    alt Hash istnieje w FAQ
        FAQSystem->>FAQSystem: count += 1<br/>last_asked = now()
    else Hash nie istnieje
        FAQSystem->>FAQSystem: Dodaj nowe FAQ<br/>(count=1)
    end
    
    FAQSystem->>FAQSystem: Sortuj po count (desc)
    FAQSystem->>FAQSystem: Ogranicz do top 9
    
    FAQSystem->>FileSystem: save_faq_data(faq_data)<br/>[LOCK_EX - write lock]
    FileSystem->>FileSystem: json.dump()<br/>fsync()
    FileSystem-->>FAQSystem: Zapisano
    
    FAQSystem-->>FastAPI: faq_updated (bool)
    
    FastAPI-->>Frontend: {answer, updated_faq, sources}
    
    Frontend->>Frontend: Wyświetl odpowiedź
    alt updated_faq == true
        Frontend->>FastAPI: GET /api/faq
        FastAPI->>FAQSystem: load_faq_data()
        FAQSystem-->>FastAPI: List[FAQItem]
        FastAPI-->>Frontend: FAQ list
        Frontend->>Frontend: renderFAQ()
    end
```

## 3. Ładowanie FAQ przy Starcie

```mermaid
sequenceDiagram
    participant Browser
    participant Frontend
    participant FastAPI
    participant FAQSystem
    participant FileSystem

    Browser->>Frontend: Załaduj index.html
    Frontend->>Frontend: DOMContentLoaded event
    
    Frontend->>FastAPI: GET /api/faq
    
    FastAPI->>FAQSystem: load_faq_data()
    FAQSystem->>FileSystem: open(faq_data.json, 'r')<br/>[LOCK_SH]
    FileSystem-->>FAQSystem: JSON data
    FAQSystem->>FileSystem: unlock
    FAQSystem-->>FastAPI: faq_data (dict)
    
    FastAPI->>FastAPI: Sortuj po count (desc)
    FastAPI->>FastAPI: Ogranicz do MAX_FAQ_ITEMS (9)
    FastAPI->>FastAPI: Konwertuj do List[FAQItem]
    
    FastAPI-->>Frontend: List[FAQItem]
    
    Frontend->>Frontend: renderFAQ()
    Frontend->>Frontend: Wyświetl kafelki FAQ
```

## 4. Kliknięcie w Kafelek FAQ

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant FastAPI
    participant RAGSystem

    User->>Frontend: Kliknij kafelek FAQ
    Frontend->>Frontend: loadFAQAnswer(encodedQuestion)
    Frontend->>Frontend: questionInput.value = question
    Frontend->>Frontend: sendQuestion()
    
    Note over Frontend: [Teraz jak w Przepływie 2]
    Frontend->>FastAPI: POST /api/question
    FastAPI->>RAGSystem: search(question)
    RAGSystem-->>FastAPI: {answer, sources}
    FastAPI-->>Frontend: Answer
    Frontend->>Frontend: Wyświetl odpowiedź
```

## 5. File Locking - Race Condition Prevention

```mermaid
sequenceDiagram
    participant Request1
    participant Request2
    participant FAQSystem
    participant FileSystem

    par Request 1 i Request 2 jednocześnie
        Request1->>FAQSystem: update_faq(q1, a1)
        Request2->>FAQSystem: update_faq(q2, a2)
    end
    
    Request1->>FileSystem: open(faq_data.json, 'r')<br/>LOCK_SH (shared)
    FileSystem-->>Request1: Lock acquired
    
    Request2->>FileSystem: open(faq_data.json, 'r')<br/>LOCK_SH (shared)
    FileSystem-->>Request2: Lock acquired<br/>(shared lock OK)
    
    Request1->>FileSystem: json.load()
    FileSystem-->>Request1: Data
    Request1->>FileSystem: unlock
    
    Request2->>FileSystem: json.load()
    FileSystem-->>Request2: Data
    Request2->>FileSystem: unlock
    
    Request1->>Request1: Aktualizuj FAQ
    Request2->>Request2: Aktualizuj FAQ
    
    Request1->>FileSystem: open(faq_data.json, 'w')<br/>LOCK_EX (exclusive)
    FileSystem-->>Request1: Lock acquired
    
    Request2->>FileSystem: open(faq_data.json, 'w')<br/>LOCK_EX (exclusive)
    Note over FileSystem: Czeka na unlock Request1
    
    Request1->>FileSystem: json.dump()<br/>fsync()
    Request1->>FileSystem: unlock
    
    FileSystem-->>Request2: Lock acquired
    Request2->>FileSystem: json.dump()<br/>fsync()
    Request2->>FileSystem: unlock
```

## 6. Hash'owanie Pytań

```mermaid
flowchart TD
    A[Pytanie użytkownika] --> B[question.lower]
    B --> C[question.strip]
    C --> D[Usuń znaki specjalne<br/>tylko alphanumeric + spacje]
    D --> E[Normalizuj spacje<br/>wiele spacji -> jedna]
    E --> F[MD5 hash]
    F --> G[32-znakowy hex string]
    
    style A fill:#e1f5ff
    style G fill:#c8e6c9
```

## 7. MMR Search (Maximum Marginal Relevance)

```mermaid
flowchart TD
    A[Pytanie użytkownika] --> B[Embedding pytania]
    B --> C[ChromaDB MMR Search]
    C --> D[fetch_k = 16<br/>Pobierz 16 kandydatów]
    D --> E[Algorytm MMR]
    E --> F[Wybierz 8 najbardziej<br/>różnorodnych fragmentów]
    F --> G[Top 8 fragmentów<br/>z dokumentów]
    G --> H[Prompt: kontekst + pytanie]
    H --> I[LLM generuje odpowiedź]
    
    style A fill:#e1f5ff
    style I fill:#c8e6c9
```

## 8. Chunking Dokumentów

```mermaid
flowchart TD
    A[PDF Dokument] --> B[PyPDFLoader]
    B --> C[Lista stron]
    C --> D[RecursiveCharacterTextSplitter]
    D --> E{Sprawdź separatory}
    E -->|"\n\n"| F[Podziel na paragrafy]
    E -->|"\n"| G[Podziel na linie]
    E -->|". "| H[Podziel na zdania]
    E -->|";"| I[Podziel na sekcje]
    E -->|" "| J[Podziel na słowa]
    F --> K{chunk_size > 500?}
    G --> K
    H --> K
    I --> K
    J --> K
    K -->|TAK| E
    K -->|NIE| L[Chunk gotowy]
    L --> M[chunk_overlap = 100 znaków]
    M --> N[Lista chunków]
    N --> O[ChromaDB.from_documents]
    
    style A fill:#e1f5ff
    style O fill:#c8e6c9
```

## 9. Pełny Przepływ Systemu (High-Level)

```mermaid
graph TB
    subgraph Frontend
        A[User Interface]
        B[JavaScript]
    end
    
    subgraph Backend API
        C[FastAPI]
        D[Endpoints]
    end
    
    subgraph RAG System
        E[RAGSystem]
        F[Embeddings]
        G[LLM]
        H[QA Chain]
    end
    
    subgraph Vector Store
        I[ChromaDB]
        J[HNSW Index]
    end
    
    subgraph FAQ System
        K[FAQ Manager]
        L[File Locking]
        M[faq_data.json]
    end
    
    subgraph External Services
        N[OpenAI API]
    end
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    E --> G
    E --> H
    F --> N
    G --> N
    H --> I
    I --> J
    D --> K
    K --> L
    L --> M
    
    style A fill:#e1f5ff
    style N fill:#fff3e0
    style M fill:#f3e5f5
```

## 10. Error Handling Flow

```mermaid
sequenceDiagram
    participant User
    participant Frontend
    participant FastAPI
    participant RAGSystem
    participant FAQSystem

    User->>Frontend: Zadaj pytanie
    Frontend->>FastAPI: POST /api/question
    
    alt Pytanie za krótkie/długie
        FastAPI-->>Frontend: HTTP 400 Bad Request
        Frontend->>User: Wyświetl błąd walidacji
    else RAG nie gotowy
        FastAPI-->>Frontend: HTTP 503 Service Unavailable
        Frontend->>User: Wyświetl komunikat o braku gotowości
    else Błąd w RAG
        FastAPI->>RAGSystem: search(question)
        RAGSystem-->>FastAPI: Exception
        FastAPI->>FastAPI: Log error (exc_info=True)
        FastAPI-->>Frontend: HTTP 500 Internal Server Error
        Frontend->>User: Wyświetl komunikat o błędzie
    else Błąd w FAQ
        FastAPI->>FAQSystem: update_faq()
        FAQSystem-->>FastAPI: Exception
        FastAPI->>FastAPI: Log error
        FastAPI-->>Frontend: HTTP 200 (ale bez updated_faq)
        Frontend->>User: Wyświetl odpowiedź (FAQ nie zaktualizowane)
    else Sukces
        FastAPI->>RAGSystem: search(question)
        RAGSystem-->>FastAPI: {answer, sources}
        FastAPI->>FAQSystem: update_faq()
        FAQSystem-->>FastAPI: faq_updated
        FastAPI-->>Frontend: HTTP 200 {answer, updated_faq, sources}
        Frontend->>User: Wyświetl odpowiedź
    end
```

