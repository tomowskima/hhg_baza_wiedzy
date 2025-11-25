# Analiza Projektu hhg_baza_wiedzy w Kontekście Artykułu o Optymalizacji Okna Kontekstowego LLM

## Wprowadzenie

Ten dokument odnosi nasz projekt `hhg_baza_wiedzy` do kluczowych aspektów poruszonych w artykule "Optymalizacja wydajności LLM: dogłębna analiza wpływu okna kontekstowego" autorstwa Kuriko IWAI.

---

## 1. Okno Kontekstowe - Definicja i Implementacja

### Artykuł:
> "Okno kontekstu (lub długość kontekstu) definiuje maksymalną liczbę tokenów — wliczając monit wejściowy, wszelkie instrukcje systemowe i wygenerowaną odpowiedź modelu — które LLM może jednocześnie przetworzyć i obsłużyć podczas pętli autoregresyjnej."

### Nasza Implementacja:

**Lokalizacja w kodzie:** `main.py`, linie 177-181

```python
self.llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0.1,  # Niska temperatura dla dokładności
    max_tokens=1000  # Więcej tokenów dla map_reduce
)
```

**Analiza:**
- ✅ Ustawiamy `max_tokens=1000` - limit tokenów dla odpowiedzi
- ⚠️ **BRAK**: Nie kontrolujemy bezpośrednio rozmiaru okna kontekstowego wejściowego
- ⚠️ **BRAK**: Nie mamy eksperymentalnej kontroli nad rozmiarem kontekstu jak w artykule (2048, 4096, 6144, 8192)

**Rekomendacja:** Powinniśmy dodać kontrolę nad `max_completion_tokens` i rozmiarem kontekstu wejściowego.

---

## 2. Chunking Dokumentów - Strategia Podziału

### Artykuł:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, 
    chunk_overlap=50, 
    separators=["\n\n", "\n", " ", ""]
)
```

### Nasza Implementacja:

**Lokalizacja w kodzie:** `main.py`, linie 226-231

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,      # 500 znaków (domyślnie)
    chunk_overlap=CHUNK_OVERLAP,  # 100 znaków (domyślnie)
    length_function=len,
    separators=["\n\n", "\n", ". ", ";", " "]
)
```

**Porównanie:**

| Parametr | Artykuł | Nasz Projekt | Różnica |
|----------|---------|--------------|---------|
| `chunk_size` | 512 | 500 | -12 znaków (2.3% mniej) |
| `chunk_overlap` | 50 | 100 | +50 znaków (100% więcej) |
| `separators` | `["\n\n", "\n", " ", ""]` | `["\n\n", "\n", ". ", ";", " "]` | Dodatkowe separatory (zdania, średniki) |

**Analiza:**
- ✅ Używamy tej samej strategii chunkingu (RecursiveCharacterTextSplitter)
- ✅ Większy overlap (100 vs 50) - lepsze zachowanie kontekstu między chunkami
- ✅ Dodatkowe separatory (". ", ";") - lepszy podział na zdania i sekcje
- ✅ Konfigurowalne przez zmienne środowiskowe (`CHUNK_SIZE`, `CHUNK_OVERLAP`)

**Kod konfiguracji:** `main.py`, linie 119-120
```python
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
```

---

## 3. Przechowywanie Dokumentów w ChromaDB

### Artykuł:
```python
vector_db = Chroma.from_documents(
    documents=document_chunks, 
    embedding=embedding_model, 
    persist_directory="./_rag_db_chroma"
)
```

### Nasza Implementacja:

**Lokalizacja w kodzie:** `main.py`, linie 243-247

```python
self.vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=self.embeddings,
    persist_directory=CHROMA_PERSIST_DIR
)
```

**Porównanie:**

| Aspekt | Artykuł | Nasz Projekt | Status |
|--------|---------|--------------|--------|
| Metoda tworzenia | `Chroma.from_documents()` | `Chroma.from_documents()` | ✅ Identyczne |
| Embedding model | `text-embedding-ada-002` | `text-embedding-3-small` | ⚠️ Nowszy model |
| Persist directory | `./_rag_db_chroma` | `./chroma_db` (konfigurowalne) | ✅ Konfigurowalne |

**Kod inicjalizacji embeddings:** `main.py`, linie 173-175
```python
self.embeddings = OpenAIEmbeddings(
    model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
)
```

**Analiza:**
- ✅ Używamy tego samego podejścia do przechowywania
- ✅ Używamy nowszego modelu embedding (`text-embedding-3-small` vs `text-embedding-ada-002`)
- ✅ Katalog persystencji jest konfigurowalny przez zmienną środowiskową

---

## 4. Konfiguracja LLM - Temperature i Sampling

### Artykuł:
```python
TEMPERATURE=0.2
TOP_P=1e-10
TOP_K=1.0
MAX_RESPONSE_TOKENS = 2048
```

### Nasza Implementacja:

**Lokalizacja w kodzie:** `main.py`, linie 177-181

```python
self.llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0.1,  # Niska temperatura dla dokładności
    max_tokens=1000  # Więcej tokenów dla map_reduce
)
```

**Porównanie:**

| Parametr | Artykuł | Nasz Projekt | Różnica |
|----------|---------|--------------|---------|
| `temperature` | 0.2 | 0.1 | -0.1 (bardziej deterministyczny) |
| `top_p` | 1e-10 | ❌ Nie ustawione | ⚠️ Brak kontroli |
| `top_k` | 1.0 | ❌ Nie ustawione | ⚠️ Brak kontroli |
| `max_tokens` | 2048 | 1000 | -1048 tokenów (51% mniej) |

**Analiza:**
- ✅ Niska temperatura (0.1 vs 0.2) - jeszcze bardziej deterministyczne odpowiedzi
- ⚠️ **BRAK**: Nie kontrolujemy `top_p` i `top_k` - brak pełnej kontroli nad sampling strategy
- ⚠️ **BRAK**: Mniejszy limit tokenów (1000 vs 2048) - może ograniczać długość odpowiedzi

**Rekomendacja:** Dodać kontrolę nad `top_p` i zwiększyć `max_tokens` dla dłuższych odpowiedzi.

---

## 5. Konfiguracja RAG Pipeline - Retrieval Strategy

### Artykuł:
```python
retriever = vector_db.as_retriever(search_kwargs={'k': k})
# k=50 w eksperymencie
```

### Nasza Implementacja:

**Lokalizacja w kodzie:** `main.py`, linie 304-307

```python
retriever=self.vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance - lepsze wyniki
    search_kwargs={"k": MAX_SEARCH_RESULTS, "fetch_k": MAX_SEARCH_RESULTS * 2}
)
```

**Porównanie:**

| Aspekt | Artykuł | Nasz Projekt | Status |
|--------|---------|--------------|--------|
| Search type | Standard similarity | `mmr` (Maximum Marginal Relevance) | ✅ Lepsze podejście |
| `k` (liczba wyników) | 50 | 8 (MAX_SEARCH_RESULTS) | ⚠️ Znacznie mniej |
| `fetch_k` | ❌ Nie używane | 16 (MAX_SEARCH_RESULTS * 2) | ✅ Dodatkowa optymalizacja |

**Kod konfiguracji:** `main.py`, linia 121
```python
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 8))
```

**Analiza:**
- ✅ Używamy **MMR (Maximum Marginal Relevance)** - lepsze niż standardowe similarity search
  - MMR zapewnia różnorodność wyników, nie tylko największą podobność
  - Zapobiega "zagubieniu w środku" (lost in the middle) problem
- ⚠️ Mniejsza liczba wyników (8 vs 50) - może ograniczać kontekst
- ✅ `fetch_k` = 16 - pobiera więcej kandydatów przed wyborem top 8 (podobne do artykułu)

**Rekomendacja:** Rozważyć zwiększenie `MAX_SEARCH_RESULTS` do 10-15 dla lepszego kontekstu.

---

## 6. Prompt Engineering - Template i Instrukcje

### Artykuł:
```python
system_prompt = (
    "You are an expert answer generator. Based ONLY on the following context, answer the complex, multi-hop question from the user."
    "If you cannot find the answer in the provided context, state that you do not have enough information."
    "\n\nContext:\n{context}"
)
```

### Nasza Implementacja:

**Lokalizacja w kodzie:** `main.py`, linie 271-293

```python
prompt_template = """Jesteś ekspertem HR analizującym dokumenty o procesach rekrutacji, zatrudniania i zarządzania personelem.
Twoim zadaniem jest stworzenie KOMPENDIUM całej wiedzy na zadane pytanie.

WAŻNE ZASADY:
1. Jeśli pytanie dotyczy CZASU - odpowiedz o czasie (w dniach/tygodniach/miesiącach)
2. Jeśli pytanie dotyczy KWOT - odpowiedz o kwotach (w PLN/EUR/USD)
3. Jeśli pytanie dotyczy PROCENTÓW - odpowiedz o procentach (w %)
4. Jeśli pytanie dotyczy ETAPÓW - wymień wszystkie etapy procesu
5. Jeśli pytanie dotyczy WYMAGAŃ - wymień wszystkie wymagania
6. Jeśli pytanie dotyczy DOKUMENTÓW - wymień wszystkie potrzebne dokumenty
7. Jeśli pytanie dotyczy PRAW - odpowiedz zgodnie z przepisami prawa pracy
8. Jeśli nie znajdziesz dokładnej odpowiedzi w kontekście, napisz "Nie znalazłem tej informacji w dokumentach"
9. Bądź WYCZERPUJĄCY i SZCZEGÓŁOWY - nie skracaj odpowiedzi!
10. Cytuj dokładne wartości jeśli są podane w dokumentach
11. Używaj TYLKO informacji z kontekstu
12. Stwórz pełny obraz tematu - kompendium całej wiedzy

Kontekst:
{context}

Pytanie: {question}

KOMPENDIUM CAŁEJ WIEDZY (bądź szczegółowy i wyczerpujący):"""
```

**Porównanie:**

| Aspekt | Artykuł | Nasz Projekt | Status |
|--------|---------|--------------|--------|
| Domena eksperta | Generic | HR-specific | ✅ Bardziej specjalistyczny |
| Instrukcje | 2 główne zasady | 12 szczegółowych zasad | ✅ Znacznie bardziej szczegółowy |
| Obsługa braku odpowiedzi | ✅ Tak | ✅ Tak (zasada 8) | ✅ Identyczne |
| Wymuszenie użycia kontekstu | ✅ Tak | ✅ Tak (zasada 11) | ✅ Identyczne |
| Język | Angielski | Polski | ⚠️ Różnica językowa |
| Cel odpowiedzi | Multi-hop Q&A | Kompendium wiedzy | ✅ Bardziej ambitny cel |

**Analiza:**
- ✅ Znacznie bardziej szczegółowy prompt z 12 zasadami vs 2 w artykule
- ✅ Specjalizacja domenowa (HR) - lepsze dla konkretnych przypadków użycia
- ✅ Wymuszenie "kompendium wiedzy" - bardziej wyczerpujące odpowiedzi
- ✅ Obsługa różnych typów pytań (czas, kwoty, procenty, etapy, wymagania, dokumenty, prawo)

---

## 7. Kontrola Rozmiaru Kontekstu - Token Pruning

### Artykuł:
```python
def _truncate_doc_list(doc_list: List[Document], max_tokens_for_docs: int):
    # Pruning context window (input)
    token_pruner = RunnableLambda(lambda doc_list: _truncate_doc_list(...))
```

**Artykuł testuje różne rozmiary okna kontekstowego:**
```python
CONTEXT_WINDOWS = [2048, 4096, 6144, 8192]
```

### Nasza Implementacja:

**Lokalizacja w kodzie:** `main.py`, linie 301-310

```python
self.qa_chain = RetrievalQA.from_chain_type(
    llm=self.llm,
    chain_type="stuff",
    retriever=self.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": MAX_SEARCH_RESULTS, "fetch_k": MAX_SEARCH_RESULTS * 2}
    ),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)
```

**Analiza:**
- ❌ **BRAK**: Nie mamy eksperymentalnej kontroli nad rozmiarem okna kontekstowego
- ❌ **BRAK**: Nie implementujemy token pruning jak w artykule
- ⚠️ Używamy `chain_type="stuff"` - wszystkie chunki są przekazywane do LLM bez kontroli rozmiaru
- ⚠️ Brak mechanizmu `_truncate_doc_list()` do kontrolowanego przycinania kontekstu

**Rekomendacja:** Dodać funkcję token pruning i eksperymentalną kontrolę rozmiaru okna kontekstowego.

---

## 8. Metryki Oceny - Co Brakuje w Naszym Projekcie

### Artykuł używa:

1. **Reference-based Metrics:**
   - BERTScore (semantic similarity)
   - ROUGE-L (content overlap)

2. **LLM-as-a-Judge Metrics:**
   - Factuality/Faithfulness Score (1-5)
   - Response Coherence (1-5)

### Nasza Implementacja:

**Lokalizacja w kodzie:** Brak metryk oceny

**Analiza:**
- ❌ **BRAK**: Nie mamy żadnych metryk oceny jakości odpowiedzi
- ❌ **BRAK**: Nie mamy ground truth answers do porównania
- ❌ **BRAK**: Nie mamy LLM-as-a-Judge evaluation
- ❌ **BRAK**: Nie mamy automatycznych metryk (BERTScore, ROUGE-L)

**Rekomendacja:** Dodać system oceny odpowiedzi:
1. BERTScore i ROUGE-L dla automatycznej oceny
2. LLM-as-a-Judge (GPT-4) dla factuality i coherence
3. Ground truth answers dla pytań testowych

---

## 9. Kompromisy Okna Kontekstowego - Nasze Podejście

### Artykuł omawia:

1. **Koszt obliczeniowy** - O(n²) complexity
2. **"Lost in the middle"** - degradacja uwagi
3. **Prompt injection** - bezpieczeństwo

### Nasza Implementacja:

**1. Koszt obliczeniowy:**
- ⚠️ Używamy `chain_type="stuff"` - wszystkie chunki są przekazywane jednocześnie
- ⚠️ Brak optymalizacji dla długich kontekstów
- ✅ MMR search pomaga z różnorodnością, ale nie redukuje kosztów obliczeniowych

**2. "Lost in the middle":**
- ✅ **ROZWIĄZANE**: Używamy MMR (Maximum Marginal Relevance) zamiast prostego similarity search
  - MMR wybiera różnorodne fragmenty, nie tylko najpodobniejsze
  - `fetch_k=16` pobiera więcej kandydatów przed wyborem top 8
- ✅ Prompt engineering z 12 zasadami pomaga modelowi skupić się na ważnych informacjach

**3. Prompt injection:**
- ⚠️ **BRAK**: Nie mamy specjalnych zabezpieczeń przed prompt injection
- ⚠️ **BRAK**: Nie walidujemy kontekstu przed przekazaniem do LLM

**Kod MMR search:** `main.py`, linie 304-307
```python
retriever=self.vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance - lepsze wyniki
    search_kwargs={"k": MAX_SEARCH_RESULTS, "fetch_k": MAX_SEARCH_RESULTS * 2}
)
```

---

## 10. Eksperymentalna Konfiguracja - Co Możemy Dodać

### Artykuł testuje:
```python
CONTEXT_WINDOWS = [2048, 4096, 6144, 8192]
TEMPERATURE=0.2
TOP_P=1e-10
```

### Nasze Możliwości Eksperymentalne:

**Obecna konfiguracja:** `main.py`, linie 109-124
```python
PDF_FOLDER = os.getenv("PDF_FOLDER", "documents/")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 8))
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
CHAIN_TYPE = os.getenv("CHAIN_TYPE", "stuff")
```

**Co możemy dodać:**
1. ✅ Eksperymentalne rozmiary okna kontekstowego (jak w artykule)
2. ✅ Kontrola nad `top_p` i `top_k`
3. ✅ Kontrola nad `max_tokens` odpowiedzi
4. ✅ Token pruning function
5. ✅ Metryki oceny (BERTScore, ROUGE-L, LLM-as-a-Judge)

---

## 11. Podsumowanie - Mapowanie Kodu do Koncepcji Artykułu

| Koncepcja z Artykułu | Nasza Implementacja | Lokalizacja w Kodzie | Status |
|----------------------|---------------------|----------------------|--------|
| **Chunking** | RecursiveCharacterTextSplitter | `main.py:226-231` | ✅ Implementowane |
| **ChromaDB Storage** | Chroma.from_documents() | `main.py:243-247` | ✅ Implementowane |
| **Embeddings** | OpenAIEmbeddings | `main.py:173-175` | ✅ Implementowane (nowszy model) |
| **LLM Config** | ChatOpenAI | `main.py:177-181` | ⚠️ Częściowo (brak top_p/top_k) |
| **RAG Pipeline** | RetrievalQA | `main.py:301-310` | ✅ Implementowane |
| **MMR Search** | search_type="mmr" | `main.py:304-307` | ✅ Implementowane (lepsze niż artykuł) |
| **Prompt Engineering** | 12 zasad HR | `main.py:271-293` | ✅ Bardziej szczegółowy niż artykuł |
| **Token Pruning** | ❌ Brak | - | ❌ Nie implementowane |
| **Context Window Control** | ❌ Brak | - | ❌ Nie implementowane |
| **Evaluation Metrics** | ❌ Brak | - | ❌ Nie implementowane |
| **Temperature Control** | 0.1 | `main.py:179` | ✅ Implementowane (niższe niż artykuł) |

---

## 12. Rekomendacje Ulepszeń Bazujące na Artykule

### Priorytet 1: Dodanie Kontroli Okna Kontekstowego

**Kod do dodania:**
```python
# W main.py, dodać do konfiguracji
MAX_CONTEXT_TOKENS = int(os.getenv("MAX_CONTEXT_TOKENS", 4096))
CONTEXT_WINDOWS = [2048, 4096, 6144, 8192]  # Dla eksperymentów

# Funkcja token pruning
def _truncate_doc_list(doc_list: List[Document], max_tokens: int) -> List[Document]:
    import tiktoken
    tokenizer = tiktoken.encoding_for_model("gpt-3.5-turbo")
    truncated = []
    current_tokens = 0
    
    for doc in doc_list:
        doc_tokens = len(tokenizer.encode(doc.page_content))
        if current_tokens + doc_tokens <= max_tokens:
            truncated.append(doc)
            current_tokens += doc_tokens
        else:
            break
    return truncated
```

### Priorytet 2: Dodanie Metryk Oceny

**Kod do dodania:**
```python
# Nowy plik: evaluation.py
from evaluate import load
from typing import List, Tuple

def compute_bert_and_rouge(predictions: List[str], references: List[str]) -> Tuple[float, float]:
    bert_score_metric = load("bertscore")
    berts = bert_score_metric.compute(
        predictions=predictions, 
        references=references, 
        lang="pl",  # Polski
        model_type="bert-base-multilingual-cased"
    )
    avg_bert = sum(berts['f1']) / len(berts['f1'])
    
    rouge_metric = load("rouge")
    rouges = rouge_metric.compute(
        predictions=predictions, 
        references=references, 
        rouge_types=['rougeL']
    )
    rouge = rouges['rougeL']
    
    return avg_bert, rouge
```

### Priorytet 3: Rozszerzenie Konfiguracji LLM

**Kod do modyfikacji:** `main.py`, linie 177-181
```python
self.llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=float(os.getenv("TEMPERATURE", 0.1)),
    top_p=float(os.getenv("TOP_P", 1.0)),  # DODAĆ
    max_tokens=int(os.getenv("MAX_TOKENS", 1000)),
    max_retries=int(os.getenv("MAX_RETRIES", 1))  # DODAĆ
)
```

### Priorytet 4: Eksperymentalny Framework

**Nowy endpoint:** `main.py`
```python
@app.post("/api/experiment")
async def run_experiment(experiment_config: ExperimentConfig):
    """Uruchamia eksperyment z różnymi rozmiarami okna kontekstowego"""
    results = {}
    
    for context_window in experiment_config.context_windows:
        # Utwórz RAG chain z kontrolowanym oknem kontekstowym
        rag_chain = create_rag_chain_with_context_limit(
            llm=self.llm,
            vectorstore=self.vectorstore,
            max_context_tokens=context_window
        )
        
        # Testuj na pytaniach
        predictions = []
        for question in experiment_config.questions:
            result = rag_chain.invoke({"query": question})
            predictions.append(result["answer"])
        
        # Oblicz metryki
        bert, rouge = compute_bert_and_rouge(
            predictions, 
            experiment_config.ground_truths
        )
        
        results[context_window] = {
            "bert_score": bert,
            "rouge_l": rouge,
            "predictions": predictions
        }
    
    return results
```

---

## 13. Wnioski

### Co Robimy Dobrze (Lepiej niż Artykuł):

1. ✅ **MMR Search** - używamy Maximum Marginal Relevance zamiast prostego similarity search
2. ✅ **Szczegółowy Prompt** - 12 zasad vs 2 w artykule
3. ✅ **Większy Overlap** - 100 vs 50 znaków (lepsze zachowanie kontekstu)
4. ✅ **Niższa Temperature** - 0.1 vs 0.2 (bardziej deterministyczne)
5. ✅ **Specjalizacja domenowa** - HR-specific vs generic

### Co Powinniśmy Dodać (Z Artykułu):

1. ❌ **Kontrola okna kontekstowego** - eksperymentalne testowanie różnych rozmiarów
2. ❌ **Token pruning** - kontrola rozmiaru kontekstu przed przekazaniem do LLM
3. ❌ **Metryki oceny** - BERTScore, ROUGE-L, LLM-as-a-Judge
4. ❌ **Kontrola sampling** - top_p, top_k
5. ❌ **Eksperymentalny framework** - systematyczne testowanie różnych konfiguracji

### Kompromisy w Naszym Projekcie:

1. ⚠️ **Mniejsze okno kontekstowe** - 8 fragmentów vs 50 w artykule
   - **Plus**: Szybsze, tańsze
   - **Minus**: Może brakować kontekstu dla złożonych pytań

2. ⚠️ **Brak kontroli rozmiaru** - wszystkie chunki przekazywane jednocześnie
   - **Plus**: Prostsze
   - **Minus**: Może przekraczać optymalne okno kontekstowe

3. ⚠️ **Brak metryk** - nie wiemy jak dobre są odpowiedzi
   - **Plus**: Szybszy development
   - **Minus**: Brak danych do optymalizacji

---

## 14. Kod Referencyjny - Konkretne Lokalizacje

### Chunking Strategy
```python
# main.py, linie 226-231
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,      # 500
    chunk_overlap=CHUNK_OVERLAP,  # 100
    length_function=len,
    separators=["\n\n", "\n", ". ", ";", " "]
)
```

### ChromaDB Storage
```python
# main.py, linie 243-247
self.vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=self.embeddings,
    persist_directory=CHROMA_PERSIST_DIR
)
```

### MMR Search (Lepsze niż Artykuł)
```python
# main.py, linie 304-307
retriever=self.vectorstore.as_retriever(
    search_type="mmr",  # Maximum Marginal Relevance
    search_kwargs={"k": MAX_SEARCH_RESULTS, "fetch_k": MAX_SEARCH_RESULTS * 2}
)
```

### Prompt Template (Bardziej Szczegółowy)
```python
# main.py, linie 271-293
prompt_template = """Jesteś ekspertem HR...
[12 szczegółowych zasad]
"""
```

### LLM Configuration
```python
# main.py, linie 177-181
self.llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0.1,
    max_tokens=1000
)
```

---

## 15. Bibliografia i Odniesienia

1. **Lost in the Middle** - Nasz projekt rozwiązuje to przez MMR search
2. **LongBench** - Możemy użyć do benchmarkowania naszego systemu
3. **LLM-as-a-Judge** - Powinniśmy dodać do oceny odpowiedzi

---

**Autor analizy:** AI Assistant  
**Data:** 2025-11-23  
**Wersja projektu:** 2.0.0  
**Artykuł referencyjny:** "Optymalizacja wydajności LLM: dogłębna analiza wpływu okna kontekstowego" - Kuriko IWAI

