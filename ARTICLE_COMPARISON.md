# Porównanie projektu hhg_baza_wiedzy z artykułem o optymalizacji okna kontekstowego LLM

## Wprowadzenie

Ten dokument szczegółowo odnosi nasz projekt `hhg_baza_wiedzy` do koncepcji i praktyk opisanych w artykule "Optymalizacja wydajności LLM: dogłębna analiza wpływu okna kontekstowego" autorstwa Kuriko IWAI.

---

## 1. Okno kontekstowe i jego wpływ na wydajność

### 1.1 Definicja okna kontekstowego (z artykułu)

> "Okno kontekstu (lub długość kontekstu) definiuje maksymalną liczbę tokenów — wliczając monit wejściowy, wszelkie instrukcje systemowe i wygenerowaną odpowiedź modelu — które LLM może jednocześnie przetworzyć i obsłużyć podczas pętli autoregresyjnej."

### 1.2 Nasza implementacja

**Lokalizacja w kodzie:** `main.py`, linie 177-181

```python
self.llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0.1,  # Niska temperatura dla dokładności
    max_tokens=1000  # Więcej tokenów dla map_reduce
)
```

**Analiza:**
- ✅ Ustawiamy `max_tokens=1000` - limit tokenów wyjściowych
- ❌ **BRAK**: Nie kontrolujemy bezpośrednio rozmiaru okna kontekstowego wejściowego
- ⚠️ **RÓŻNICA**: Artykuł testuje różne rozmiary okna (2048, 4096, 6144, 8192), my używamy domyślnego okna GPT-3.5-turbo (~4096 tokenów)

**Rekomendacja:**
- Dodać parametr `max_context_tokens` do konfiguracji
- Implementować `token_pruner` jak w artykule (linie 300-310 w artykule)

---

## 2. Chunking dokumentów

### 2.1 Strategia chunkingu z artykułu

**Kod z artykułu:**
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, 
    chunk_overlap=50, 
    separators=["\n\n", "\n", " ", ""]
)
```

### 2.2 Nasza implementacja

**Lokalizacja w kodzie:** `main.py`, linie 226-231

```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,  # 500 znaków (z .env)
    chunk_overlap=CHUNK_OVERLAP,  # 100 znaków (z .env)
    length_function=len,
    separators=["\n\n", "\n", ". ", ";", " "]
)
```

**Porównanie:**

| Aspekt | Artykuł | Nasz projekt | Analiza |
|--------|---------|--------------|---------|
| `chunk_size` | 512 znaków | 500 znaków | ✅ **PODOBNE** - oba w zakresie 500-512 |
| `chunk_overlap` | 50 znaków | 100 znaków | ⚠️ **RÓŻNICA** - używamy większego overlap (20% vs ~10%) |
| `separators` | `["\n\n", "\n", " ", ""]` | `["\n\n", "\n", ". ", ";", " "]` | ✅ **PODOBNE** - oba używają hierarchicznego podziału |
| `length_function` | domyślna (tokeny?) | `len` (znaki) | ⚠️ **RÓŻNICA** - artykuł może używać tokenizer, my znaki |

**Analiza:**
- ✅ Używamy tej samej biblioteki: `RecursiveCharacterTextSplitter`
- ✅ Podobna strategia separators (hierarchiczna)
- ⚠️ Większy overlap (100 vs 50) - może być lepszy dla zachowania kontekstu, ale kosztowniejszy
- ❌ **BRAK**: Nie używamy tokenizera do precyzyjnego liczenia tokenów

**Rekomendacja:**
- Rozważyć użycie `tiktoken` do precyzyjnego liczenia tokenów (mamy już `tiktoken==0.8.0` w requirements.txt)
- Przetestować różne wartości `chunk_overlap` (50, 100, 150)

---

## 3. Przechowywanie dokumentów w ChromaDB

### 3.1 Implementacja z artykułu

**Kod z artykułu:**
```python
vector_db = Chroma.from_documents(
    documents=document_chunks, 
    embedding=embedding_model, 
    persist_directory="./_rag_db_chroma"
)
```

### 3.2 Nasza implementacja

**Lokalizacja w kodzie:** `main.py`, linie 243-247

```python
self.vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=self.embeddings,
    persist_directory=CHROMA_PERSIST_DIR
)
```

**Porównanie:**

| Aspekt | Artykuł | Nasz projekt | Analiza |
|--------|---------|--------------|---------|
| Biblioteka | `langchain_community.vectorstores.Chroma` | `langchain_chroma.Chroma` | ⚠️ **RÓŻNICA** - różne pakiety, ale ta sama funkcjonalność |
| Embeddings | `OpenAIEmbeddings(model='text-embedding-ada-002')` | `OpenAIEmbeddings(model='text-embedding-3-small')` | ⚠️ **RÓŻNICA** - nowszy model embeddings |
| Persist directory | `./_rag_db_chroma` | `./chroma_db` (z .env) | ✅ **PODOBNE** - oba zapisują lokalnie |

**Analiza:**
- ✅ Używamy tej samej koncepcji: `Chroma.from_documents()`
- ✅ Oba zapisują do lokalnego katalogu
- ⚠️ Używamy nowszego modelu embeddings (`text-embedding-3-small` vs `text-embedding-ada-002`)

**Rekomendacja:**
- Rozważyć testowanie różnych modeli embeddings (ada-002 vs 3-small vs 3-large)

---

## 4. Konfiguracja LLM

### 4.1 Konfiguracja z artykułu

**Kod z artykułu:**
```python
TEMPERATURE=0.2
TOP_P=1e-10
TOP_K=1.0
MAX_RESPONSE_TOKENS = 2048
```

### 4.2 Nasza implementacja

**Lokalizacja w kodzie:** `main.py`, linie 177-181

```python
self.llm = ChatOpenAI(
    model=OPENAI_MODEL,  # gpt-3.5-turbo
    temperature=0.1,  # Niska temperatura dla dokładności
    max_tokens=1000  # Więcej tokenów dla map_reduce
)
```

**Porównanie:**

| Parametr | Artykuł | Nasz projekt | Analiza |
|----------|---------|--------------|---------|
| `temperature` | 0.2 | 0.1 | ✅ **PODOBNE** - oba niskie (dokładność) |
| `top_p` | 1e-10 | domyślne (~1.0) | ⚠️ **RÓŻNICA** - artykuł używa greedy sampling |
| `top_k` | 1.0 | domyślne | ⚠️ **RÓŻNICA** - artykuł używa greedy sampling |
| `max_tokens` | 2048 | 1000 | ⚠️ **RÓŻNICA** - mniejszy limit wyjściowy |
| Model | Llama 3.1 8B | GPT-3.5-turbo | ⚠️ **RÓŻNICA** - różne modele |

**Analiza:**
- ✅ Oba używają niskiej temperatury dla dokładności
- ⚠️ Artykuł używa greedy sampling (top_p=1e-10, top_k=1.0) dla deterministycznych wyników
- ⚠️ Nasz model ma mniejszy limit wyjściowy (1000 vs 2048)

**Rekomendacja:**
- Rozważyć dodanie `top_p` i `top_k` do konfiguracji dla bardziej deterministycznych wyników
- Zwiększyć `max_tokens` do 2048 dla dłuższych odpowiedzi (kompendium)

---

## 5. RAG Pipeline - Wyszukiwanie i Retrieval

### 5.1 Implementacja z artykułu

**Kod z artykułu:**
```python
def create_rag_chain(
    llm,
    max_doc_tokens: int,  # for exp - context window size
    vector_db: Chroma = vector_db, 
    k: int = 50, 
    prompt_template: ChatPromptTemplate = rag_prompt_template
):
    retriever = vector_db.as_retriever(search_kwargs={'k': k})
    token_pruner = RunnableLambda(lambda doc_list: _truncate_doc_list(...))
    # ... reszta pipeline
```

**Kluczowe elementy:**
- `k=50` - pobiera 50 dokumentów
- `token_pruner` - przycina kontekst do `max_doc_tokens`
- Testuje różne `max_doc_tokens`: [2048, 4096, 6144, 8192]

### 5.2 Nasza implementacja

**Lokalizacja w kodzie:** `main.py`, linie 301-310

```python
self.qa_chain = RetrievalQA.from_chain_type(
    llm=self.llm,
    chain_type="stuff",
    retriever=self.vectorstore.as_retriever(
        search_type="mmr",  # Maximum Marginal Relevance - lepsze wyniki
        search_kwargs={"k": MAX_SEARCH_RESULTS, "fetch_k": MAX_SEARCH_RESULTS * 2}
    ),
    chain_type_kwargs={"prompt": PROMPT},
    return_source_documents=True
)
```

**Konfiguracja:** `main.py`, linie 119-121
```python
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 8))  # Więcej fragmentów
```

**Porównanie:**

| Aspekt | Artykuł | Nasz projekt | Analiza |
|--------|---------|--------------|---------|
| Liczba dokumentów (`k`) | 50 | 8 | ⚠️ **RÓŻNICA** - znacznie mniej dokumentów |
| Search type | domyślne (similarity) | `mmr` (Maximum Marginal Relevance) | ✅ **LEPSZE** - MMR zapewnia większą różnorodność |
| `fetch_k` | brak | `MAX_SEARCH_RESULTS * 2` (16) | ✅ **LEPSZE** - MMR wymaga większego zbioru kandydatów |
| Token pruning | ✅ `_truncate_doc_list()` | ❌ **BRAK** | ❌ **BRAK** - nie kontrolujemy rozmiaru kontekstu |
| Testowanie różnych okien | ✅ [2048, 4096, 6144, 8192] | ❌ **BRAK** | ❌ **BRAK** - nie testujemy różnych rozmiarów |

**Analiza:**
- ✅ Używamy MMR search - lepsze niż proste similarity search
- ⚠️ Pobieramy tylko 8 dokumentów vs 50 w artykule
- ❌ **BRAK**: Nie mamy token prunera - nie kontrolujemy rozmiaru kontekstu
- ❌ **BRAK**: Nie testujemy różnych rozmiarów okna kontekstowego

**Rekomendacja:**
- Dodać `token_pruner` do kontroli rozmiaru kontekstu
- Rozważyć zwiększenie `k` do 20-30 (kompromis między jakością a kosztem)
- Zaimplementować testowanie różnych rozmiarów okna kontekstowego

---

## 6. Prompt Template

### 6.1 Prompt z artykułu

**Kod z artykułu:**
```python
system_prompt = (
    "You are an expert answer generator. Based ONLY on the following context, answer the complex, multi-hop question from the user."
    "If you cannot find the answer in the provided context, state that you do not have enough information."
    "\n\nContext:\n{context}"
)
```

### 6.2 Nasz prompt

**Lokalizacja w kodzie:** `main.py`, linie 270-293

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

| Aspekt | Artykuł | Nasz projekt | Analiza |
|--------|---------|--------------|---------|
| Język | Angielski | Polski | ⚠️ **RÓŻNICA** - różne języki |
| Długość | Krótki (~3 linie) | Długi (12 zasad) | ⚠️ **RÓŻNICA** - nasz jest bardziej szczegółowy |
| Instrukcje | Ogólne | Specyficzne dla HR | ✅ **LEPSZE** - bardziej ukierunkowane |
| Fallback | "do not have enough information" | "Nie znalazłem tej informacji w dokumentach" | ✅ **PODOBNE** - oba mają fallback |
| Format | `{context}` + `{question}` | `{context}` + `{question}` | ✅ **PODOBNE** - ten sam format |

**Analiza:**
- ✅ Oba używają tego samego formatu: `{context}` + `{question}`
- ✅ Oba mają instrukcję fallback gdy brak informacji
- ✅ Nasz prompt jest bardziej szczegółowy i ukierunkowany na HR
- ⚠️ Dłuższy prompt = więcej tokenów w oknie kontekstowym

**Rekomendacja:**
- Monitorować rozmiar promptu (może zajmować ~500-800 tokenów)
- Rozważyć skrócenie promptu jeśli okno kontekstowe jest ograniczone

---

## 7. Metryki oceny

### 7.1 Metryki z artykułu

Artykuł używa:
1. **Reference-based metrics:**
   - BERTScore (semantic similarity)
   - ROUGE-L (content overlap, fluency)

2. **LLM-as-a-Judge metrics:**
   - Factuality/Faithfulness (1-5)
   - Response Coherence (1-5)

### 7.2 Nasza implementacja

**Lokalizacja w kodzie:** ❌ **BRAK**

**Analiza:**
- ❌ **BRAK**: Nie mamy żadnych metryk oceny jakości odpowiedzi
- ❌ **BRAK**: Nie mamy ground truth do porównania
- ❌ **BRAK**: Nie mamy LLM-as-a-Judge

**Rekomendacja:**
- Zaimplementować podstawowe metryki (BERTScore, ROUGE-L)
- Dodać endpoint `/api/evaluate` do oceny odpowiedzi
- Rozważyć LLM-as-a-Judge dla oceny factuality i coherence

---

## 8. Kompromisy związane z oknem kontekstowym

### 8.1 Kompromisy z artykułu

Artykuł wymienia:
1. **Koszt obliczeniowy** - O(n²) complexity
2. **"Lost in the middle"** - degradacja uwagi
3. **Prompt injection** - większe okno = więcej miejsca na ataki

### 8.2 Nasze podejście

**Lokalizacja w kodzie:** `main.py`, linie 301-310

**Analiza:**
- ⚠️ Używamy `chain_type="stuff"` - wszystkie dokumenty w jednym prompt
- ⚠️ Nie kontrolujemy rozmiaru okna kontekstowego
- ⚠️ Nie mamy mechanizmu obsługi "lost in the middle"

**Rekomendacja:**
- Rozważyć `chain_type="map_reduce"` dla bardzo długich kontekstów
- Dodać `token_pruner` do kontroli rozmiaru
- Monitorować pozycję ważnych informacji w kontekście

---

## 9. Eksperyment z różnymi rozmiarami okna kontekstowego

### 9.1 Eksperyment z artykułu

Artykuł testuje: `C = {2048, 4096, 6144, 8192}` tokenów

**Wyniki:**
- Optymalne okno: **8192 tokeny**
- Najwyższe wyniki w BERT, ROUGE-L, Coherence
- Factuality rośnie z rozmiarem okna

### 9.2 Nasz projekt

**Lokalizacja w kodzie:** ❌ **BRAK eksperymentów**

**Analiza:**
- ❌ Nie testujemy różnych rozmiarów okna kontekstowego
- ❌ Nie mamy metryk do porównania
- ❌ Używamy domyślnego okna GPT-3.5-turbo (~4096 tokenów)

**Rekomendacja:**
- Zaimplementować eksperyment podobny do artykułu
- Testować różne wartości `max_doc_tokens`
- Zbierać metryki dla każdej konfiguracji

---

## 10. Konkretne fragmenty kodu - Mapowanie

### 10.1 Chunking (Artykuł vs Nasz kod)

**Artykuł:**
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=512, chunk_overlap=50, 
    separators=["\n\n", "\n", " ", ""]
)
```

**Nasz kod:** `main.py`, linie 226-231
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,  # 500
    chunk_overlap=CHUNK_OVERLAP,  # 100
    length_function=len,
    separators=["\n\n", "\n", ". ", ";", " "]
)
```

**Korespondencja:** ✅ **TAK** - używamy tej samej biblioteki i podobnej strategii

---

### 10.2 ChromaDB Storage (Artykuł vs Nasz kod)

**Artykuł:**
```python
vector_db = Chroma.from_documents(
    documents=document_chunks, 
    embedding=embedding_model, 
    persist_directory="./_rag_db_chroma"
)
```

**Nasz kod:** `main.py`, linie 243-247
```python
self.vectorstore = Chroma.from_documents(
    documents=all_chunks,
    embedding=self.embeddings,
    persist_directory=CHROMA_PERSIST_DIR
)
```

**Korespondencja:** ✅ **TAK** - identyczna koncepcja i API

---

### 10.3 RAG Chain (Artykuł vs Nasz kod)

**Artykuł:**
```python
retriever = vector_db.as_retriever(search_kwargs={'k': 50})
token_pruner = RunnableLambda(lambda doc_list: _truncate_doc_list(...))
rag_chain = create_rag_chain(llm, max_doc_tokens, vector_db, k=50)
```

**Nasz kod:** `main.py`, linie 301-310
```python
retriever=self.vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": MAX_SEARCH_RESULTS, "fetch_k": MAX_SEARCH_RESULTS * 2}
)
self.qa_chain = RetrievalQA.from_chain_type(...)
```

**Korespondencja:** ⚠️ **CZĘŚCIOWO** - używamy innego API (RetrievalQA vs custom chain), ale podobna koncepcja

---

### 10.4 LLM Configuration (Artykuł vs Nasz kod)

**Artykuł:**
```python
llm = ChatOpenAI(
    temperature=0.2,
    top_p=1e-10,
    max_completion_tokens=2048,
)
```

**Nasz kod:** `main.py`, linie 177-181
```python
self.llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0.1,
    max_tokens=1000
)
```

**Korespondencja:** ⚠️ **CZĘŚCIOWO** - podobne parametry, ale różne wartości

---

## 11. Wnioski i rekomendacje

### 11.1 Co mamy dobrze

1. ✅ Używamy MMR search (lepsze niż proste similarity)
2. ✅ Szczegółowy prompt template dla HR
3. ✅ Podobna strategia chunking
4. ✅ ChromaDB z persist directory
5. ✅ Niska temperatura dla dokładności

### 11.2 Co możemy ulepszyć (na podstawie artykułu)

1. ❌ **Dodać token pruner** - kontrola rozmiaru okna kontekstowego
2. ❌ **Zwiększyć k** - więcej dokumentów (8 → 20-30)
3. ❌ **Zaimplementować metryki** - BERTScore, ROUGE-L, LLM-as-a-Judge
4. ❌ **Testować różne okna** - eksperyment z [2048, 4096, 6144, 8192]
5. ❌ **Dodać greedy sampling** - top_p=1e-10 dla determinizmu
6. ⚠️ **Zwiększyć max_tokens** - 1000 → 2048 dla dłuższych odpowiedzi
7. ⚠️ **Użyć tokenizera** - precyzyjne liczenie tokenów zamiast znaków

### 11.3 Priorytety implementacji

**Wysoki priorytet:**
1. Token pruner dla kontroli okna kontekstowego
2. Zwiększenie `k` do 20-30
3. Zwiększenie `max_tokens` do 2048

**Średni priorytet:**
4. Implementacja podstawowych metryk (BERTScore, ROUGE-L)
5. Testowanie różnych rozmiarów okna kontekstowego

**Niski priorytet:**
6. LLM-as-a-Judge dla factuality/coherence
7. Greedy sampling (top_p, top_k)

---

## 12. Kod do implementacji (na podstawie artykułu)

### 12.1 Token Pruner

```python
import tiktoken

def _truncate_doc_list(doc_list: List[Document], max_tokens_for_docs: int) -> List[Document]:
    """Przycina listę dokumentów do maksymalnej liczby tokenów"""
    tokenizer = tiktoken.encoding_for_model("gpt-4")
    total_tokens = 0
    truncated_docs = []
    
    for doc in doc_list:
        doc_tokens = len(tokenizer.encode(doc.page_content))
        if total_tokens + doc_tokens <= max_tokens_for_docs:
            truncated_docs.append(doc)
            total_tokens += doc_tokens
        else:
            # Przycinamy ostatni dokument jeśli potrzeba
            remaining_tokens = max_tokens_for_docs - total_tokens
            if remaining_tokens > 100:  # Minimum 100 tokenów
                truncated_content = tokenizer.decode(
                    tokenizer.encode(doc.page_content)[:remaining_tokens]
                )
                truncated_doc = Document(
                    page_content=truncated_content,
                    metadata=doc.metadata
                )
                truncated_docs.append(truncated_doc)
            break
    
    return truncated_docs
```

**Lokalizacja w naszym kodzie:** Dodać do `main.py` jako funkcję pomocniczą, użyć w `create_qa_chain()`

---

### 12.2 Eksperyment z różnymi oknami

```python
CONTEXT_WINDOWS = [2048, 4096, 6144, 8192]

for max_doc_tokens in CONTEXT_WINDOWS:
    # Utwórz retriever z token pruner
    retriever = self.vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 30, "fetch_k": 60}
    )
    
    # Dodaj token pruner
    token_pruner = RunnableLambda(
        lambda doc_list: _truncate_doc_list(doc_list, max_doc_tokens)
    )
    
    # Testuj różne konfiguracje
    # ...
```

**Lokalizacja w naszym kodzie:** Dodać jako opcjonalny eksperymentalny tryb w `RAGSystem`

---

### 12.3 Metryki oceny

```python
from evaluate import load

def evaluate_answer(prediction: str, reference: str) -> dict:
    """Ocenia odpowiedź używając BERTScore i ROUGE-L"""
    bert_score_metric = load("bertscore")
    rouge_metric = load("rouge")
    
    berts = bert_score_metric.compute(
        predictions=[prediction],
        references=[reference],
        lang="en",
        model_type="bert-base-uncased"
    )
    
    rouges = rouge_metric.compute(
        predictions=[prediction],
        references=[reference],
        rouge_types=['rougeL']
    )
    
    return {
        "bert_score": sum(berts['f1']) / len(berts['f1']),
        "rouge_l": rouges['rougeL']
    }
```

**Lokalizacja w naszym kodzie:** Dodać jako nowy moduł `evaluation.py` lub endpoint `/api/evaluate`

---

## 13. Podsumowanie mapowania

| Koncepcja z artykułu | Nasz kod | Status |
|---------------------|----------|--------|
| Chunking z RecursiveCharacterTextSplitter | ✅ `main.py:226-231` | ✅ Zaimplementowane |
| ChromaDB storage | ✅ `main.py:243-247` | ✅ Zaimplementowane |
| RAG pipeline | ✅ `main.py:301-310` | ✅ Zaimplementowane (ale inny API) |
| LLM configuration | ✅ `main.py:177-181` | ✅ Zaimplementowane |
| Prompt template | ✅ `main.py:270-293` | ✅ Zaimplementowane |
| Token pruner | ❌ Brak | ❌ **DO DODANIA** |
| Testowanie różnych okien | ❌ Brak | ❌ **DO DODANIA** |
| Metryki oceny | ❌ Brak | ❌ **DO DODANIA** |
| Greedy sampling | ❌ Brak | ⚠️ **OPCJONALNE** |
| Większe k (50) | ⚠️ k=8 | ⚠️ **DO ZWIĘKSZENIA** |

---

## 14. Bibliografia i referencje

Artykuł odnosi się do:
- "Lost in the Middle: How Language Models Use Long Contexts" (arXiv:2307.03172)
- "LongBench: A Bilingual, Multitask Benchmark for Long Context Understanding" (arXiv: 2308.14508)
- "An Empirical Study of LLM-as-a-Judge for LLM Evaluation" (arXiv: 2403.02839)

**Nasze rekomendacje:**
- Przeczytać "Lost in the Middle" - problem degradacji uwagi w długich kontekstach
- Rozważyć implementację technik z LongBench dla lepszej obsługi długich kontekstów

---

*Dokument utworzony: 2025-11-23*
*Ostatnia aktualizacja: 2025-11-23*

