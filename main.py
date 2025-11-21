"""
RAG System Backend - FastAPI z OpenAI i ChromaDB
System do analizy dokumentów PDF z użyciem RAG z stuff dla kompendium
ver. 10 - HR Edition z MMR Search
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import os
from datetime import datetime
from collections import defaultdict
import hashlib
import logging
from dotenv import load_dotenv

# Import dla RAG
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Ładowanie zmiennych środowiskowych
load_dotenv()

# Konfiguracja logowania
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="RAG System API", version="2.0.0")

# CORS - pozwala na komunikację z frontendem
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # W produkcji zmień na konkretną domenę
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== Konfiguracja ====================

# Sprawdzenie klucza API
if not os.getenv("OPENAI_API_KEY"):
    logger.error("Brak klucza OPENAI_API_KEY w pliku .env!")
    raise ValueError("Ustaw OPENAI_API_KEY w pliku .env")

# Konfiguracja folderów i parametrów
PDF_FOLDER = os.getenv("PDF_FOLDER", "documents/")
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))  # Zmniejszone dla lepszego map_reduce
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 100))
MAX_SEARCH_RESULTS = int(os.getenv("MAX_SEARCH_RESULTS", 8))  # Więcej fragmentów
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
CHAIN_TYPE = os.getenv("CHAIN_TYPE", "stuff")

# ==================== Modele danych ====================

class Question(BaseModel):
    question: str

class Answer(BaseModel):
    answer: str
    updated_faq: bool = False
    sources: List[str] = []

class FAQItem(BaseModel):
    question: str
    answer: str
    count: int
    last_asked: str
    question_hash: str

# ==================== System RAG z Stuff dla Kompendium ====================

class RAGSystem:
    def __init__(self):
        self.vectorstore = None
        self.qa_chain = None
        self.embeddings = None
        self.llm = None
        self.initialize()

    def initialize(self):
        """Inicjalizacja systemu RAG"""
        logger.info("Inicjalizacja systemu RAG - HR Edition dla kompendium...")

        # Inicjalizacja modeli OpenAI
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY nie jest ustawione w .env")
            
            # Ustaw zmienną środowiskową dla OpenAI (langchain-openai czyta z env)
            os.environ["OPENAI_API_KEY"] = api_key
            
            self.embeddings = OpenAIEmbeddings(
                model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
            )

            self.llm = ChatOpenAI(
                model=OPENAI_MODEL,
                temperature=0.1,  # Niska temperatura dla dokładności
                max_tokens=1000  # Więcej tokenów dla map_reduce
            )
        except Exception as e:
            logger.error(f"Błąd inicjalizacji OpenAI: {e}")
            raise

        # Sprawdź czy istnieje baza wektorowa
        if os.path.exists(CHROMA_PERSIST_DIR):
            logger.info("Ładowanie istniejącej bazy wektorowej...")
            self.load_vectorstore()
        else:
            logger.info("Tworzenie nowej bazy wektorowej...")
            self.create_vectorstore()

        # Utworzenie łańcucha QA z map_reduce
        self.create_qa_chain()

    def create_vectorstore(self):
        """Tworzy nową bazę wektorową z dokumentów PDF"""

        # Sprawdź czy folder z dokumentami istnieje
        if not os.path.exists(PDF_FOLDER):
            os.makedirs(PDF_FOLDER)
            logger.warning(f"Utworzono folder {PDF_FOLDER} - dodaj pliki PDF!")
            return

        # Sprawdź czy są jakieś PDFy
        pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.endswith('.pdf')]
        if not pdf_files:
            logger.warning(f"Brak plików PDF w folderze {PDF_FOLDER}")
            return

        logger.info(f"Znaleziono {len(pdf_files)} plików PDF")

        # Załaduj każdy PDF osobno
        all_chunks = []
        for pdf_file in pdf_files:
            pdf_path = os.path.join(PDF_FOLDER, pdf_file)
            try:
                loader = PyPDFLoader(pdf_path)
                pages = loader.load()
                logger.info(f"✅ Załadowano: {pdf_file} ({len(pages)} stron)")

                # Podziel na mniejsze fragmenty
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=CHUNK_SIZE,
                    chunk_overlap=CHUNK_OVERLAP,
                    length_function=len,
                    separators=["\n\n", "\n", ". ", ";", " "]
                )

                chunks = text_splitter.split_documents(pages)
                all_chunks.extend(chunks)

            except Exception as e:
                logger.error(f"❌ Błąd ładowania {pdf_file}: {e}")
                continue

        logger.info(f"Załadowano łącznie {len(all_chunks)} fragmentów")

        if all_chunks:
            self.vectorstore = Chroma.from_documents(
                documents=all_chunks,
                embedding=self.embeddings,
                persist_directory=CHROMA_PERSIST_DIR
            )
            # W nowszych wersjach langchain_chroma nie ma potrzeby wywoływania persist()
            logger.info("✅ Baza wektorowa utworzona i zapisana")
        else:
            logger.warning("Brak fragmentów do indeksowania")

    def load_vectorstore(self):
        """Ładuje istniejącą bazę wektorową"""
        self.vectorstore = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=self.embeddings
        )
        logger.info("Baza wektorowa załadowana")

    def create_qa_chain(self):
        """Tworzy łańcuch pytanie-odpowiedź z stuff dla kompendium całej wiedzy"""
        
        if not self.vectorstore:
            logger.warning("Brak bazy wektorowej - używam trybu mock")
            return

        # Polski prompt dostosowany do dokumentów HR z lepszymi zasadami dla kompendium
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

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )

        # Utwórz łańcuch QA z stuff ale z lepszymi ustawieniami dla kompendium
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

        logger.info(f"Łańcuch QA - HR Edition (kompendium) utworzony")

    def search(self, question: str) -> dict:
        """Wyszukuje odpowiedź na pytanie używając HR Edition dla kompendium całej wiedzy"""

        if not self.qa_chain:
            # Tryb mock jeśli nie ma dokumentów
            return {
                "answer": "System RAG nie jest jeszcze skonfigurowany. Dodaj dokumenty PDF do folderu 'documents/' i zrestartuj serwer.",
                "sources": []
            }

        try:
            # Wykonaj zapytanie
            result = self.qa_chain({"query": question})

            # Wyodrębnij źródła
            sources = []
            if "source_documents" in result:
                for doc in result["source_documents"]:
                    source = doc.metadata.get("source", "").replace("\\", "/")
                    source = os.path.basename(source)
                    if source and source not in sources:
                        sources.append(source)

            return {
                "answer": result["result"],
                "sources": sources
            }

        except Exception as e:
            logger.error(f"Błąd podczas wyszukiwania: {e}")
            return {
                "answer": f"Wystąpił błąd podczas przetwarzania pytania: {str(e)}",
                "sources": []
            }

    def reset_database(self):
        """Resetuje bazę wektorową"""
        import shutil

        if os.path.exists(CHROMA_PERSIST_DIR):
            shutil.rmtree(CHROMA_PERSIST_DIR)
            logger.info("Baza wektorowa usunięta")

        self.initialize()
        logger.info("Baza wektorowa zresetowana")

# Globalna instancja systemu RAG
rag_system = None

# ==================== Przechowywanie danych FAQ ====================

FAQ_FILE = "faq_data.json"
MAX_FAQ_ITEMS = 9

def load_faq_data():
    """Ładuje dane FAQ z pliku"""
    if os.path.exists(FAQ_FILE):
        try:
            with open(FAQ_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_faq_data(data):
    """Zapisuje dane FAQ do pliku"""
    with open(FAQ_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def get_question_hash(question: str) -> str:
    """Tworzy hash pytania dla identyfikacji podobnych pytań"""
    normalized = question.lower().strip()
    normalized = ''.join(c for c in normalized if c.isalnum() or c.isspace())
    normalized = ' '.join(normalized.split())
    return hashlib.md5(normalized.encode()).hexdigest()

def update_faq(question: str, answer: str) -> bool:
    """Aktualizuje FAQ i zwraca True jeśli lista FAQ się zmieniła"""
    faq_data = load_faq_data()
    question_hash = get_question_hash(question)

    if question_hash in faq_data:
        faq_data[question_hash]['count'] += 1
        faq_data[question_hash]['last_asked'] = datetime.now().isoformat()
        old_position = get_faq_position(faq_data, question_hash)
    else:
        faq_data[question_hash] = {
            'question': question,
            'answer': answer[:200] + "..." if len(answer) > 200 else answer,
            'count': 1,
            'last_asked': datetime.now().isoformat(),
            'question_hash': question_hash
        }
        old_position = None

    sorted_items = sorted(
        faq_data.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )

    if len(sorted_items) > MAX_FAQ_ITEMS:
        items_to_keep = dict(sorted_items[:MAX_FAQ_ITEMS])
        faq_data = items_to_keep
    else:
        faq_data = dict(sorted_items)

    save_faq_data(faq_data)

    new_position = get_faq_position(faq_data, question_hash)
    return old_position != new_position

def get_faq_position(faq_data: dict, question_hash: str) -> Optional[int]:
    """Zwraca pozycję pytania w FAQ"""
    sorted_items = sorted(
        faq_data.items(),
        key=lambda x: x[1]['count'],
        reverse=True
    )
    for i, (hash_key, _) in enumerate(sorted_items):
        if hash_key == question_hash:
            return i
    return None

# ==================== Endpointy API ====================

@app.get("/")
async def root():
    """Serwuje stronę HTML jeśli istnieje, w przeciwnym razie zwraca info o API"""
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())

    return {
        "name": "RAG System API - HR Edition dla kompendium",
        "version": "2.0.0",
        "status": "ready" if rag_system else "initializing",
        "chain_type": CHAIN_TYPE,
        "endpoints": {
            "POST /api/question": "Zadaj pytanie systemowi RAG",
            "GET /api/faq": "Pobierz listę najczęściej zadawanych pytań",
            "GET /health": "Sprawdź status systemu",
            "POST /api/reset": "Resetuj bazę wektorową"
        }
    }

@app.get("/health")
async def health_check():
    """Sprawdzenie statusu systemu"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "rag_ready": rag_system is not None,
        "model": OPENAI_MODEL,
        "chain_type": CHAIN_TYPE,
        "documents_folder": PDF_FOLDER
    }

@app.post("/api/question", response_model=Answer)
async def ask_question(question: Question):
    """Endpoint do zadawania pytań systemowi RAG"""
    try:
        if not question.question or len(question.question.strip()) < 3:
            raise HTTPException(status_code=400, detail="Pytanie jest zbyt krótkie")

        if len(question.question) > 1000:
            raise HTTPException(status_code=400, detail="Pytanie jest zbyt długie (max 1000 znaków)")

        # Pobierz odpowiedź z systemu RAG
        result = rag_system.search(question.question)

        # Aktualizuj FAQ
        faq_updated = update_faq(question.question, result["answer"])

        return Answer(
            answer=result["answer"],
            updated_faq=faq_updated,
            sources=result["sources"]
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Błąd podczas przetwarzania pytania: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/faq", response_model=List[FAQItem])
async def get_faq():
    """Pobiera listę najczęściej zadawanych pytań"""
    try:
        faq_data = load_faq_data()

        sorted_items = sorted(
            faq_data.values(),
            key=lambda x: x['count'],
            reverse=True
        )

        faq_list = []
        for item in sorted_items[:MAX_FAQ_ITEMS]:
            faq_list.append(FAQItem(**item))

        return faq_list

    except Exception as e:
        logger.error(f"Błąd podczas pobierania FAQ: {e}")
        raise HTTPException(status_code=500, detail="Błąd wewnętrzny serwera")

@app.post("/api/reset")
async def reset_database():
    """Resetuje bazę wektorową i przetwarza dokumenty od nowa"""
    try:
        rag_system.reset_database()
        return {"status": "success", "message": "Baza wektorowa została zresetowana"}
    except Exception as e:
        logger.error(f"Błąd podczas resetowania: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/reset-faq")
async def reset_faq():
    """Resetuje system kafelków FAQ - usuwa wszystkie zapisane pytania"""
    try:
        if os.path.exists(FAQ_FILE):
            os.remove(FAQ_FILE)
            logger.info("Plik FAQ został usunięty")
        return {"status": "success", "message": "System kafelków FAQ został zresetowany"}
    except Exception as e:
        logger.error(f"Błąd podczas resetowania FAQ: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# ==================== Uruchomienie ====================

if __name__ == "__main__":
    import uvicorn
    import sys

    print("=" * 60)
    print("🚀 Uruchamianie systemu RAG - HR Edition dla kompendium...")
    print("=" * 60)

    # Sprawdź argumenty CLI
    if "--reset-db" in sys.argv:
        print("🔄 Resetowanie bazy wektorowej...")
        if os.path.exists(CHROMA_PERSIST_DIR):
            import shutil
            shutil.rmtree(CHROMA_PERSIST_DIR)
            print("✅ Baza wektorowa usunięta")

    # Inicjalizacja systemu RAG
    try:
        rag_system = RAGSystem()
        print("✅ System RAG - HR Edition dla kompendium zainicjalizowany")
    except Exception as e:
        print(f"❌ Błąd inicjalizacji RAG: {e}")
        print("Sprawdź czy masz klucz API w pliku .env")
        sys.exit(1)

    print("=" * 60)
    print("📝 Frontend: http://localhost:8000")
    print("📚 API Docs: http://localhost:8000/docs")
    print(f"🔗 Typ łańcucha: {CHAIN_TYPE}")
    print("=" * 60)

    # Uruchom serwer
    uvicorn.run(app, host="0.0.0.0", port=8000)