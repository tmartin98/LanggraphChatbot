PDF dokumentum feldolgozás: A rendszer automatikusan feldolgozza és vektorizálja a feltöltött PDF dokumentumokat
Agentic működés: A chatbot autonóm módon tervezi meg a válaszadás lépéseit, bontja részfeladatokra a komplex kérdéseket
RAG technológia: A válaszok generálása a dokumentumból kinyert releváns információk alapján történik
LangGraph workflow: A komplex agentic viselkedés strukturált gráf alapú workflow-val van implementálva
Lokális működés: A rendszer teljesen lokálisan fut, Ollama LLM használatával
Streamlit UI: Könnyen használható, böngésző alapú felhasználói felület

Előfeltételek

Python 3.8+
Ollama telepítve és futtatva
Git

Lépések

Klónozza a repository-t:
git clone https://github.com/yourusername/agentic-rag-chatbot.git
cd agentic-rag-chatbot

Függőségek telepítése:
pip install -r requirements.txt

Indítsa el az Ollama szervert a megfelelő modellel:
ollama run llama3
Megjegyzés: Más modellt is használhat, de akkor frissítse az app.py fájlban a OLLAMA_MODEL változót.
Indítsa el
