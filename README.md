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

1,
  git clone https://github.com/tmartin98/LanggraphChatbot.git

2,
  pip install -r requirements.txt

3,
  ollama run llama3.2

4,
  streamlit run streamlit_app.py
