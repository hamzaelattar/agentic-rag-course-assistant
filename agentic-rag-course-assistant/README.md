# Agentic RAG Course Assistant

Assistant de cours local : FastAPI sert l’agent LangGraph existant (Qwen3, PyTorch/CUDA, Qdrant, Neo4j et SQLite) à une interface React sombre et responsive.

## Prérequis

- Windows, Python et l’environnement `.venv` du projet.
- Docker Desktop avec Neo4j démarré selon `docker-compose.yml`.
- Node.js 20 ou plus récent.
- Le PDF du cours et l’index Qdrant déjà préparés.
- Variables Neo4j configurées dans `.env` (ne jamais publier ce fichier).

## Installer

À partir de `H:\Rag\agentic-rag-course-assistant` :

```powershell
..\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
cd frontend
npm install
cd ..
```

## Démarrer localement

Terminal 1, un seul worker pour sérialiser les inférences CUDA :

```powershell
..\.venv\Scripts\Activate.ps1
$env:FRONTEND_ORIGIN = "http://localhost:5173"
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Terminal 2 :

```powershell
cd H:\Rag\agentic-rag-course-assistant\frontend
npm run dev
```

Ouvrir `http://localhost:5173`. L’API répond sur `http://127.0.0.1:8000/api/health`.

## Tests

```powershell
python -m pip install pytest
python -m pytest tests/test_api.py -q
cd frontend
npm run build
npm test
```

Les tests API isolent les appels coûteux à l’agent. L’application réelle utilise toujours le graphe LangGraph et la mémoire SQLite existants.

## API

- `GET /api/health`
- `POST /api/conversations` — crée un UUID et son enregistrement SQLite.
- `GET /api/conversations` et `GET /api/conversations/{thread_id}`
- `POST /api/chat` — corps `{ "thread_id": "…", "message": "…" }`

La suppression renvoie `501` : les checkpoints LangGraph et les métadonnées sont dans le même fichier SQLite, mais aucune suppression atomique et sûre n’est implémentée. Le service verrouille les appels d’agent dans un processus pour ne pas lancer deux inférences CUDA simultanément. Pour cette machine, conserver un seul worker Uvicorn.

## Sécurité et limites

Le service est conçu pour un usage local mono-utilisateur. Il n’a pas d’authentification ni d’isolation multi-utilisateur. Ne l’exposez pas sur un réseau public. Le rendu Mermaid utilise le niveau de sécurité strict ; les réponses Markdown ne rendent pas de HTML arbitraire. Le statut de santé indique que l’API répond, pas que chaque service externe ou base est disponible.
