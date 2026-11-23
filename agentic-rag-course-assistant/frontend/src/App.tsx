import { FormEvent, KeyboardEvent, useEffect, useRef, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import { Activity, ArrowDown, ArrowUp, BookOpen, Check, ChevronDown, CircleHelp, Globe2, Layers3, LoaderCircle, Menu, MessageSquare, Plus, Search, Sparkles, Workflow, X, Copy } from 'lucide-react'
import MermaidRenderer from './MermaidRenderer'
import type { ChatResult, Conversation, Message, Source } from './types'

const API = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'
const suggestions = [
  { icon: BookOpen, title: 'Expliquer un concept', text: 'Explique la self-attention et le rôle des matrices Q, K et V.' },
  { icon: Layers3, title: 'Explorer une page', text: 'Résume la page 3 du cours.' },
  { icon: Workflow, title: 'Créer un diagramme', text: 'Génère un diagramme du pipeline RAG.' },
  { icon: Globe2, title: 'Chercher sur Internet', text: 'Recherche les dernières nouveautés de PyTorch.' },
]
const toolNames: Record<string, string> = { course_rag: 'Course RAG', document_lookup: 'Document Lookup', web_search: 'Web Search', diagram_generator: 'Diagram Generator', direct: 'Direct' }

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, { ...options, headers: { 'Content-Type': 'application/json', ...options?.headers } })
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || `Erreur API (${response.status})`) }
  return response.json()
}

function SourcesPanel({ sources = [] }: { sources?: Source[] }) {
  if (!sources.length) return null
  return <details className="sources"><summary><span><BookOpen size={15}/> Sources utilisées</span><ChevronDown size={15}/></summary><div className="source-list">{sources.map((source, index) => <article className="source-card" key={`${source.url || source.chunk_id || source.source}-${index}`}><div className="source-kind">{source.url ? <><Globe2 size={13}/> WEB SEARCH</> : <><BookOpen size={13}/> COURS · {source.retrieved_via?.join(', ') || 'PDF'}</>}</div><strong>{source.title || source.source || 'Document du cours'}</strong>{source.url && <a href={source.url} target="_blank" rel="noreferrer">{source.url}</a>}{source.page && <span>Page {source.page}{source.chunk_id ? ` · ${source.chunk_id}` : ''}</span>}<p>{source.snippet || source.text || ''}</p></article>)}</div></details>
}

function ActivityPanel({ tools = [] }: { tools?: string[] }) {
  if (!tools.length) return null
  return <details className="activity"><summary><span><Activity size={15}/> Activité de l’agent</span><ChevronDown size={15}/></summary><div className="tool-list">{tools.map(tool => <span className="tool-pill" key={tool}><Check size={13}/>{toolNames[tool] || tool}</span>)}</div></details>
}

function Architecture() {
  return <main className="architecture"><div className="eyebrow">VUE D’ENSEMBLE</div><h1>Architecture de l’agent</h1><p>Le graphe LangGraph orchestre la mémoire, les outils et la vérification de la réponse.</p><div className="arch-grid"><section><h2>Cycle de réponse</h2><MermaidRenderer code={'flowchart TD\n U[Question utilisateur] --> M[Mémoire SQLite]\n M --> R[Query Rewriting]\n R --> P[Planner LangGraph]\n P --> T{Outils sélectionnés}\n T --> C[Course RAG]\n T --> D[Document Lookup]\n T --> W[Web Search]\n T --> G[Diagram Generator]\n T --> A[Réponse directe]\n C --> S[Synthèse]\n D --> S\n W --> S\n G --> S\n A --> V[Answer Grading]\n S --> V\n V -->|Échec| R\n V -->|Succès| F[Réponse finale]'}/></section><section><h2>Course RAG</h2><MermaidRenderer code={'flowchart LR\n Q[Question] --> E[Qwen Embedding]\n E --> X[Qdrant Hybrid Search]\n X --> N[Neo4j Graph Expansion]\n N --> R[Qwen Reranker]\n R --> L[Qwen LLM]'}/></section></div></main>
}

export default function App() {
  const [conversations, setConversations] = useState<Conversation[]>([])
  const [active, setActive] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [busy, setBusy] = useState(false)
  const [online, setOnline] = useState(false)
  const [error, setError] = useState('')
  const [page, setPage] = useState<'chat' | 'architecture'>('chat')
  const [mobileOpen, setMobileOpen] = useState(false)
  const [filter, setFilter] = useState('')
  const [copied, setCopied] = useState<number | null>(null)
  const bottom = useRef<HTMLDivElement>(null)
  const input = useRef<HTMLTextAreaElement>(null)

  useEffect(() => { request<{ status: string }>('/api/health').then(() => setOnline(true)).catch(() => setOnline(false)) }, [])
  useEffect(() => { request<Conversation[]>('/api/conversations').then(setConversations).catch(e => setError(e.message)) }, [])
  useEffect(() => { bottom.current?.scrollIntoView({ behavior: 'smooth', block: 'end' }) }, [messages, busy])

  const openConversation = async (id: string) => {
    setError(''); setPage('chat'); setActive(id); setMobileOpen(false)
    try { const detail = await request<Conversation & { messages: (Message & { content: string })[] }>(`/api/conversations/${id}`); setMessages(detail.messages.map(m => ({ ...m, role: m.role === 'user' ? 'user' : 'assistant' }))) } catch (e) { setError((e as Error).message); setMessages([]) }
  }
  const newConversation = async () => {
    setError('')
    try { const result = await request<{ thread_id: string }>('/api/conversations', { method: 'POST' }); const item = { thread_id: result.thread_id, title: 'Nouvelle conversation' }; setConversations(prev => [item, ...prev]); setActive(item.thread_id); setMessages([]); setPage('chat'); setMobileOpen(false); input.current?.focus() } catch (e) { setError((e as Error).message) }
  }
  const send = async (event?: FormEvent, value = draft) => {
    event?.preventDefault(); const message = value.trim(); if (!message || busy) return
    setError(''); setDraft(''); setMessages(prev => [...prev, { role: 'user', content: message }]); setBusy(true)
    try {
      let threadId = active
      if (!threadId) { const result = await request<{ thread_id: string }>('/api/conversations', { method: 'POST' }); threadId = result.thread_id; setActive(threadId); setConversations(prev => [{ thread_id: threadId!, title: message.slice(0, 72) }, ...prev]) }
      const result = await request<ChatResult>('/api/chat', { method: 'POST', body: JSON.stringify({ thread_id: threadId, message }) })
      setMessages(prev => [...prev, { role: 'assistant', content: result.answer, sources: result.sources, diagram_code: result.diagram_code, selected_tools: result.selected_tools }])
      setConversations(prev => [{ thread_id: threadId!, title: message.slice(0, 72) }, ...prev.filter(item => item.thread_id !== threadId)])
    } catch (e) { setError((e as Error).message); setMessages(prev => [...prev, { role: 'assistant', content: `La demande a échoué : ${(e as Error).message}` }]) } finally { setBusy(false) }
  }
  const onKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => { if (event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); void send() } }
  const copyAnswer = async (text: string, index: number) => { await navigator.clipboard?.writeText(text); setCopied(index); setTimeout(() => setCopied(null), 1500) }
  const visible = conversations.filter(c => c.title.toLowerCase().includes(filter.toLowerCase()))

  return <div className="app-shell"><div className={`sidebar-wrap ${mobileOpen ? 'mobile-open' : ''}`}><aside className="sidebar"><div className="brand"><div className="brand-mark"><Sparkles size={19}/></div><div><b>Agentic RAG</b><small>COURSE ASSISTANT</small></div><button className="mobile-close" onClick={() => setMobileOpen(false)}><X size={18}/></button></div><button className="new-chat" onClick={() => void newConversation()}><Plus size={17}/> Nouvelle conversation</button><div className="side-label">ESPACE DE TRAVAIL</div><button className={`nav-item ${page === 'chat' ? 'selected' : ''}`} onClick={() => setPage('chat')}><MessageSquare size={16}/> Chat</button><button className={`nav-item ${page === 'architecture' ? 'selected' : ''}`} onClick={() => setPage('architecture')}><Workflow size={16}/> Architecture</button><div className="conversations-head"><span>RÉCENTES</span><Search size={14}/></div><div className="search-box"><Search size={14}/><input value={filter} onChange={e => setFilter(e.target.value)} placeholder="Rechercher" /></div><div className="conversation-list">{visible.map(item => <button key={item.thread_id} className={`conversation-item ${active === item.thread_id && page === 'chat' ? 'active' : ''}`} onClick={() => void openConversation(item.thread_id)}><MessageSquare size={14}/><span>{item.title}</span></button>)}{!visible.length && <div className="empty-history">Aucune conversation</div>}</div><div className="sidebar-foot"><div className={`api-status ${online ? 'up' : ''}`}><span/>{online ? 'API connectée' : 'API hors ligne'}</div><span className="local-tag">LOCAL · CUDA</span></div></aside></div>{mobileOpen && <button className="scrim" aria-label="Fermer le menu" onClick={() => setMobileOpen(false)}/>}<section className="main-column"><header className="topbar"><button className="menu-button" onClick={() => setMobileOpen(true)}><Menu size={18}/></button><div className="crumb">{page === 'chat' ? 'Assistant' : 'Architecture'} <span>/</span> <b>{page === 'chat' ? (active ? conversations.find(c => c.thread_id === active)?.title || 'Conversation' : 'Nouvelle session') : 'Système'}</b></div><div className="top-status"><span className={online ? 'dot-green' : 'dot-red'}/>{online ? 'Système opérationnel' : 'Connexion indisponible'}</div></header>{page === 'architecture' ? <Architecture/> : <><main className={`chat-area ${messages.length ? 'has-messages' : ''}`}>{!messages.length ? <div className="welcome"><div className="welcome-icon"><Sparkles size={22}/></div><div className="eyebrow">VOTRE ESPACE D’APPRENTISSAGE</div><h1>Bonjour ! Que souhaitez-vous<br className="desktop-break"/> apprendre aujourd’hui ?</h1><p>Explorez vos cours grâce à un assistant intelligent combinant recherche<br className="desktop-break"/> hybride, graphes de connaissances et intelligence artificielle.</p><div className="suggestions">{suggestions.map(({ icon: Icon, title, text }) => <button key={title} className="suggestion" onClick={() => { setDraft(text); input.current?.focus() }}><span><Icon size={17}/></span><b>{title}</b><small>{text}</small><ArrowDown size={14} className="suggestion-arrow"/></button>)}</div></div> : <div className="messages">{messages.map((message, index) => <article className={`message ${message.role}`} key={`${index}-${message.content.slice(0, 12)}`}><div className="avatar">{message.role === 'user' ? 'A' : <Sparkles size={16}/>}</div><div className="message-body"><div className="message-meta">{message.role === 'user' ? 'VOUS' : 'AGENTIC RAG'} {message.role === 'assistant' && message.selected_tools?.map(tool => <span className="tool-badge" key={tool}>{toolNames[tool] || tool}</span>)}</div>{message.role === 'user' ? <p className="user-text">{message.content}</p> : <><div className="markdown"><ReactMarkdown remarkPlugins={[remarkGfm]}>{message.content}</ReactMarkdown></div>{message.diagram_code && <MermaidRenderer code={message.diagram_code}/>}<ActivityPanel tools={message.selected_tools}/><SourcesPanel sources={message.sources}/><button className="copy-answer" onClick={() => void copyAnswer(message.content, index)}>{copied === index ? <Check size={14}/> : <Copy size={14}/>} {copied === index ? 'Copié' : 'Copier la réponse'}</button></>}</div></article>)}{busy && <div className="thinking"><span className="avatar"><Sparkles size={15}/></span><div><LoaderCircle size={16} className="spin"/> L’agent traite votre demande…</div></div>}<div ref={bottom}/></div>}</main><div className="composer-wrap"><div className="error-banner" hidden={!error}><CircleHelp size={15}/>{error}</div><form className="composer" onSubmit={e => void send(e)}><textarea ref={input} value={draft} onChange={e => setDraft(e.target.value)} onKeyDown={onKeyDown} placeholder="Posez une question sur votre cours..." rows={1} disabled={busy}/><button type="submit" disabled={busy || !draft.trim()} aria-label="Envoyer"><ArrowUp size={18}/></button></form><div className="composer-hint"><span><kbd>Entrée</kbd> envoyer <kbd>⇧ Entrée</kbd> nouvelle ligne</span><span>Les réponses sont générées localement</span></div></div></>}</section></div>
}
