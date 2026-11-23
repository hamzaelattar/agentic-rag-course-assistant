export type Source = { title?: string; url?: string; snippet?: string; source?: string; page?: number; chunk_id?: string; retrieved_via?: string[]; text?: string }
export type Message = { role: 'user' | 'assistant'; content: string; sources?: Source[]; diagram_code?: string | null; selected_tools?: string[] }
export type Conversation = { thread_id: string; title: string; updated_at?: string | null }
export type ChatResult = { thread_id: string; answer: string; rewritten_question: string; selected_tools: string[]; sources: Source[]; diagram_code: string | null; answer_valid: boolean | null; rewrite_attempts: number }
