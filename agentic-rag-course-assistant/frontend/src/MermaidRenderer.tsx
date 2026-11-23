import { useEffect, useId, useState } from 'react'
import mermaid from 'mermaid'
import { Copy, Maximize2 } from 'lucide-react'

mermaid.initialize({ startOnLoad: false, securityLevel: 'strict', theme: 'dark', themeVariables: { darkMode: true, background: '#151d2b', primaryColor: '#24213d', primaryTextColor: '#f3f6fc', lineColor: '#8b5cf6', secondaryColor: '#172b3b' } })

export default function MermaidRenderer({ code }: { code: string }) {
  const id = useId().replace(/:/g, '')
  const [svg, setSvg] = useState('')
  const [error, setError] = useState(false)
  const [large, setLarge] = useState(false)
  useEffect(() => { let alive = true; mermaid.render(`diagram-${id}`, code).then(({ svg: out }) => { if (alive) { setSvg(out); setError(false) } }).catch(() => { if (alive) setError(true) }); return () => { alive = false } }, [code, id])
  const copy = () => navigator.clipboard?.writeText(code)
  return <div className={`diagram ${large ? 'diagram-large' : ''}`}><div className="diagram-tools"><span>Diagramme Mermaid</span><button onClick={copy} aria-label="Copier le code"><Copy size={15}/></button><button onClick={() => setLarge(!large)} aria-label="Agrandir"><Maximize2 size={15}/></button></div>{error ? <pre className="diagram-fallback">{code}</pre> : <div className="diagram-svg" dangerouslySetInnerHTML={{ __html: svg }}/>}</div>
}
