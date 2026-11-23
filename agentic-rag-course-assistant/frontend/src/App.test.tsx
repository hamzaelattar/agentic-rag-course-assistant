import { afterEach, describe, expect, it, vi } from 'vitest'
import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import App from './App'

afterEach(() => { cleanup(); vi.restoreAllMocks() })

describe('interface du Course Assistant', () => {
  it('affiche l’accueil en français et les suggestions', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }))
    render(<App />)
    expect(screen.getByText(/Bonjour ! Que souhaitez-vous/)).toBeInTheDocument()
    expect(screen.getByText('Expliquer un concept')).toBeInTheDocument()
    expect(screen.getByText('Architecture')).toBeInTheDocument()
  })

  it('remplit le champ depuis une suggestion', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, json: async () => [] }))
    render(<App />)
    await userEvent.click(screen.getByText('Créer un diagramme'))
    expect(screen.getByPlaceholderText('Posez une question sur votre cours...')).toHaveValue('Génère un diagramme du pipeline RAG.')
  })
})
