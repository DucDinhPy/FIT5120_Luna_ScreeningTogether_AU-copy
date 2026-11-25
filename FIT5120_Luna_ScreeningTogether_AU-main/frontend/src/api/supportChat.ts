export interface SupportTurn {
  question: string
  answer: string
}

export interface SupportSource {
  title: string
  url: string | null
  publisher: string | null
  page: number | null
}

interface SupportReply {
  answer: string
  sources: SupportSource[]
}

const API_BASE_URL = (
  import.meta.env.VITE_API_BASE_URL?.trim() ||
  (import.meta.env.DEV ? 'http://127.0.0.1:8000' : 'https://api.screeningtogether.app')
).replace(/\/+$/, '')

export async function askSupport(
  message: string,
  history: SupportTurn[],
  signal: AbortSignal,
): Promise<SupportReply | null> {
  const response = await fetch(`${API_BASE_URL}/support-chat`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message, history }),
    signal,
  })
  if (response.status === 503) return null
  if (!response.ok) throw new Error('Support chat request failed')
  return response.json()
}
