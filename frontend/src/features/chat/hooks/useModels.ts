import { useEffect, useState } from 'react'
import { fetchModels } from '../api'
import type { Model } from '../types'

/**
 * Used to seed state before the real list has arrived, and again if the
 * request fails outright — the composer's model select must never be empty.
 */
export const FALLBACK_MODELS: Model[] = ['openai/gpt-oss-120b', 'openai/gpt-oss-20b']
export const FALLBACK_DEFAULT_MODEL: Model = FALLBACK_MODELS[0]

/**
 * Fetches the available models once on mount.
 *
 * The backend is the source of truth for which models exist and can do tool
 * calling — this deliberately holds no hardcoded list beyond the fallback
 * above, so a model going away (as `groq/compound` did) doesn't require a
 * frontend change.
 */
export function useModels() {
  const [models, setModels] = useState<Model[]>(FALLBACK_MODELS)
  const [defaultModel, setDefaultModel] = useState<Model>(FALLBACK_DEFAULT_MODEL)
  const [isLoading, setIsLoading] = useState(true)

  useEffect(() => {
    let cancelled = false
    const controller = new AbortController()

    fetchModels(controller.signal)
      .then((result) => {
        if (cancelled) return
        if (result.models.length > 0) setModels(result.models)
        setDefaultModel(result.default)
      })
      .catch(() => {
        // Swallowed: the fallback list above keeps the composer usable.
      })
      .finally(() => {
        if (!cancelled) setIsLoading(false)
      })

    return () => {
      cancelled = true
      controller.abort()
    }
  }, [])

  return { models, defaultModel, isLoading }
}
