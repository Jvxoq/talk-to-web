import { useEffect, useState } from 'react'
import { fetchModels } from '../api'
import type { Model } from '../types'

/**
 * Used before the list loads or if it fails, so the select is never empty.
 */
export const FALLBACK_MODELS: Model[] = ['qwen/qwen3.6-27b']
export const FALLBACK_DEFAULT_MODEL: Model = FALLBACK_MODELS[0]

/**
 * Fetches the available models once on mount. The backend owns the list.
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
