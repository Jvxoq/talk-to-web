import { useEffect } from 'react'
import { AnimatePresence, motion, useReducedMotion } from 'motion/react'
import { FileText, RefreshCw, Trash2, X } from 'lucide-react'
import { buttonClass, IconButton } from '../../../components/ui'
import { easeStandard, springs, timing } from '../../../lib/motion'
import { useDocuments } from '../hooks/useDocuments'

interface DocumentManagerProps {
  open: boolean
  onClose: () => void
}

/**
 * The document manager: every file this account has uploaded, with a delete
 * per row.
 *
 * A dialog rather than a sidebar tab - documents are managed rarely compared
 * to how often a conversation is switched, so it does not need to compete for
 * permanent screen space the way the conversation list does.
 */
export function DocumentManager({ open, onClose }: DocumentManagerProps) {
  const { documents, isLoading, error, deletingId, refresh, remove } = useDocuments()
  const reduce = useReducedMotion() ?? false

  // Loaded fresh every time the dialog opens, rather than kept warm in the
  // background - indexing finishes after the upload request returns, so a
  // list fetched once at sign-in would show yesterday's chunk counts.
  useEffect(() => {
    if (open) void refresh()
  }, [open, refresh])

  useEffect(() => {
    if (!open) return
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') onClose()
    }
    document.addEventListener('keydown', onKeyDown)
    return () => document.removeEventListener('keydown', onKeyDown)
  }, [open, onClose])

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="modal-backdrop"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: timing.fast, ease: easeStandard }}
          onClick={onClose}
        >
          <motion.div
            className="modal"
            role="dialog"
            aria-modal="true"
            aria-labelledby="document-manager-title"
            initial={{ opacity: 0, y: reduce ? 0 : 12, scale: reduce ? 1 : 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: reduce ? 0 : 8, scale: reduce ? 1 : 0.98 }}
            transition={springs.card}
            onClick={(event) => event.stopPropagation()}
          >
            <div className="modal-header">
              <h2 id="document-manager-title">Documents</h2>
              <div className="modal-header-actions">
                <IconButton
                  variant="ghost"
                  label="Refresh documents"
                  icon={<RefreshCw strokeWidth={2} aria-hidden="true" />}
                  onClick={() => void refresh()}
                  disabled={isLoading}
                />
                <IconButton
                  variant="ghost"
                  label="Close"
                  icon={<X strokeWidth={2} aria-hidden="true" />}
                  onClick={onClose}
                />
              </div>
            </div>

            {error && (
              <p className="modal-error" role="alert">
                {error}
              </p>
            )}

            {documents.length === 0 ? (
              <p className="modal-empty">{isLoading ? 'Loading…' : 'No documents uploaded yet.'}</p>
            ) : (
              <ul className="document-list">
                <AnimatePresence initial={false}>
                  {documents.map((document) => (
                    <motion.li
                      key={document.id}
                      className="document-item"
                      initial={{ opacity: 0, y: reduce ? 0 : -6 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      transition={springs.card}
                    >
                      <FileText
                        className="document-icon"
                        strokeWidth={2}
                        aria-hidden="true"
                      />
                      <span className="document-name">{document.name}</span>
                      <span className="document-status">
                        {document.chunksIndexed > 0 ? `${document.chunksIndexed} chunks` : 'Indexing…'}
                      </span>
                      <motion.button
                        type="button"
                        className={buttonClass('ghost', { icon: true, className: 'document-delete' })}
                        onClick={() => void remove(document.id)}
                        disabled={deletingId === document.id}
                        aria-label={`Delete ${document.name}`}
                        title="Delete document"
                        whileHover={{ x: -1, y: -1 }}
                        whileTap={{ x: 0, y: 0 }}
                        transition={springs.press}
                      >
                        <Trash2 strokeWidth={2} aria-hidden="true" />
                      </motion.button>
                    </motion.li>
                  ))}
                </AnimatePresence>
              </ul>
            )}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  )
}
