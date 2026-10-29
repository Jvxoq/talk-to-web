import { Component, type ErrorInfo, type ReactNode } from 'react'
import { buttonClass } from './buttonVariants'
import './ui.css'

type Props = { children: ReactNode }
type State = { error: Error | null }

/**
 * Catches render errors and offers a reload, instead of a blank page.
 *
 * A class, since `getDerivedStateFromError` has no hook equivalent.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // The console is the frontend's only error sink.
    console.error('Unhandled render error', error, info.componentStack)
  }

  render() {
    if (!this.state.error) return this.props.children

    return (
      <div className="crash" role="alert">
        <h1 className="crash__title">Something broke</h1>
        <p className="crash__body">
          The page hit an error it could not recover from. Reloading usually fixes it — your
          conversations and documents are saved on the server, so nothing is lost.
        </p>
        <button
          type="button"
          className={buttonClass('primary')}
          onClick={() => window.location.reload()}
        >
          Reload the page
        </button>
      </div>
    )
  }
}
