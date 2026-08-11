import { Component, type ErrorInfo, type ReactNode } from 'react'
import { buttonClass } from './buttonVariants'
import './ui.css'

type Props = { children: ReactNode }
type State = { error: Error | null }

/**
 * Last line of defence for a render-time throw.
 *
 * Without one, React 19 unmounts the whole tree on an uncaught error and the
 * user is left staring at a blank white document with no way back — the worst
 * possible failure, because it looks like the site is down rather than like
 * something went wrong. Everything recoverable (a failed fetch, a 401, a
 * rejected upload) is already handled where it happens; what reaches here is a
 * bug, so the only honest offer is a reload.
 *
 * Still a class: `getDerivedStateFromError` has no hook equivalent in React 19.
 */
export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null }

  static getDerivedStateFromError(error: Error): State {
    return { error }
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // The browser console is the only sink the frontend has. Sentry is wired on
    // the backend only, so a render crash leaves no trace anywhere else.
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
