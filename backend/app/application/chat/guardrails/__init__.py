"""Application-layer guardrail composition.

The domain module (`app.domain.chat.guardrails`) owns the detectors; this
package owns the policy decisions built on top of them - what to do with a
finding, and how a tool result gets fenced off from the model. Nothing here
constructs a detector's regex or a toggle's default; those come in as
constructor arguments so this layer stays free of `app.settings`.
"""
