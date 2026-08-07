const FENCE = /^\s*(```|~~~)/
const DELIMITER_ROW = /^\s*\|?[\s:|-]+\|[\s:|-]*$/
const SETEXT_RULE = /^\s*(-{1,}|={1,}|\*{3,}|_{3,})\s*$/

/** Lines like `| a | b |` — the shape both table rows and their headers share. */
function isTableLine(line: string | undefined): boolean {
  return line !== undefined && line.trimStart().startsWith('|')
}

/**
 * Hides a table that hasn't got its delimiter row yet.
 *
 * A table only becomes a table once the `|---|---|` line under the header is
 * complete. Before that the parser sees an ordinary paragraph, so the header
 * flashes on screen as a line of raw pipes.
 */
function dropIncompleteTable(lines: string[]): string[] {
  let start = lines.length
  while (start > 0 && isTableLine(lines[start - 1])) start--
  if (start === lines.length) return lines

  // The delimiter is only known to be finished once a further line exists —
  // the last line is always the one mid-stream truncation cut in half.
  const settled = DELIMITER_ROW.test(lines[start + 1] ?? '') && lines.length > start + 2
  return settled ? lines : lines.slice(0, start)
}

/**
 * Makes a half-written markdown string safe to render.
 *
 * Streaming cuts the text at an arbitrary token, which leaves blocks hanging
 * open: an unclosed code fence swallows the rest of the bubble, a lone `---`
 * retroactively turns the paragraph above it into a heading, and a table with
 * no delimiter row shows up as raw pipes. Each settles a few tokens later, so
 * the job here is only to keep the in-between frames from looking broken.
 */
export function normalizeStreamingMarkdown(text: string): string {
  let lines = text.split('\n')

  // A trailing rule is ambiguous until the line after it arrives: on its own it
  // reads as a setext underline for the paragraph above.
  if (SETEXT_RULE.test(lines[lines.length - 1] ?? '')) lines = lines.slice(0, -1)

  lines = dropIncompleteTable(lines)

  const fences = lines.filter((line) => FENCE.test(line)).length
  if (fences % 2 === 1) lines = [...lines, '```']

  return lines.join('\n')
}
