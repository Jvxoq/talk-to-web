const FENCE = /^\s*(```|~~~)/
const DELIMITER_ROW = /^\s*\|?[\s:|-]+\|[\s:|-]*$/
const SETEXT_RULE = /^\s*(-{1,}|={1,}|\*{3,}|_{3,})\s*$/

/** Lines like `| a | b |` — the shape both table rows and their headers share. */
function isTableLine(line: string | undefined): boolean {
  return line !== undefined && line.trimStart().startsWith('|')
}

/**
 * Hides a table until its `|---|` delimiter row is complete, to avoid raw pipes.
 */
function dropIncompleteTable(lines: string[]): string[] {
  let start = lines.length
  while (start > 0 && isTableLine(lines[start - 1])) start--
  if (start === lines.length) return lines

  // The delimiter is finished only once another line follows it.
  const settled = DELIMITER_ROW.test(lines[start + 1] ?? '') && lines.length > start + 2
  return settled ? lines : lines.slice(0, start)
}

/**
 * Makes half-streamed markdown safe to render.
 *
 * Handles an unclosed code fence, a trailing `---`, and an unfinished table.
 */
export function normalizeStreamingMarkdown(text: string): string {
  let lines = text.split('\n')

  // A trailing `---` would turn the line above into a heading.
  if (SETEXT_RULE.test(lines[lines.length - 1] ?? '')) lines = lines.slice(0, -1)

  lines = dropIncompleteTable(lines)

  const fences = lines.filter((line) => FENCE.test(line)).length
  if (fences % 2 === 1) lines = [...lines, '```']

  return lines.join('\n')
}
