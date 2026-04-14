/**
 * Per-step feature extraction for the v5 MLP stuck detector.
 *
 * Stateless pure functions — all session state (outputHistory, stepCount) is
 * owned by the caller (SessionDetector). This makes each function independently
 * testable without needing a full session object.
 *
 * Feature order matches src/training/train.py STEP_FEATURES (with step_index_norm
 * dropped — it was a known train/inference mismatch and ablation showed no
 * statistically significant cost to removing it):
 *   [tool_idx, cmd_hash, file_hash, output_similarity, has_prior_output,
 *    output_length, is_error]
 */

import { crc32 } from 'node:zlib'

// CRC32 → [0, 1): use 2**32, NOT 1<<32
// JS bitwise operators truncate to 32-bit signed integers: 1<<32 === 1, not 4294967296.
const CRC32_NORM = 1 / 2 ** 32

// Tool name mapping: Claude Code names → abstract categories (matches parsers/nlile.py)
export const TOOL_MAP = {
  Bash: 'bash',
  bash: 'bash',
  Edit: 'edit',
  edit: 'edit',
  Write: 'edit',
  write: 'edit',
  MultiEdit: 'edit',
  Read: 'view',
  read: 'view',
  Grep: 'search',
  grep: 'search',
  Glob: 'search',
  glob: 'search',
  Agent: 'other',
  Task: 'other',
  TodoRead: 'other',
  TodoWrite: 'other',
}

export const TOOL_NAMES = ['bash', 'edit', 'view', 'search', 'create', 'submit', 'other']
export const TOOL_TO_IDX = Object.fromEntries(TOOL_NAMES.map((t, i) => [t, i]))

// Tools whose outputs are meaningless (edit success strings) — skip output processing
const EDIT_TOOLS = new Set(['edit', 'create', 'submit'])

const MAX_OUTPUT_LINES = 100
const SILENT_CMD_RE = /^(cd|pushd|popd|source|export|set|unset|alias|ulimit|umask)\b/
const FILE_EXT_RE = /\.[a-zA-Z]{1,5}$/
const SYSTEM_REMINDER_RE = /<system-reminder>[\s\S]*?<\/system-reminder>/gi
const ERROR_RE =
  /error|traceback|exception|failed|failure|fatal|cannot|unable to|not found|permission denied|segmentation fault|core dumped|FAIL|ModuleNotFoundError|ImportError|SyntaxError|TypeError|ValueError|KeyError|AttributeError|RuntimeError|FileNotFoundError/i

// Phase 2 helpers — must produce identical keys to extract_features.py's
// _PATH_TOKEN_RE / _TOKEN_SPLIT_RE / _coarse_program for train/inference parity.
const PATH_TOKEN_RE = /(?:\/?[\w@.\-]+\/)+[\w@.\-]+(?:\.[a-zA-Z0-9_]{1,8})?|[\w@.\-]+\.[a-zA-Z0-9_]{1,8}/g
const TOKEN_SPLIT_RE = /[A-Za-z_][\w./\-]+|\b\d+\b/g
const RECENT_TOKEN_HISTORY = 5
// log1p(50) ≈ 3.93; matches Python _FILE_REPEAT_NORM
const FILE_REPEAT_NORM = Math.log1p(50)

function extractPathTokens(text) {
  if (!text) return new Set()
  const out = new Set()
  const re = new RegExp(PATH_TOKEN_RE.source, 'g')
  let m
  while ((m = re.exec(text)) !== null) {
    const tok = m[0].trim()
    if (tok.length < 2) continue
    if (/^\d+$/.test(tok)) continue
    out.add(tok)
  }
  return out
}

function commandTokenSet(cmd) {
  if (!cmd) return new Set()
  const out = new Set()
  const re = new RegExp(TOKEN_SPLIT_RE.source, 'g')
  let m
  while ((m = re.exec(cmd)) !== null) {
    const tok = m[0]
    if (tok.startsWith('-')) continue
    if (tok.length < 2) continue
    out.add(tok.toLowerCase())
  }
  return out
}

function coarseProgram(cmd) {
  if (!cmd) return ''
  const parts = cmd.trim().split(/\s*(?:&&|;)\s*/)
  const real = parts.filter((p) => p.trim() && !SILENT_CMD_RE.test(p.trim()))
  if (real.length === 0) {
    const tokens = cmd.trim().split(/\s+/)
    if (tokens.length === 0 || !tokens[0]) return ''
    return tokens[0].split('/').pop()
  }
  const first = real[0].trim().split(/\s*\|\s*/)[0]
  const tokens = first.trim().split(/\s+/)
  if (tokens.length === 0 || !tokens[0]) return ''
  return tokens[0].split('/').pop()
}

/**
 * Parse a raw Claude Code tool call into a normalized step dict.
 *
 * @param {string} toolName  Claude Code tool name (e.g. "Bash", "Edit")
 * @param {object} input     Tool input object from the API message
 * @param {string} output    Tool output text
 * @returns {{ tool: string, cmd: string, file: string|null, output: string }}
 */
export function parseToolCall(toolName, input, output) {
  const tool = TOOL_MAP[toolName] ?? 'other'
  // command/file_path/pattern are used at full length. description/prompt are
  // fallbacks for Task/Agent tools, truncated to 200 chars. Uses || (falsy-check)
  // to mirror Python's "if not cmd: cmd = description[:200]" (nlile.py:56-59).
  const primaryCmd = input?.command || input?.file_path || input?.pattern || ''
  const cmd = primaryCmd
    ? String(primaryCmd)
    : String(input?.description ?? input?.prompt ?? '').slice(0, 200)
  const rawFile = input?.file_path ?? input?.path ?? null
  return {
    tool,
    cmd,
    file: rawFile !== null && rawFile !== undefined ? String(rawFile) : null,
    output: output ?? '',
  }
}

/**
 * Extract 'base_command:target_file' for semantic command matching.
 * Must produce the same key as Python's _cmd_semantic_key() in extract_features.py.
 *
 * @param {string} cmd  raw bash command string
 * @returns {string}    semantic key
 */
export function cmdSemanticKey(cmd) {
  if (!cmd) return ''
  const parts = cmd.trim().split(/\s*(?:&&|;)\s*/)
  const real = parts.filter((p) => p.trim() && !SILENT_CMD_RE.test(p.trim()))
  if (real.length === 0) {
    const t = cmd.trim().split(/\s+/)
    return t.length > 0 ? t[0] : ''
  }
  const first = real[0].trim().split(/\s*\|\s*/)[0]
  const tokens = first.trim().split(/\s+/)
  if (tokens.length === 0) return ''
  const si = tokens[0].lastIndexOf('/')
  const base = si >= 0 ? tokens[0].slice(si + 1) : tokens[0]
  let target = null
  for (let i = 1; i < tokens.length; i++) {
    if (tokens[i].startsWith('-')) continue
    if (FILE_EXT_RE.test(tokens[i]) || tokens[i].includes('/')) {
      const ti = tokens[i].lastIndexOf('/')
      target = ti >= 0 ? tokens[i].slice(ti + 1) : tokens[i]
      break
    }
  }
  return target ? `${base}:${target}` : base
}

/**
 * Normalize an output string to a set of canonical lines for Jaccard comparison.
 * Strips addresses, timestamps, PIDs, and temp paths so minor variations don't
 * prevent output-similarity detection.
 *
 * @param {string} output
 * @returns {Set<string>}
 */
export function normalizeToSet(output) {
  if (!output) return new Set()
  const lines = output.trim().split('\n').slice(0, MAX_OUTPUT_LINES)
  const result = new Set()
  for (let line of lines) {
    line = line.replace(/0x[0-9a-fA-F]+/g, '0xADDR')
    line = line.replace(/\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}/g, 'TIMESTAMP')
    line = line.replace(/pid[=: ]\d+/gi, 'pid=PID')
    line = line.replace(/\/tmp\/[^\s]+/g, '/tmp/TMPFILE')
    line = line.replace(/\d+\.\d{3,}s/g, 'N.NNNs')
    line = line.trim()
    if (line) result.add(line)
  }
  return result
}

/**
 * Maximum prior output slots kept per cmd_hash in outputHistory.
 * The previous implementation stored a single Set per key and overwrote it
 * on every call, so only the most recent predecessor was visible to the
 * Jaccard comparison. Multi-slot keeps the last K predecessors and takes
 * the MAX similarity against any of them — catches "agent re-reads the
 * same file across many turns" patterns that were previously invisible.
 *
 * Empirical effect on the 10-task benchmark: +0.012 pooled LR-AUC vs
 * Sonnet ground-truth labels, with no regression on any single task (see
 * benchmarks/FEATURE_FIX_NOTES.md). On 03_llvm_loop_vec specifically,
 * lifts output_similarity on 12 Sonnet-STUCK steps that previously read
 * zero.
 */
const OUTPUT_HISTORY_SLOTS = 5

/**
 * Jaccard similarity between two output sets.
 *
 * @param {Set} setA
 * @param {Set|null} setB  null means no prior output (returns 0)
 * @returns {number}  value in [0, 1]
 */
export function jaccard(setA, setB) {
  if (!setB) return 0.0
  if (setA.size === 0 && setB.size === 0) return 1.0
  let intersection = 0
  for (const v of setA) if (setB.has(v)) intersection++
  const union = setA.size + setB.size - intersection
  return union === 0 ? 1.0 : intersection / union
}

/**
 * Max Jaccard similarity of setA against any Set in the prior slots list.
 *
 * @param {Set} setA
 * @param {Set[]|undefined} priors  array of up to OUTPUT_HISTORY_SLOTS sets
 * @returns {number}  value in [0, 1]
 */
export function maxJaccard(setA, priors) {
  if (!priors || priors.length === 0) return 0.0
  let best = 0
  for (const p of priors) {
    const j = jaccard(setA, p)
    if (j > best) best = j
    if (best >= 1.0) break
  }
  return best
}

/**
 * Per-session feature extraction state. v5 sessions hold just outputHistory;
 * v6 (Phase 2) sessions hold three additional caches for the new features.
 *
 * Constructed by SessionDetector. Mutated in-place by computeFeatures.
 */
export class FeatureState {
  constructor() {
    this.outputHistory = new Map() // cmdHashInt → Set[] (length ≤ OUTPUT_HISTORY_SLOTS)
    this.fileTouchCount = new Map() // file path → count of prior steps touching it
    this.recentTokenSets = [] // last RECENT_TOKEN_HISTORY token sets
  }
}

/**
 * Compute the 10 v6 per-step features (or 7 if state is a plain Map for
 * backward compat).
 *
 * Side effects: appends current outputSet to the slot list, bumps file
 * touch counts, pushes current token set onto recentTokenSets ring.
 *
 * @param {{ tool: string, cmd: string, file: string|null, output: string }} step
 * @param {FeatureState|Map} state  per-session feature state
 * @returns {Float32Array}  length-7 (legacy) or length-10 (Phase 2) feature vector
 */
export function computeFeatures(step, state) {
  // Backward-compat: if state is a plain Map (old code path), treat it as
  // the outputHistory and emit 7-feature vectors. New code passes a
  // FeatureState instance and gets 10 features.
  const isPhase2 = state instanceof FeatureState
  const outputHistory = isPhase2 ? state.outputHistory : state

  const { tool, cmd, file, output } = step
  const toolIdx = TOOL_TO_IDX[tool] ?? TOOL_TO_IDX['other']

  // CRC32 hashes (unsigned 32-bit via >>> 0)
  const fileHashInt = file ? crc32(Buffer.from(file, 'utf8')) >>> 0 : null
  const cmdKey = tool === 'bash' && cmd ? cmdSemanticKey(cmd) : cmd ? `${tool}:${cmd}` : null
  const cmdHashInt = cmdKey ? crc32(Buffer.from(cmdKey, 'utf8')) >>> 0 : null

  const cleanOutput = stripSystemReminders(output)
  const isEditTool = EDIT_TOOLS.has(tool)

  const outputSet = isEditTool ? new Set() : normalizeToSet(cleanOutput)
  const priors = cmdHashInt !== null ? outputHistory.get(cmdHashInt) : undefined
  const hasPrior = !isEditTool && priors !== undefined && priors.length > 0
  const outputSim = isEditTool ? 0.0 : maxJaccard(outputSet, priors)

  // ── Phase 2 features ────────────────────────────────────────────────
  let fileRepeatNorm = 0
  let cmdHashCoarse = 0
  let recentTokenJacc = 0
  let curPaths = null
  let curTokens = null
  if (isPhase2) {
    curPaths = extractPathTokens(cmd)
    if (file) curPaths.add(file)
    if (curPaths.size > 0) {
      let repeatSum = 0
      for (const p of curPaths) repeatSum += state.fileTouchCount.get(p) ?? 0
      fileRepeatNorm = Math.min(1.0, Math.log1p(repeatSum) / FILE_REPEAT_NORM)
    }
    const coarseStr = tool === 'bash' ? coarseProgram(cmd) : (step.tool_name ?? tool)
    if (coarseStr) {
      cmdHashCoarse = (crc32(Buffer.from(coarseStr, 'utf8')) >>> 0) * CRC32_NORM
    }
    curTokens = commandTokenSet(cmd)
    if (curTokens.size > 0 && state.recentTokenSets.length > 0) {
      for (const prev of state.recentTokenSets) {
        if (prev.size === 0) continue
        let inter = 0
        for (const t of curTokens) if (prev.has(t)) inter++
        const union = curTokens.size + prev.size - inter
        if (union > 0) {
          const j = inter / union
          if (j > recentTokenJacc) recentTokenJacc = j
        }
      }
    }
  }

  const featLen = isPhase2 ? 10 : 7
  const features = new Float32Array(featLen)
  features[0] = toolIdx
  features[1] = cmdHashInt !== null ? cmdHashInt * CRC32_NORM : 0.0
  features[2] = fileHashInt !== null ? fileHashInt * CRC32_NORM : 0.0
  features[3] = outputSim
  features[4] = hasPrior ? 1.0 : 0.0
  features[5] = Math.log1p(cleanOutput ? cleanOutput.split('\n').length - 1 : 0)
  features[6] = cleanOutput && ERROR_RE.test(cleanOutput.slice(0, 2000)) ? 1.0 : 0.0
  if (isPhase2) {
    features[7] = fileRepeatNorm
    features[8] = cmdHashCoarse
    features[9] = recentTokenJacc
  }

  // ── State updates (after recording features) ────────────────────────
  if (cmdHashInt !== null && !isEditTool) {
    const slots = outputHistory.get(cmdHashInt)
    if (slots === undefined) {
      outputHistory.set(cmdHashInt, [outputSet])
    } else {
      slots.push(outputSet)
      if (slots.length > OUTPUT_HISTORY_SLOTS) slots.shift()
    }
  }
  if (isPhase2) {
    if (curPaths) {
      for (const p of curPaths) {
        state.fileTouchCount.set(p, (state.fileTouchCount.get(p) ?? 0) + 1)
      }
    }
    if (curTokens) {
      state.recentTokenSets.push(curTokens)
      if (state.recentTokenSets.length > RECENT_TOKEN_HISTORY) state.recentTokenSets.shift()
    }
  }

  return features
}

function stripSystemReminders(text) {
  if (!text || !text.includes('<system-reminder')) return text
  return text.replace(SYSTEM_REMINDER_RE, '')
}
