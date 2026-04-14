import { describe, test } from 'node:test'
import assert from 'node:assert/strict'
import {
  parseToolCall,
  computeFeatures,
  cmdSemanticKey,
  jaccard,
  normalizeToSet,
  TOOL_TO_IDX,
  FeatureState,
} from '../features.mjs'

describe('parseToolCall', () => {
  test('maps Bash → bash, extracts cmd and output', () => {
    const s = parseToolCall('Bash', { command: 'ls -la' }, 'total 8')
    assert.equal(s.tool, 'bash')
    assert.equal(s.cmd, 'ls -la')
    assert.equal(s.file, null)
    assert.equal(s.output, 'total 8')
  })

  test('maps Edit → edit, extracts file', () => {
    const s = parseToolCall('Edit', { file_path: 'src/foo.py' }, 'OK')
    assert.equal(s.tool, 'edit')
    assert.equal(s.file, 'src/foo.py')
  })

  test('maps Write → edit', () => {
    assert.equal(parseToolCall('Write', {}, '').tool, 'edit')
  })

  test('maps MultiEdit → edit', () => {
    assert.equal(parseToolCall('MultiEdit', {}, '').tool, 'edit')
  })

  test('maps Read → view', () => {
    assert.equal(parseToolCall('Read', { file_path: 'README.md' }, '').tool, 'view')
  })

  test('maps Grep → search', () => {
    assert.equal(parseToolCall('Grep', { pattern: 'foo' }, '').tool, 'search')
  })

  test('maps Glob → search', () => {
    assert.equal(parseToolCall('Glob', { pattern: '**/*.py' }, '').tool, 'search')
  })

  test('uses description fallback for Task/Agent tools', () => {
    const s = parseToolCall('Task', { description: 'run the build' }, '')
    assert.equal(s.cmd, 'run the build')
  })

  test('uses prompt fallback when description absent', () => {
    const s = parseToolCall('Agent', { prompt: 'analyze the codebase' }, '')
    assert.equal(s.cmd, 'analyze the codebase')
  })

  test('truncates description/prompt fallback to 200 chars', () => {
    const long = 'x'.repeat(300)
    const s = parseToolCall('Task', { description: long }, '')
    assert.equal(s.cmd.length, 200)
  })

  test('does not truncate regular bash command', () => {
    const long = 'a'.repeat(300)
    const s = parseToolCall('Bash', { command: long }, '')
    assert.equal(s.cmd.length, 300)
  })

  test('does not truncate long file path', () => {
    const long = '/very/long/' + 'a'.repeat(300) + '.py'
    const s = parseToolCall('Read', { file_path: long }, '')
    assert.equal(s.cmd, long)
  })

  test('falls back to description when command is empty string', () => {
    const s = parseToolCall('Task', { command: '', description: 'fallback text' }, '')
    assert.equal(s.cmd, 'fallback text')
  })

  test('falls back to description when command is null', () => {
    const s = parseToolCall('Task', { command: null, description: 'fallback text' }, '')
    assert.equal(s.cmd, 'fallback text')
  })

  test('unknown tool maps to other', () => {
    assert.equal(parseToolCall('UnknownTool', {}, '').tool, 'other')
  })

  test('null input does not throw', () => {
    const s = parseToolCall('Bash', null, 'out')
    assert.equal(s.cmd, '')
    assert.equal(s.file, null)
    assert.equal(s.output, 'out')
  })

  test('undefined output defaults to empty string', () => {
    const s = parseToolCall('Bash', { command: 'ls' }, undefined)
    assert.equal(s.output, '')
  })
})

describe('cmdSemanticKey', () => {
  test('extracts bare command', () => {
    assert.equal(cmdSemanticKey('ls -la'), 'ls')
  })

  test('extracts base:target when file argument present', () => {
    assert.equal(cmdSemanticKey('gcc -O2 test.c'), 'gcc:test.c')
  })

  test('strips leading path from binary', () => {
    assert.equal(cmdSemanticKey('/usr/bin/gcc test.c'), 'gcc:test.c')
  })

  test('skips silent commands, uses first real command', () => {
    assert.equal(cmdSemanticKey('cd /tmp && ls'), 'ls')
  })

  test('handles pipe — uses first segment', () => {
    assert.equal(cmdSemanticKey('ls -la | grep foo'), 'ls')
  })

  test('empty string returns empty string', () => {
    assert.equal(cmdSemanticKey(''), '')
  })
})

describe('computeFeatures', () => {
  test('returns Float32Array of length 7', () => {
    const feats = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: '' }, new Map())
    assert.ok(feats instanceof Float32Array)
    assert.equal(feats.length, 7)
  })

  test('tool_idx [0] matches TOOL_TO_IDX for each tool', () => {
    for (const [tool, idx] of Object.entries(TOOL_TO_IDX)) {
      const feats = computeFeatures({ tool, cmd: '', file: null, output: '' }, new Map())
      assert.equal(feats[0], idx, `tool=${tool}`)
    }
  })

  test('output_length [5] is 0 for empty output', () => {
    const feats = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: '' }, new Map())
    assert.equal(feats[5], 0)
  })

  test('output_length [5] is log1p(newline_count)', () => {
    const output = 'a\nb\nc' // 2 newlines
    const feats = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output }, new Map())
    assert.ok(Math.abs(feats[5] - Math.log1p(2)) < 1e-6)
  })

  test('is_error [6] is 1 for error output', () => {
    const feats = computeFeatures(
      { tool: 'bash', cmd: 'ls', file: null, output: 'Error: file not found' },
      new Map(),
    )
    assert.equal(feats[6], 1.0)
  })

  test('is_error [6] is 0 for clean output', () => {
    const feats = computeFeatures(
      { tool: 'bash', cmd: 'ls', file: null, output: 'hello world' },
      new Map(),
    )
    assert.equal(feats[6], 0.0)
  })

  test('has_prior_output [4] is 0 on first call', () => {
    const feats = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'hi' }, new Map())
    assert.equal(feats[4], 0.0)
  })

  test('has_prior_output [4] is 1 after same command run again', () => {
    const history = new Map()
    computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'hi' }, history)
    const feats2 = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'hi' }, history)
    assert.equal(feats2[4], 1.0)
  })

  test('output_similarity [3] is 1.0 for identical output repeated', () => {
    const history = new Map()
    computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'foo\nbar' }, history)
    const feats2 = computeFeatures(
      { tool: 'bash', cmd: 'ls', file: null, output: 'foo\nbar' },
      history,
    )
    assert.equal(feats2[3], 1.0)
  })

  test('output_similarity [3] is 0.0 for completely different output', () => {
    const history = new Map()
    computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'foo' }, history)
    const feats2 = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'bar' }, history)
    assert.equal(feats2[3], 0.0)
  })

  test('multi-slot: matches an older predecessor, not just the most recent', () => {
    // Pattern: call A (output X), call A (output Y), call A (output X again).
    // Single-slot would compare the 3rd call to Y (low similarity). Multi-slot
    // should find the match against the earlier X and report 1.0.
    const history = new Map()
    computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'alpha\nbeta' }, history)
    computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'gamma\ndelta' }, history)
    const feats3 = computeFeatures(
      { tool: 'bash', cmd: 'ls', file: null, output: 'alpha\nbeta' },
      history,
    )
    assert.equal(feats3[3], 1.0)
    assert.equal(feats3[4], 1.0)
  })

  test('multi-slot: FIFO eviction after N+1 entries', () => {
    // Push 6 distinct outputs (K=5 slots), the oldest should be evicted.
    // The 7th call with content matching the oldest should then score 0.
    const history = new Map()
    const outputs = ['a', 'b', 'c', 'd', 'e', 'f']
    for (const o of outputs) {
      computeFeatures({ tool: 'bash', cmd: 'x', file: null, output: o }, history)
    }
    // Slots now hold b, c, d, e, f (a was evicted). Call with 'a' should
    // score zero because the 'a' slot is gone.
    const feats = computeFeatures({ tool: 'bash', cmd: 'x', file: null, output: 'a' }, history)
    assert.equal(feats[3], 0.0)
    // Call with 'f' (still in slots) should match.
    const feats2 = computeFeatures({ tool: 'bash', cmd: 'x', file: null, output: 'f' }, history)
    assert.equal(feats2[3], 1.0)
  })

  test('edit tool has output_similarity=0 and has_prior=0 even after repeat', () => {
    const history = new Map()
    computeFeatures({ tool: 'edit', cmd: 'foo.py', file: 'foo.py', output: 'ok' }, history)
    const feats2 = computeFeatures(
      { tool: 'edit', cmd: 'foo.py', file: 'foo.py', output: 'ok' },
      history,
    )
    assert.equal(feats2[3], 0.0) // output_similarity
    assert.equal(feats2[4], 0.0) // has_prior_output
  })

  test('cmd_hash [1] is in [0, 1)', () => {
    const feats = computeFeatures(
      { tool: 'bash', cmd: 'ls -la', file: null, output: '' },
      new Map(),
    )
    assert.ok(feats[1] >= 0 && feats[1] < 1)
  })

  test('file_hash [2] is in [0, 1)', () => {
    const feats = computeFeatures(
      { tool: 'view', cmd: '', file: 'src/main.py', output: '' },
      new Map(),
    )
    assert.ok(feats[2] >= 0 && feats[2] < 1)
  })

  test('file_hash [2] is 0 for null file', () => {
    const feats = computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: '' }, new Map())
    assert.equal(feats[2], 0.0)
  })

  test('same command produces same cmd_hash across calls', () => {
    const feats1 = computeFeatures(
      { tool: 'bash', cmd: 'make test', file: null, output: '' },
      new Map(),
    )
    const feats2 = computeFeatures(
      { tool: 'bash', cmd: 'make test', file: null, output: '' },
      new Map(),
    )
    assert.equal(feats1[1], feats2[1])
  })
})

describe('Phase 2 features (FeatureState, 10-dim)', () => {
  test('plain Map gives 7-dim vector (backward compat)', () => {
    const feats = computeFeatures(
      { tool: 'bash', cmd: 'ls', file: null, output: 'a' },
      new Map(),
    )
    assert.equal(feats.length, 7)
  })

  test('FeatureState gives 10-dim vector', () => {
    const feats = computeFeatures(
      { tool: 'bash', cmd: 'ls', file: null, output: 'a' },
      new FeatureState(),
    )
    assert.equal(feats.length, 10)
  })

  test('file_repeat_count_norm grows with repeated file touches', () => {
    const state = new FeatureState()
    const cmd = 'cat /scratch/foo.cpp'
    const f0 = computeFeatures({ tool: 'bash', cmd, file: null, output: 'x' }, state)
    const f1 = computeFeatures({ tool: 'bash', cmd, file: null, output: 'y' }, state)
    const f2 = computeFeatures({ tool: 'bash', cmd, file: null, output: 'z' }, state)
    assert.equal(f0[7], 0.0) // first touch
    assert.ok(f1[7] > 0.0)
    assert.ok(f2[7] > f1[7])
    assert.ok(f2[7] <= 1.0)
  })

  test('file_repeat_count_norm zero when files do not overlap', () => {
    const state = new FeatureState()
    computeFeatures({ tool: 'bash', cmd: 'cat /a/foo.cpp', file: null, output: '' }, state)
    const f = computeFeatures(
      { tool: 'bash', cmd: 'cat /b/bar.cpp', file: null, output: '' },
      state,
    )
    assert.equal(f[7], 0.0)
  })

  test('file_repeat_count_norm picks up native tool file field', () => {
    const state = new FeatureState()
    computeFeatures(
      { tool: 'view', cmd: '/scratch/foo.cpp', file: '/scratch/foo.cpp', output: '' },
      state,
    )
    const f = computeFeatures(
      { tool: 'bash', cmd: 'grep bar /scratch/foo.cpp', file: null, output: '' },
      state,
    )
    assert.ok(f[7] > 0)
  })

  test('cmd_hash_coarse collapses different git subcommands', () => {
    const s1 = new FeatureState()
    const s2 = new FeatureState()
    const f1 = computeFeatures({ tool: 'bash', cmd: 'git log --oneline', file: null, output: '' }, s1)
    const f2 = computeFeatures({ tool: 'bash', cmd: 'git diff HEAD', file: null, output: '' }, s2)
    assert.equal(f1[8], f2[8])  // both → just "git"
  })

  test('cmd_hash_coarse uses tool name for non-bash', () => {
    const s = new FeatureState()
    const f1 = computeFeatures(
      { tool: 'view', tool_name: 'Read', cmd: '/a/b.txt', file: '/a/b.txt', output: '' },
      s,
    )
    const f2 = computeFeatures(
      { tool: 'view', tool_name: 'Read', cmd: '/c/d.txt', file: '/c/d.txt', output: '' },
      s,
    )
    assert.equal(f1[8], f2[8])  // both Read tools → same coarse
  })

  test('recent_token_jaccard high for similar greps', () => {
    const state = new FeatureState()
    computeFeatures(
      { tool: 'bash', cmd: 'grep -rn getSCEVExprForVPValue /scratch/llvm/lib', file: null, output: '' },
      state,
    )
    const f = computeFeatures(
      { tool: 'bash', cmd: 'grep -B5 getSCEVExprForVPValue /scratch/llvm/lib/foo.cpp', file: null, output: '' },
      state,
    )
    assert.ok(f[9] > 0.3)
  })

  test('recent_token_jaccard low for unrelated commands', () => {
    const state = new FeatureState()
    computeFeatures({ tool: 'bash', cmd: 'git log', file: null, output: '' }, state)
    const f = computeFeatures(
      { tool: 'bash', cmd: 'ninja -C build opt', file: null, output: '' },
      state,
    )
    assert.ok(f[9] < 0.2)
  })

  test('recent_token_jaccard zero on first command', () => {
    const f = computeFeatures(
      { tool: 'bash', cmd: 'grep foo bar.txt', file: null, output: '' },
      new FeatureState(),
    )
    assert.equal(f[9], 0.0)
  })

  test('FeatureState multi-slot output history works the same as Map', () => {
    // Same multi-slot test as the legacy path but with FeatureState
    const state = new FeatureState()
    computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'alpha\nbeta' }, state)
    computeFeatures({ tool: 'bash', cmd: 'ls', file: null, output: 'gamma\ndelta' }, state)
    const f3 = computeFeatures(
      { tool: 'bash', cmd: 'ls', file: null, output: 'alpha\nbeta' },
      state,
    )
    assert.equal(f3[3], 1.0) // output_similarity matches first ls (older slot)
  })
})

describe('jaccard', () => {
  test('returns 0 for null prior (no prior output)', () => {
    assert.equal(jaccard(new Set(['a']), null), 0.0)
  })

  test('returns 1 for two empty sets', () => {
    assert.equal(jaccard(new Set(), new Set()), 1.0)
  })

  test('returns 1 for identical sets', () => {
    assert.equal(jaccard(new Set(['a', 'b']), new Set(['a', 'b'])), 1.0)
  })

  test('returns 0 for disjoint sets', () => {
    assert.equal(jaccard(new Set(['a']), new Set(['b'])), 0.0)
  })

  test('returns 1/3 for one-of-three overlap', () => {
    // {a,b} ∩ {b,c} = {b}, union = {a,b,c} → 1/3
    assert.ok(Math.abs(jaccard(new Set(['a', 'b']), new Set(['b', 'c'])) - 1 / 3) < 1e-9)
  })

  test('returns 0.5 when one set is a subset of size half', () => {
    // {a,b} ∩ {a} = {a}, union = {a,b} → 1/2
    assert.equal(jaccard(new Set(['a', 'b']), new Set(['a'])), 0.5)
  })
})

describe('normalizeToSet', () => {
  test('returns empty set for empty input', () => {
    assert.equal(normalizeToSet('').size, 0)
  })

  test('normalizes hex addresses', () => {
    const s = normalizeToSet('addr=0xdeadbeef')
    assert.ok(s.has('addr=0xADDR'))
  })

  test('normalizes timestamps', () => {
    const s = normalizeToSet('2024-01-15 12:34:56 event')
    assert.ok(s.has('TIMESTAMP event'))
  })

  test('deduplicates identical lines', () => {
    const s = normalizeToSet('foo\nfoo\nfoo')
    assert.equal(s.size, 1)
  })
})
