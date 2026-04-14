/**
 * Per-session stuck detector.
 *
 * Composes feature extraction, ring buffer history, and MLP inference into a
 * stateful object — one instance per active Claude Code session.
 *
 * Call addStep() once per tool call in order. The ring buffer holds the last
 * N=5 per-step feature vectors; previous scores are NOT fed back (ablation
 * showed identical F1 with or without them, and dropping them eliminates the
 * train/inference distribution mismatch).
 */

import { parseToolCall, computeFeatures, FeatureState } from './features.mjs'
import { RingBuffer } from './ring_buffer.mjs'

export class SessionDetector {
  /**
   * @param {{ forward: (input: Float32Array) => number, inputDim?: number }} mlp
   *   loaded MLP instance. If mlp.inputDim is set, the ring buffer is sized
   *   accordingly so v5 (42-dim) and v6 (60-dim) checkpoints can both be
   *   served. Defaults to the v5 7-feature ring.
   */
  constructor(mlp) {
    this._mlp = mlp
    // FeatureState carries everything per-session: outputHistory (multi-slot
    // since schema 4), fileTouchCount + recentTokenSets (since schema 5).
    // Backward-compat: if features.mjs is given a FeatureState, it returns
    // 10-dim; if it gets a plain Map it returns 7-dim. We always use
    // FeatureState here so v6 inference works; v5 weights still load
    // because the ring buffer just reads the first 7 dims.
    this._featureState = new FeatureState()
    // RingBuffer signature: (n_history, featureDim). We use the default
    // n=5 history depth and let featureDim follow the loaded MLP — 7 for
    // v5 weights, 10 for v6/v6_pwN weights.
    this._ring = new RingBuffer(undefined, mlp.featureDim ?? 7)
    this._stepCount = 0
  }

  /**
   * Process one tool call and return the stuck score.
   *
   * @param {string} toolName  Claude Code tool name (e.g. "Bash", "Edit")
   * @param {object} input     Tool input object from the API message
   * @param {string} output    Tool output text
   * @returns {number}  MLP sigmoid score in [0, 1]
   */
  addStep(toolName, input, output) {
    const step = parseToolCall(toolName, input, output)
    // computeFeatures detects FeatureState and returns 10-dim. The ring
    // buffer can still consume those features regardless of MLP width.
    const features = computeFeatures(step, this._featureState)
    const inputVec = this._ring.buildInput(features)
    const score = this._mlp.forward(inputVec)
    this._ring.push(features)
    this._stepCount++
    return score
  }
}
