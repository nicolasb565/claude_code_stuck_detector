/**
 * Per-step MLP inference. Supports both v5 (42-dim, 7 features × 6 slots) and
 * v6 (60-dim, 10 features × 6 slots) checkpoints — the dimension is inferred
 * from the loaded fc1.weight shape so the same code serves both.
 *
 * Architecture: Linear(input_dim,64) → ReLU → Linear(64,32) → ReLU → Linear(32,1) → Sigmoid
 * Normalization: mean/std applied to every dim (no score-history dimension).
 */

import { readFileSync } from 'node:fs'

/**
 * Load an MLP instance from a JSON weights file.
 *
 * @param {string} weightsPath  path to stuck_weights.json
 * @returns {MLP}
 */
export function loadMLP(weightsPath) {
  const weights = JSON.parse(readFileSync(weightsPath, 'utf8'))
  return new MLP(weights)
}

export class MLP {
  /** @param {object} weights  parsed stuck_weights.json */
  constructor(weights) {
    this._fc1w = weights['fc1.weight'] // 64 × input_dim
    this._fc1b = weights['fc1.bias'] // 64
    this._fc2w = weights['fc2.weight'] // 32 × 64
    this._fc2b = weights['fc2.bias'] // 32
    this._fc3w = weights['fc3.weight'] // 1 × 32
    this._fc3b = weights['fc3.bias'] // 1
    this._mean = new Float32Array(weights['norm_mean'])
    this._std = new Float32Array(weights['norm_std'])
    // Infer input_dim from the fc1 weight shape: rows are output, cols are input
    this.inputDim = this._fc1w[0].length
    // featureDim is inputDim / (1 + N_HISTORY); N_HISTORY=5 in the production
    // pipeline, so 42→7 (v5), 60→10 (v6).
    this.featureDim = Math.round(this.inputDim / 6)
  }

  /**
   * Run the forward pass on a raw (un-normalized) input vector.
   *
   * @param {Float32Array} input  length-inputDim input from RingBuffer.buildInput()
   * @returns {number}  sigmoid score in [0, 1]
   */
  forward(input) {
    const x = new Float32Array(this.inputDim)
    for (let i = 0; i < this.inputDim; i++) {
      x[i] = (input[i] - this._mean[i]) / (this._std[i] || 1e-6)
    }

    const h1 = _matVec(this._fc1w, x, this._fc1b)
    _relu(h1)
    const h2 = _matVec(this._fc2w, h1, this._fc2b)
    _relu(h2)
    const h3 = _matVec(this._fc3w, h2, this._fc3b)
    return _sigmoid(h3[0])
  }
}

/** Dense matrix-vector multiply: out[i] = bias[i] + sum_j(weight[i][j] * input[j]) */
function _matVec(weight, input, bias) {
  const out = new Float32Array(weight.length)
  for (let i = 0; i < weight.length; i++) {
    let s = bias[i]
    const row = weight[i]
    for (let j = 0; j < input.length; j++) s += row[j] * input[j]
    out[i] = s
  }
  return out
}

function _relu(arr) {
  for (let i = 0; i < arr.length; i++) if (arr[i] < 0) arr[i] = 0
}

function _sigmoid(x) {
  return 1 / (1 + Math.exp(-x))
}
