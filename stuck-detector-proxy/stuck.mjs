/**
 * Stuck detection: analyze thinking blocks for circular reasoning.
 * Uses a trained classifier (TF-IDF features + LogReg) via Python subprocess,
 * with heuristic fallback if classifier unavailable.
 * When detected, inject a corrective nudge into the messages.
 */

import { execSync } from "child_process";
import { dirname, join } from "path";
import { fileURLToPath } from "url";

const __dirname = dirname(fileURLToPath(import.meta.url));
const CLASSIFY_SCRIPT = join(__dirname, "classify.py");
const PYTHON = process.env.STUCK_PYTHON || "/home/nicolas/source/classifier-repos/venv/bin/python3";

let lastNudgeTurn = -999;
let turnCounter = 0;
let classifierAvailable = null; // null = untested

export function resetState() {
  lastNudgeTurn = -999;
  turnCounter = 0;
}

/**
 * Call the trained classifier via Python subprocess.
 * Returns {score, label, reasons} or null if unavailable.
 */
function callClassifier(text) {
  if (classifierAvailable === false) return null;

  try {
    const result = execSync(`${PYTHON} "${CLASSIFY_SCRIPT}"`, {
      input: text,
      timeout: 5000,
      encoding: "utf-8",
      stdio: ["pipe", "pipe", "pipe"],
    });
    const parsed = JSON.parse(result.trim());
    if (classifierAvailable === null) {
      classifierAvailable = true;
      console.log("[stuck-detector] Classifier loaded successfully");
    }
    return parsed;
  } catch (e) {
    if (classifierAvailable === null) {
      classifierAvailable = false;
      console.log(`[stuck-detector] Classifier unavailable, using heuristic fallback: ${e.message}`);
    }
    return null;
  }
}

/**
 * Heuristic fallback when classifier is unavailable.
 */
function heuristicCheck(text) {
  let score = 0;
  const reasons = [];

  // Repeated substrings
  const seen = {};
  for (let i = 0; i < text.length - 20; i += 10) {
    const sub = text.substring(i, i + 20);
    seen[sub] = (seen[sub] || 0) + 1;
    if (seen[sub] >= 3) {
      score += 0.4;
      reasons.push("repeated_substring");
      break;
    }
  }

  // Circle keywords
  const matches = text.match(
    /\b(try again|let me try|another approach|actually,|wait,|hmm|let me reconsider|that didn't work|same error|still failing)\b/gi,
  );
  if (matches && matches.length >= 5) {
    score += 0.3;
    reasons.push(`circle_kw(${matches.length})`);
  }

  // Self-similarity
  if (text.length > 2000) {
    const half = Math.floor(text.length / 2);
    const words1 = new Set(
      text.slice(0, half).toLowerCase().split(/\s+/).filter((w) => w.length > 4),
    );
    const words2 = text.slice(half).toLowerCase().split(/\s+/).filter((w) => w.length > 4);
    let overlap = 0;
    for (const w of words2) if (words1.has(w)) overlap++;
    if (words2.length > 0 && overlap / words2.length > 0.6) {
      score += 0.3;
      reasons.push("high_overlap");
    }
  }

  return {
    score,
    label: score >= 0.5 ? "stuck" : "productive",
    reasons,
  };
}

export function pruneIfStuck(messages, log) {
  turnCounter++;
  const cooldown = parseInt(process.env.STUCK_COOLDOWN || "5", 10);
  if (turnCounter - lastNudgeTurn < cooldown) return messages;

  // Extract thinking from the last assistant message
  let lastAssistantIdx = -1;
  for (let i = messages.length - 1; i >= 0; i--) {
    if (messages[i].role === "assistant") {
      lastAssistantIdx = i;
      break;
    }
  }
  if (lastAssistantIdx === -1) return messages;

  const lastAssistant = messages[lastAssistantIdx];
  if (!Array.isArray(lastAssistant.content)) return messages;

  let thinking = "";
  for (const block of lastAssistant.content) {
    if (block.type === "thinking" && block.thinking) {
      thinking += block.thinking;
    }
  }

  if (thinking.length < 500) return messages;

  // Extract thinking from previous assistant messages for cross-window comparison
  const prevThinkings = [];
  for (let i = lastAssistantIdx - 1; i >= 0; i--) {
    if (messages[i].role !== "assistant" || !Array.isArray(messages[i].content)) continue;
    let prevThinking = "";
    for (const block of messages[i].content) {
      if (block.type === "thinking" && block.thinking) prevThinking += block.thinking;
    }
    if (prevThinking.length > 200) {
      prevThinkings.push(prevThinking);
      if (prevThinkings.length >= 3) break; // last 3 thinking blocks
    }
  }

  // Cross-window similarity: compare current thinking to recent previous ones
  let crossWindowScore = 0;
  if (prevThinkings.length > 0) {
    const currWords = new Set(thinking.toLowerCase().split(/\s+/).filter(w => w.length > 4));
    for (const prev of prevThinkings) {
      const prevWords = prev.toLowerCase().split(/\s+/).filter(w => w.length > 4);
      if (prevWords.length === 0) continue;
      let overlap = 0;
      for (const w of prevWords) if (currWords.has(w)) overlap++;
      const sim = overlap / prevWords.length;
      crossWindowScore = Math.max(crossWindowScore, sim);
    }
  }

  // Try classifier first, fall back to heuristic
  const threshold = parseFloat(process.env.STUCK_THRESHOLD || "0.85");
  let result = callClassifier(thinking);
  if (!result) {
    result = heuristicCheck(thinking);
  }

  // Combine classifier score with cross-window similarity.
  // High classifier score + low cross-window = wrong hypothesis but new ground (don't nudge)
  // High classifier score + high cross-window = genuinely stuck (nudge)
  const crossWindowThreshold = parseFloat(process.env.STUCK_CROSS_WINDOW_THRESHOLD || "0.5");
  const combinedStuck = result.score >= threshold &&
    (prevThinkings.length === 0 || crossWindowScore >= crossWindowThreshold);

  if (!combinedStuck) {
    if (result.score >= threshold) {
      log?.("stuck_suppressed", {
        turnCount: turnCounter,
        classifierScore: result.score,
        crossWindowScore: Math.round(crossWindowScore * 100) / 100,
        reason: "low cross-window similarity — likely wrong hypothesis but progressing",
      });
    }
    return messages;
  }

  log?.("stuck_detected", {
    turnCount: turnCounter,
    thinkingLength: thinking.length,
    score: result.score,
    crossWindowScore: Math.round(crossWindowScore * 100) / 100,
    label: result.label,
    reasons: result.reasons,
    method: classifierAvailable ? "classifier" : "heuristic",
  });

  lastNudgeTurn = turnCounter;

  // Build recent tool call summary
  const recentTools = [];
  for (const msg of messages.slice(-20)) {
    if (!Array.isArray(msg.content)) continue;
    for (const block of msg.content) {
      if (block.type === "tool_use") {
        const detail =
          block.input?.command || block.input?.file_path || block.input?.pattern || "";
        recentTools.push(`${block.name}: ${String(detail).slice(-60)}`);
      }
    }
  }

  const nudge = {
    role: "user",
    content: [
      {
        type: "text",
        text:
          `[CONTEXT MONITOR — turn ${turnCounter}, confidence ${(result.score * 100).toFixed(0)}%]\n\n` +
          `Your recent thinking shows signs of repeated reasoning patterns. ` +
          `You may be going in circles.\n\n` +
          `Recent tool calls:\n  ${recentTools.slice(-8).join("\n  ")}\n\n` +
          `Review your last few turns critically:\n` +
          `- Are you retrying the same approach with minor variations?\n` +
          `- Are you investigating the same files/functions repeatedly?\n` +
          `- Has your hypothesis changed or are you stuck on the same one?\n\n` +
          `If you are going in circles, try a fundamentally different strategy.\n` +
          `State what you have learned so far and what new approach you will try.`,
      },
    ],
  };

  log?.("stuck_nudge_injected", {
    turnCount: turnCounter,
    score: result.score,
    method: classifierAvailable ? "classifier" : "heuristic",
    recentTools: recentTools.slice(-5),
  });

  return [...messages, nudge];
}
