/**
 * CNN-based stuck detection for Claude Code.
 *
 * Replaces the LogReg text classifier with a CNN that operates on
 * tool-call behavioral features (CRC32 hashed commands, Jaccard output
 * similarity, cycle detection). Language-agnostic — works across
 * programming languages and task types.
 *
 * Uses sliding window history: fires only if 2 of last 3 windows
 * score above threshold, reducing isolated false positives.
 */

import { classifyWindow, normalizeFeatures, config } from "./classify_cnn.mjs";
import { StuckDetectorState } from "./abstract_step.mjs";

const WINDOW_SIZE = config.window_size;
const HISTORY_SIZE = 3;
const FIRE_THRESHOLD = 2; // fire if 2 out of 3 windows score above threshold

// Per-session state
const sessions = new Map();

function getSession(messages) {
  let key = "";
  for (const msg of messages) {
    if (msg.role === "user") {
      const text = Array.isArray(msg.content)
        ? msg.content.map(b => b.text || "").join("")
        : String(msg.content);
      key = text.slice(0, 200);
      break;
    }
  }
  if (!key) key = "__default__";

  if (!sessions.has(key)) {
    sessions.set(key, {
      detector: new StuckDetectorState(),
      windowScores: [],
      turnCounter: 0,
      lastNudgeTurn: -999,
      initialized: false,
    });
  }
  return sessions.get(key);
}

export function resetState() {
  sessions.clear();
}

/**
 * Extract tool calls from messages that haven't been processed yet.
 * Returns array of { toolName, input, output, thinking } objects.
 */
function extractNewToolCalls(messages, session) {
  const toolCalls = [];

  // Find all tool_use blocks in assistant messages
  const pendingResults = new Map();

  for (const msg of messages) {
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      let thinking = "";
      for (const block of msg.content) {
        if (block.type === "thinking") {
          thinking = block.thinking || "";
        } else if (block.type === "tool_use") {
          pendingResults.set(block.id, {
            toolName: block.name,
            input: block.input || {},
            output: "",
            thinking,
          });
          thinking = ""; // only first tool call gets thinking
        }
      }
    } else if (msg.role === "user" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_result" && pendingResults.has(block.tool_use_id)) {
          const tc = pendingResults.get(block.tool_use_id);
          const content = block.content;
          if (Array.isArray(content)) {
            tc.output = content
              .filter(b => b.type === "text")
              .map(b => b.text || "")
              .join(" ");
          } else if (typeof content === "string") {
            tc.output = content;
          }
        }
      }
    }
  }

  // Return all tool calls in order
  for (const msg of messages) {
    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_use" && pendingResults.has(block.id)) {
          toolCalls.push(pendingResults.get(block.id));
        }
      }
    }
  }

  return toolCalls;
}

export function pruneIfStuck(messages, log) {
  const session = getSession(messages);
  session.turnCounter++;

  const cooldown = parseInt(process.env.STUCK_COOLDOWN || "5", 10);
  if (session.turnCounter - session.lastNudgeTurn < cooldown) return messages;

  // On first call, process all existing tool calls to build history
  if (!session.initialized) {
    const allToolCalls = extractNewToolCalls(messages, session);
    for (const tc of allToolCalls) {
      session.detector.addStep(tc.toolName, tc.input, tc.output, tc.thinking);
    }
    session.initialized = true;
  } else {
    // Process only the last assistant message's tool calls
    const lastAssistant = [...messages].reverse().find(m => m.role === "assistant");
    if (lastAssistant && Array.isArray(lastAssistant.content)) {
      let thinking = "";
      for (const block of lastAssistant.content) {
        if (block.type === "thinking") {
          thinking = block.thinking || "";
        } else if (block.type === "tool_use") {
          // Find the corresponding tool result
          let output = "";
          for (const msg of messages) {
            if (msg.role === "user" && Array.isArray(msg.content)) {
              for (const b of msg.content) {
                if (b.type === "tool_result" && b.tool_use_id === block.id) {
                  output = Array.isArray(b.content)
                    ? b.content.filter(x => x.type === "text").map(x => x.text).join(" ")
                    : String(b.content || "");
                }
              }
            }
          }
          session.detector.addStep(block.name, block.input || {}, output, thinking);
          thinking = "";
        }
      }
    }
  }

  // Get current window
  const window = session.detector.getWindow(WINDOW_SIZE);
  if (!window) return messages;

  // Normalize and classify
  const normalizedCont = window.continuous.map(row => normalizeFeatures(row));
  const { score, stuck } = classifyWindow(
    window.toolIndices, normalizedCont, window.windowFeatures
  );

  // Sliding window history: fire only if 2 of last 3 are above threshold
  session.windowScores.push(score);
  if (session.windowScores.length > HISTORY_SIZE) {
    session.windowScores.shift();
  }

  const aboveThreshold = session.windowScores.filter(s => s >= config.threshold).length;
  const shouldFire = aboveThreshold >= FIRE_THRESHOLD;

  if (!shouldFire) return messages;

  log?.("cnn_stuck_detected", {
    turnCount: session.turnCounter,
    score,
    threshold: config.threshold,
    windowScores: [...session.windowScores],
    stepCount: session.detector.stepCount,
  });

  session.lastNudgeTurn = session.turnCounter;

  // Build recent tool call summary for the nudge
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
          `[CONTEXT MONITOR — turn ${session.turnCounter}, confidence ${(score * 100).toFixed(0)}%]\n\n` +
          `Your recent actions show signs of repetitive patterns. ` +
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

  log?.("cnn_nudge_injected", {
    turnCount: session.turnCounter,
    score,
    method: "cnn",
    recentTools: recentTools.slice(-5),
  });

  return [...messages, nudge];
}
