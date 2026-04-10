#!/usr/bin/env node
/**
 * Nudge injection test — replay a session up to a stuck point, inject a
 * nudge, let the model run a few steps, display before/after for human review.
 *
 * Usage:
 *   ANTHROPIC_API_KEY=sk-... node proxy/test_nudge.mjs session.jsonl [opts]
 *
 * Options:
 *   --level    0|1|2   Nudge level to inject (default: 0)
 *   --step     N       Truncate after tool call N (default: auto = peak CNN window)
 *   --steps    N       API calls after first nudge (default: 3)
 *   --cooldown N       Re-fire nudge every N API calls if still high-scoring (default: 0 = off)
 *
 * Tool calls in "after" steps use stub results ("[test mode — tool not executed]")
 * so the model can keep going without an actual execution environment.
 */

import { readFileSync } from "fs";
import { StuckDetectorState } from "./abstract_step.mjs";
import { classifyWindow, normalizeFeatures, config } from "./classify_cnn.mjs";

const WINDOW_SIZE = config.window_size;

// ── CLI ──────────────────────────────────────────────────────────────────────

const argv = process.argv.slice(2);
const sessionFile = argv.find(a => !a.startsWith("-"));
if (!sessionFile) {
  console.log("Usage: node proxy/test_nudge.mjs session.jsonl [--level 0|1|2] [--step N] [--steps N] [--cooldown N]");
  process.exit(0);
}

function getFlag(name, def) {
  const i = argv.indexOf(`--${name}`);
  return i >= 0 ? argv[i + 1] : def;
}

const nudgeLevel  = parseInt(getFlag("level",    "0"), 10);
const forcedStep  = getFlag("step", null);
const stepsAfter  = parseInt(getFlag("steps",    "3"), 10);
const cooldown    = parseInt(getFlag("cooldown", "0"), 10);
const model       = process.env.ANTHROPIC_MODEL || "claude-sonnet-4-6";
const apiKey      = process.env.ANTHROPIC_API_KEY;
if (!apiKey) { console.error("Error: ANTHROPIC_API_KEY not set"); process.exit(1); }

// ── Session parsing ──────────────────────────────────────────────────────────

function parseSession(filepath) {
  const lines = readFileSync(filepath, "utf-8").trim().split("\n");
  const apiMessages  = [];   // API-format messages (thinking stripped)
  const toolCallList = [];   // flat ordered list of tool calls
  const outputMap    = new Map();

  // First pass: collect tool results by id
  for (const line of lines) {
    let entry; try { entry = JSON.parse(line); } catch { continue; }
    const msg = entry.message;
    if (!msg || msg.role !== "user" || !Array.isArray(msg.content)) continue;
    for (const b of msg.content) {
      if (b.type !== "tool_result") continue;
      const c = b.content;
      outputMap.set(b.tool_use_id,
        Array.isArray(c) ? c.filter(x => x.type === "text").map(x => x.text).join(" ")
                         : String(c || ""));
    }
  }

  // Second pass: build ordered messages and tool call index
  for (const line of lines) {
    let entry; try { entry = JSON.parse(line); } catch { continue; }
    const msg = entry.message;
    if (!msg || !Array.isArray(msg.content)) continue;

    if (msg.role === "assistant") {
      const content = msg.content.filter(b => b.type !== "thinking");
      if (!content.length) continue;
      const msgIdx = apiMessages.length;
      apiMessages.push({ role: "assistant", content });
      for (const b of content) {
        if (b.type === "tool_use") {
          toolCallList.push({
            id: b.id,
            name: b.name,
            input: b.input || {},
            output: outputMap.get(b.id) || "",
            msgIdx,
          });
        }
      }
    } else if (msg.role === "user") {
      apiMessages.push({ role: "user", content: msg.content });
    }
  }

  return { apiMessages, toolCallList };
}

// ── CNN scoring ──────────────────────────────────────────────────────────────

function scoreWindows(toolCallList) {
  const detector = new StuckDetectorState();
  for (const tc of toolCallList) {
    detector.addStep(tc.name, tc.input, tc.output, "");
  }
  const scores = [];
  for (let start = 0; start <= detector.abstractSteps.length - WINDOW_SIZE; start++) {
    const window   = detector.abstractSteps.slice(start, start + WINDOW_SIZE);
    const toolIdxs = window.map(s => s.tool_idx);
    const cont     = window.map(s => config.continuous_features.map(f => s[f] ?? 0));
    const normCont = cont.map(row => normalizeFeatures(row));
    const tools    = window.map(s => s.tool);
    const allLines = [];
    for (const s of window) if (s.output_set) for (const l of s.output_set) allLines.push(l);
    const wf = [
      new Set(tools).size / tools.length,
      1.0, 1.0,
      window.reduce((a, s) => a + s.is_error, 0) / window.length,
      window.reduce((a, s) => a + s.output_similarity, 0) / window.length,
      allLines.length > 0 ? new Set(allLines).size / allLines.length : 1.0,
    ];
    const { score } = classifyWindow(toolIdxs, normCont, wf);
    scores.push({ start, end: start + WINDOW_SIZE - 1, score });
  }
  return scores;
}

// ── Nudge text ───────────────────────────────────────────────────────────────

function makeNudgeText(level, turnNum, score, recentList) {
  const pct = (score * 100).toFixed(0);
  if (level === 0) {
    return `[CONTEXT MONITOR — turn ${turnNum}, confidence ${pct}%]\n\n` +
      `Your recent actions show signs of repetitive patterns. You may be going in circles.\n\n` +
      `Recent tool calls:\n  ${recentList}\n\n` +
      `Review your last few turns critically:\n` +
      `- Are you retrying the same approach with minor variations?\n` +
      `- Are you investigating the same files/functions repeatedly?\n` +
      `- Has your hypothesis changed or are you stuck on the same one?\n\n` +
      `If you are going in circles, try a fundamentally different strategy.\n` +
      `State what you have learned so far and what new approach you will try.`;
  } else if (level === 1) {
    return `[CONTEXT MONITOR — turn ${turnNum}, confidence ${pct}% — repeated signal]\n\n` +
      `You have been nudged before and the repetitive pattern continues.\n\n` +
      `Recent tool calls:\n  ${recentList}\n\n` +
      `You appear to be stuck in a loop. The approach you are using is not working.\n` +
      `Before your next tool call:\n` +
      `1. State in one sentence what you have been trying to do.\n` +
      `2. State specifically why it has not worked.\n` +
      `3. Propose a different approach you have not tried yet.\n\n` +
      `Do not retry the same command. Switch strategy.`;
  } else {
    return `[CONTEXT MONITOR — turn ${turnNum}, confidence ${pct}% — escalated]\n\n` +
      `STOP. You are deeply stuck and have not responded to prior nudges.\n\n` +
      `Recent tool calls:\n  ${recentList}\n\n` +
      `Do not run any more tool calls until you have answered these:\n` +
      `1. What is the root cause of the problem you are trying to solve?\n` +
      `2. What have you tried, and why did each attempt fail?\n` +
      `3. What fundamentally different approach will you take next?\n\n` +
      `If you cannot answer these, state that clearly and ask for guidance.`;
  }
}

// ── API ──────────────────────────────────────────────────────────────────────

async function callAPI(messages) {
  const res = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
    },
    body: JSON.stringify({ model, max_tokens: 2048, messages }),
  });
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`);
  return res.json();
}

// ── Display helpers ──────────────────────────────────────────────────────────

const W = 68;
function hr(c = "━") { return c.repeat(W); }
function fmtInput(input) {
  const v = input?.command || input?.file_path || input?.pattern || input?.query || "";
  return String(v).replace(/\n/g, " ").slice(0, 52);
}
function fmtOutput(output) {
  if (!output) return "—";
  return `${output.split("\n").length}L ${output.length}ch`;
}
function fmtContent(content) {
  const lines = [];
  for (const b of content) {
    if (b.type === "text" && b.text?.trim()) {
      const t = b.text.trim().replace(/\n/g, " ").slice(0, 300);
      lines.push(`  text: ${t}${b.text.length > 300 ? "…" : ""}`);
    } else if (b.type === "tool_use") {
      lines.push(`  → ${b.name}: ${fmtInput(b.input)}`);
    }
  }
  return lines.join("\n") || "  (no content)";
}

// ── Main ─────────────────────────────────────────────────────────────────────

const { apiMessages, toolCallList } = parseSession(sessionFile);
const sessionName = sessionFile.split("/").pop();

if (toolCallList.length < WINDOW_SIZE) {
  console.error(`Too few tool calls (${toolCallList.length} < ${WINDOW_SIZE})`);
  process.exit(1);
}

// Select cutoff
const windowScores = scoreWindows(toolCallList);
let cutoffStep, cnnScore;

if (forcedStep !== null) {
  cutoffStep = Math.min(parseInt(forcedStep, 10), toolCallList.length - 1);
  const w = windowScores.find(s => s.end === cutoffStep);
  cnnScore = w?.score ?? 0;
} else {
  const best = windowScores.reduce((a, b) => b.score > a.score ? b : a);
  cutoffStep = best.end;
  cnnScore = best.score;
}

// Build context messages: up to (and including) the user tool_result turn after cutoff
const cutoffMsgIdx = toolCallList[cutoffStep].msgIdx;
let contextMessages = apiMessages.slice(0, cutoffMsgIdx + 2);
// Ensure we end on a user message (complete exchange)
while (contextMessages.length > 0 && contextMessages.at(-1).role !== "user") {
  contextMessages = contextMessages.slice(0, -1);
}

// Build recent tool list for nudge text
const recentTcs = toolCallList.slice(Math.max(0, cutoffStep - 7), cutoffStep + 1);
const recentList = recentTcs.map(tc => `${tc.name}: ${fmtInput(tc.input)}`).join("\n  ");

const testMode = cnnScore < config.threshold ? "HARMLESSNESS TEST (productive zone)" : "EFFECTIVENESS TEST (stuck zone)";

// ── Header ───────────────────────────────────────────────────────────────────

console.log(hr());
console.log(`SESSION   ${sessionName}`);
console.log(`CUTOFF    step ${cutoffStep} / ${toolCallList.length - 1}   CNN ${cnnScore.toFixed(3)} (threshold ${config.threshold})   ${testMode}`);
console.log(`NUDGE     level ${nudgeLevel} / ${["soft","medium","hard"][nudgeLevel]}   steps-after ${stepsAfter}   cooldown ${cooldown || "off"}`);
console.log(hr());

// ── BEFORE ───────────────────────────────────────────────────────────────────

console.log(`\nBEFORE — last ${recentTcs.length} tool calls`);
recentTcs.forEach((tc, i) => {
  const idx = cutoffStep - recentTcs.length + 1 + i;
  const marker = idx === cutoffStep ? ">" : " ";
  console.log(`  ${marker}[${String(idx).padStart(3)}] ${tc.name.padEnd(8)} ${fmtInput(tc.input).padEnd(52)}  ${fmtOutput(tc.output)}`);
});

// ── Nudge ────────────────────────────────────────────────────────────────────

const firstNudgeText = makeNudgeText(nudgeLevel, cutoffStep + 1, cnnScore, recentList);
console.log(`\n${"─".repeat(W)}`);
console.log(`NUDGE injected (level ${nudgeLevel})`);
console.log("─".repeat(W));
for (const line of firstNudgeText.split("\n")) console.log(`  ${line}`);
console.log("─".repeat(W));

// ── AFTER: loop stepsAfter API calls ─────────────────────────────────────────

let messages = [...contextMessages, { role: "user", content: [{ type: "text", text: firstNudgeText }] }];
let currentNudgeLevel = nudgeLevel;
let stepsSinceLastNudge = 0;

for (let step = 1; step <= stepsAfter; step++) {
  console.log(`\nAFTER — step ${step} / ${stepsAfter}`);

  let response;
  try {
    response = await callAPI(messages);
  } catch (e) {
    console.error(`  API error: ${e.message}`);
    break;
  }

  const content = response.content || [];
  console.log(fmtContent(content));

  // Append assistant response
  messages = [...messages, { role: "assistant", content }];
  stepsSinceLastNudge++;

  const toolUses = content.filter(b => b.type === "tool_use");
  if (toolUses.length === 0) {
    console.log("  (no tool calls — model responded with text only)");
    break;
  }
  if (step === stepsAfter) break;

  // Stub tool results
  const stubResults = toolUses.map(tu => ({
    type: "tool_result",
    tool_use_id: tu.id,
    content: [{ type: "text", text: "[test mode — tool not executed]" }],
  }));
  messages = [...messages, { role: "user", content: stubResults }];

  // Re-fire nudge on cooldown if enabled
  if (cooldown > 0 && stepsSinceLastNudge >= cooldown) {
    currentNudgeLevel = Math.min(currentNudgeLevel + 1, 2);
    const reNudgeText = makeNudgeText(currentNudgeLevel, cutoffStep + step + 1, cnnScore, recentList);
    console.log(`\n${"─".repeat(W)}`);
    console.log(`RE-NUDGE at step ${step} (level escalated to ${currentNudgeLevel} / ${["soft","medium","hard"][currentNudgeLevel]})`);
    console.log("─".repeat(W));
    for (const line of reNudgeText.split("\n")) console.log(`  ${line}`);
    console.log("─".repeat(W));
    messages = [...messages, { role: "user", content: [{ type: "text", text: reNudgeText }] }];
    stepsSinceLastNudge = 0;
  }
}

console.log(`\n${hr()}`);
