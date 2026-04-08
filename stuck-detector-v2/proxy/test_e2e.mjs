/**
 * End-to-end test: parse a real Claude Code session, run CNN, report scores.
 */

import { readFileSync } from "fs";
import { StuckDetectorState } from "./abstract_step.mjs";
import { classifyWindow, normalizeFeatures, config } from "./classify_cnn.mjs";

const WINDOW_SIZE = config.window_size;

// Parse a Claude Code session JSONL
function parseSession(filepath) {
  const lines = readFileSync(filepath, "utf-8").trim().split("\n");
  const toolCalls = [];

  const pendingThinking = new Map(); // tool_use_id → thinking

  for (const line of lines) {
    const entry = JSON.parse(line);
    if (!entry.message) continue;
    const msg = entry.message;

    if (msg.role === "assistant" && Array.isArray(msg.content)) {
      let thinking = "";
      for (const block of msg.content) {
        if (block.type === "thinking") {
          thinking = block.thinking || "";
        } else if (block.type === "tool_use") {
          toolCalls.push({
            id: block.id,
            name: block.name,
            input: block.input || {},
            output: "",
            thinking,
          });
          thinking = "";
        }
      }
    } else if (msg.role === "user" && Array.isArray(msg.content)) {
      for (const block of msg.content) {
        if (block.type === "tool_result") {
          const tc = toolCalls.find(t => t.id === block.tool_use_id);
          if (tc) {
            const content = block.content;
            tc.output = Array.isArray(content)
              ? content.filter(b => b.type === "text").map(b => b.text).join(" ")
              : String(content || "");
          }
        }
      }
    }
  }

  return toolCalls;
}

// Run the CNN on a session
function analyzeSession(filepath) {
  const toolCalls = parseSession(filepath);
  const detector = new StuckDetectorState();

  console.log(`Session: ${filepath.split("/").pop()}`);
  console.log(`  Tool calls: ${toolCalls.length}`);

  if (toolCalls.length < WINDOW_SIZE) {
    console.log("  Too few tool calls for CNN analysis\n");
    return;
  }

  // Process all tool calls
  for (const tc of toolCalls) {
    detector.addStep(tc.name, tc.input, tc.output, tc.thinking);
  }

  // Evaluate every possible window
  const scores = [];
  const stride = 5;
  const allSteps = detector.abstractSteps;

  for (let start = 0; start <= allSteps.length - WINDOW_SIZE; start += stride) {
    const window = allSteps.slice(start, start + WINDOW_SIZE);
    const toolIndices = window.map(s => s.tool_idx);
    const continuous = window.map(s => [
      s.steps_since_same_tool, s.steps_since_same_file, s.steps_since_same_cmd,
      s.tool_count_in_window, s.file_count_in_window, s.cmd_count_in_window,
      s.output_similarity, s.output_length, s.is_error, s.step_index_norm,
      s.false_start, s.strategy_change, s.circular_lang,
      s.thinking_length, s.self_similarity,
    ]);

    // Compute window features
    const tools = window.map(s => s.tool);
    const uniqueToolsRatio = new Set(tools).size / tools.length;
    const errorRate = window.reduce((a, s) => a + s.is_error, 0) / window.length;
    const outputSimAvg = window.reduce((a, s) => a + s.output_similarity, 0) / window.length;

    // Simplified window features (files/cmds from step data)
    const allLines = [];
    for (const s of window) {
      if (s.output_set) for (const l of s.output_set) allLines.push(l);
    }
    const outputDiversity = allLines.length > 0 ? new Set(allLines).size / allLines.length : 1.0;

    const windowFeatures = [uniqueToolsRatio, 1.0, 1.0, errorRate, outputSimAvg, outputDiversity];

    const normalizedCont = continuous.map(row => normalizeFeatures(row));
    const { score } = classifyWindow(toolIndices, normalizedCont, windowFeatures);
    scores.push({ start, score });
  }

  // Report
  const maxScore = Math.max(...scores.map(s => s.score));
  const meanScore = scores.reduce((a, s) => a + s.score, 0) / scores.length;
  const stuckWindows = scores.filter(s => s.score >= config.threshold).length;

  console.log(`  Windows: ${scores.length}`);
  console.log(`  Scores: mean=${meanScore.toFixed(4)} max=${maxScore.toFixed(4)}`);
  console.log(`  Stuck windows (>=${config.threshold}): ${stuckWindows}`);

  // Show top 5 scoring windows
  const top5 = [...scores].sort((a, b) => b.score - a.score).slice(0, 5);
  console.log(`  Top 5 windows:`);
  for (const s of top5) {
    const w = allSteps.slice(s.start, s.start + WINDOW_SIZE);
    const tools = w.map(st => st.tool).join(" → ");
    console.log(`    start=${s.start} score=${s.score.toFixed(4)} tools=[${tools}]`);
  }
  console.log();
}

// Test on known sessions
import { readdirSync } from "fs";

const testTasks = [
  { dir: "/home/nicolas/.claude/projects/-home-nicolas-source-classifier-repos-worktrees-02-gcc-bug/", label: "STUCK" },
  { dir: "/home/nicolas/.claude/projects/-home-nicolas-source-classifier-repos-worktrees-08-express-bug/", label: "PRODUCTIVE" },
  { dir: "/home/nicolas/.claude/projects/-home-nicolas-source-classifier-repos-worktrees-03-llvm-bug/", label: "STUCK" },
  { dir: "/home/nicolas/.claude/projects/-home-nicolas-source-classifier-repos-worktrees-10-linux-usb-bug/", label: "PRODUCTIVE" },
];

for (const task of testTasks) {
  console.log(`=== ${task.dir.split("worktrees-")[1].replace("/", "")} (expected: ${task.label}) ===`);
  try {
    const files = readdirSync(task.dir).filter(f => f.endsWith(".jsonl")).slice(0, 2);
    for (const f of files) {
      analyzeSession(task.dir + f);
    }
  } catch (e) {
    console.log(`  Error: ${e.message}\n`);
  }
}
