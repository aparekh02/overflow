/**
 * scenarioAI — Converts natural language into CustomScenarioDef via LLM.
 * Supports OpenAI-compatible APIs. Stores API key in localStorage.
 */

import type { CustomScenarioDef } from "../mockData";

const LS_KEY = "openenv_openai_key";
const DEFAULT_KEY = "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";

export function getApiKey(): string {
  return localStorage.getItem(LS_KEY) || DEFAULT_KEY;
}

export function setApiKey(key: string) {
  localStorage.setItem(LS_KEY, key);
}

const SYSTEM_PROMPT = `You are a driving scenario generator for a self-driving car perception visualization tool called OpenENV.

When the user describes a driving scenario in natural language, you must respond with ONLY a valid JSON object (no markdown, no explanation) that matches this schema:

{
  "name": "Short scenario name",
  "description": "One-line description",
  "severity": "none" | "warning" | "critical",
  "ego": {
    "speed": <number, m/s, typical city driving is 8-14>,
    "events": [
      {
        "time": <seconds when event starts>,
        "action": "brake" | "swerve_left" | "swerve_right" | "accelerate" | "stop",
        "intensity": <0.0 to 1.0, how hard>
      }
    ]
  },
  "actors": [
    {
      "type": "vehicle" | "pedestrian" | "cyclist" | "sign",
      "label": "Human-readable label",
      "size": [length, width, height],
      "startX": <meters ahead of ego at t=0, positive = ahead>,
      "startY": <meters left of ego at t=0, negative = right side>,
      "heading": <radians, 0 = same direction as ego, PI = oncoming>,
      "speed": <m/s>,
      "events": [
        {
          "time": <seconds>,
          "speed": <new speed>,
          "heading": <new heading>,
          "targetY": <new lateral position>
        }
      ]
    }
  ],
  "incident": {
    "startTime": <seconds>,
    "endTime": <seconds>,
    "peakTime": <most dangerous moment in seconds>,
    "description": "What's happening"
  }
}

COORDINATE SYSTEM (Waymo convention):
- X axis = forward (direction ego is driving)
- Y axis = left (positive = left of ego, negative = right of ego)
- Standard lane width = 3.7m
- Ego drives in right lane, center at Y ≈ -1.85
- Left lane center at Y ≈ 1.85
- Oncoming traffic at Y ≈ 6.55 and Y ≈ 10.25
- Right sidewalk at Y ≈ -7.0
- Left sidewalk at Y ≈ 13.0

TYPICAL SIZES:
- Sedan: [4.8, 2.1, 1.5]
- SUV: [5.0, 2.2, 1.9]
- Truck: [6.5, 2.5, 3.0]
- Pedestrian: [0.6, 0.6, 1.75]
- Cyclist: [1.8, 0.7, 1.7]
- Sign: [0.1, 0.8, 1.2]

TYPICAL SPEEDS (m/s):
- City driving: 8-14 m/s (18-31 mph)
- Highway: 25-35 m/s (56-78 mph)
- Pedestrian walking: 1.2-1.5 m/s
- Pedestrian running: 2.5-4.0 m/s
- Cyclist: 4-8 m/s

IMPORTANT:
- The scene lasts 19.8 seconds (198 frames at 10 fps)
- Include parked vehicles, pedestrians on sidewalks for realism
- The ego should REACT realistically (brake for danger, stop at collision, swerve if needed)
- Make incidents clear and dramatic — the user needs to see what's happening
- startX is relative to where ego starts. Ego moves forward, so actors at startX=50 are 50m ahead at t=0
- Always include an "incident" object for dangerous scenarios
- Respond with ONLY the JSON, nothing else`;

export interface ChatMessage {
  role: "user" | "assistant" | "system";
  content: string;
}

export async function generateScenario(
  prompt: string,
  onStatus?: (msg: string) => void,
): Promise<{ scenario: CustomScenarioDef | null; error: string | null; raw: string }> {
  const key = getApiKey();
  if (!key) {
    return { scenario: null, error: "No API key set. Enter your OpenAI API key above.", raw: "" };
  }

  onStatus?.("Sending to AI…");

  try {
    const res = await fetch("https://api.openai.com/v1/chat/completions", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${key}`,
      },
      body: JSON.stringify({
        model: "gpt-4o-mini",
        temperature: 0.7,
        max_tokens: 2000,
        messages: [
          { role: "system", content: SYSTEM_PROMPT },
          { role: "user", content: prompt },
        ],
      }),
    });

    if (!res.ok) {
      const errBody = await res.text();
      if (res.status === 401) {
        return { scenario: null, error: "Invalid API key. Check your OpenAI key.", raw: errBody };
      }
      return { scenario: null, error: `API error ${res.status}: ${errBody.slice(0, 200)}`, raw: errBody };
    }

    onStatus?.("Parsing response…");

    const data = await res.json();
    const content: string = data.choices?.[0]?.message?.content ?? "";

    // Extract JSON from response (handle markdown code blocks)
    let jsonStr = content.trim();
    const jsonMatch = jsonStr.match(/```(?:json)?\s*([\s\S]*?)```/);
    if (jsonMatch) jsonStr = jsonMatch[1].trim();

    const parsed = JSON.parse(jsonStr) as CustomScenarioDef;

    // Validate basic structure
    if (!parsed.name || !parsed.ego || !Array.isArray(parsed.actors)) {
      return { scenario: null, error: "AI returned invalid scenario structure.", raw: content };
    }

    // Ensure defaults
    if (!parsed.severity) parsed.severity = "warning";
    if (!parsed.ego.speed) parsed.ego.speed = 11;
    for (const actor of parsed.actors) {
      if (!actor.size || actor.size.length !== 3) {
        actor.size = actor.type === "vehicle" ? [4.8, 2.1, 1.5]
          : actor.type === "pedestrian" ? [0.6, 0.6, 1.75]
          : actor.type === "cyclist" ? [1.8, 0.7, 1.7]
          : [0.1, 0.8, 1.2];
      }
    }

    onStatus?.("Scenario ready!");
    return { scenario: parsed, error: null, raw: content };
  } catch (e) {
    const msg = e instanceof Error ? e.message : String(e);
    if (msg.includes("JSON")) {
      return { scenario: null, error: `AI returned invalid JSON: ${msg}`, raw: "" };
    }
    return { scenario: null, error: `Error: ${msg}`, raw: "" };
  }
}
