/**
 * Shared types for the OpenENV ↔ 3D Sim WebSocket protocol.
 */

export interface RLAgentState {
  id: string;
  type: "review" | "traffic" | "incident" | "pedestrian" | "cyclist";
  x: number;
  y: number;
  vx: number;
  vy: number;
  yaw: number;
  yaw_rate: number;
  speed: number;
  accel: number;
  length: number;
  width: number;
  height: number;
  /** Predicted trajectory points [[x,y], ...] (3s horizon) */
  trajectory: [number, number][];
  /** Attention weight from review agent (0..1) */
  attention: number;
}

export interface RLConstraints {
  weather: "clear" | "rain" | "heavy_rain" | "fog" | "snow" | "overcast";
  friction: number;         // 0.25 → 1.0
  visibility_m: number;     // metres
  traffic_density: number;  // 0 → 1
  speed_limit_ms: number;
}

export interface RewardComponents {
  collision: number;
  comfort: number;
  efficiency: number;
  safety_margin: number;
  incident_prevention: number;
  constraint_compliance: number;
  total: number;
}

export interface RLSceneState {
  step: number;
  timestamp: number;
  ego: RLAgentState;
  agents: RLAgentState[];
  constraints: RLConstraints;
  reward: number;
  episode_reward: number;
  done: boolean;
  incident_type: string;
  ticket_id: string;
  reward_components: RewardComponents;
}

// Inbound messages (Python → Browser)
export type ServerMessage =
  | { type: "connected"; message: string; host: string; port: number }
  | { type: "scene_update"; state: RLSceneState }
  | { type: "episode_start"; incident_type: string; constraints: Record<string, unknown> }
  | { type: "episode_end"; total_reward: number; steps: number }
  | { type: "paused" }
  | { type: "resumed" }
  | { type: "pong" }
  | { type: "error"; message: string };

// Outbound messages (Browser → Python)
export type ClientMessage =
  | { type: "load_incident"; ticket: { id: string; text: string; severity?: string } }
  | { type: "pause" }
  | { type: "resume" }
  | { type: "reset" }
  | { type: "ghost_action"; action: [number, number, number] }
  | { type: "clear_ghost" }
  | { type: "waymo_frame"; data: { pointCount: number; boxes: unknown[]; avgIntensity?: number } }
  | { type: "ping" };
