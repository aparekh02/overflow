/**
 * wsClient — WebSocket client that connects the 3D sim to the Python OpenENV.
 *
 * Novel: This is not a passive telemetry receiver. It's a two-way coupling:
 *   • Streams current Waymo frame stats → Python env updates its constraints
 *   • Receives live RL scene states → overrides the 3D scene
 *   • User can submit incident tickets, pause, reset, or inject ghost actions
 *
 * The store is updated directly so existing Three.js components re-use the
 * same rendering pipeline for both Waymo replay and RL simulation modes.
 */

import type { ServerMessage, ClientMessage, RLSceneState } from "./types";
import { useRLStore } from "../rlStore";
import { useStore } from "../store";

const WS_URL = "ws://localhost:8765";
const RECONNECT_DELAY_MS = 2000;

class SimWebSocketClient {
  private ws: WebSocket | null = null;
  private reconnectTimer: ReturnType<typeof setTimeout> | null = null;
  private _connected = false;
  private _enabled = false;

  /** Start connecting. Call once when user switches to OpenENV mode. */
  connect(): void {
    this._enabled = true;
    this._connect();
  }

  /** Disconnect and stop reconnecting. */
  disconnect(): void {
    this._enabled = false;
    if (this.reconnectTimer) clearTimeout(this.reconnectTimer);
    this.ws?.close();
    this.ws = null;
    this._connected = false;
    useRLStore.getState().actions.setConnected(false);
  }

  get connected(): boolean {
    return this._connected;
  }

  // ── Send helpers ────────────────────────────────────────────────────────

  send(msg: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  loadIncident(id: string, text: string, severity = "moderate"): void {
    this.send({ type: "load_incident", ticket: { id, text, severity } });
  }

  pause(): void { this.send({ type: "pause" }); }
  resume(): void { this.send({ type: "resume" }); }
  reset(): void { this.send({ type: "reset" }); }

  ghostAction(steer: number, throttle: number, brake: number): void {
    this.send({ type: "ghost_action", action: [steer, throttle, brake] });
  }

  clearGhost(): void { this.send({ type: "clear_ghost" }); }

  /** Forward current Waymo frame stats so Python env can update constraints. */
  sendWaymoFrame(pointCount: number, boxes: unknown[], avgIntensity = 0.5): void {
    this.send({ type: "waymo_frame", data: { pointCount, boxes, avgIntensity } });
  }

  // ── Private ─────────────────────────────────────────────────────────────

  private _connect(): void {
    if (!this._enabled) return;
    try {
      this.ws = new WebSocket(WS_URL);

      this.ws.onopen = () => {
        this._connected = true;
        if (this.reconnectTimer) { clearTimeout(this.reconnectTimer); this.reconnectTimer = null; }
        console.log("[wsClient] Connected to OpenENV middleware");
        useRLStore.getState().actions.setConnected(true);
        useRLStore.getState().actions.setError(null);
        // Ping to confirm
        this.send({ type: "ping" });
      };

      this.ws.onmessage = (ev) => {
        try {
          const msg: ServerMessage = JSON.parse(ev.data as string);
          this._handleMessage(msg);
        } catch (e) {
          console.error("[wsClient] Parse error:", e);
        }
      };

      this.ws.onerror = () => {
        useRLStore.getState().actions.setError("WebSocket error — is the Python server running?");
      };

      this.ws.onclose = () => {
        this._connected = false;
        useRLStore.getState().actions.setConnected(false);
        if (this._enabled) {
          console.log(`[wsClient] Disconnected. Reconnecting in ${RECONNECT_DELAY_MS}ms…`);
          this.reconnectTimer = setTimeout(() => this._connect(), RECONNECT_DELAY_MS);
        }
      };
    } catch (e) {
      console.error("[wsClient] Could not create WebSocket:", e);
      if (this._enabled) {
        this.reconnectTimer = setTimeout(() => this._connect(), RECONNECT_DELAY_MS);
      }
    }
  }

  private _handleMessage(msg: ServerMessage): void {
    const actions = useRLStore.getState().actions;

    switch (msg.type) {
      case "connected":
        console.log("[wsClient] Server:", msg.message);
        break;

      case "scene_update":
        actions.setRLScene(msg.state);
        break;

      case "episode_start":
        actions.setEpisodeInfo({
          incidentType: msg.incident_type,
          constraints: msg.constraints as Record<string, number | string>,
          startTime: Date.now(),
        });
        actions.resetRewardHistory();
        // Rewind the Waymo scene to frame 0 and play — each episode
        // replays the same city clip so context always matches the incident
        {
          const waymoActions = useStore.getState().actions;
          waymoActions.setFrame(0);
          waymoActions.setPlaying(true);
        }
        console.log(`[wsClient] Episode start: ${msg.incident_type}`);
        break;

      case "episode_end":
        actions.setEpisodeEnd({ totalReward: msg.total_reward, steps: msg.steps });
        console.log(`[wsClient] Episode end: reward=${msg.total_reward.toFixed(2)} steps=${msg.steps}`);
        break;

      case "paused":
        actions.setPaused(true);
        break;

      case "resumed":
        actions.setPaused(false);
        break;

      case "error":
        console.error("[wsClient] Server error:", msg.message);
        actions.setError(msg.message);
        break;

      default:
        break;
    }
  }
}

// Singleton
export const wsClient = new SimWebSocketClient();
