/**
 * rlStore — Zustand store for OpenENV RL mode state.
 * Kept separate from the main store so Waymo replay and RL mode
 * can coexist without cross-contamination.
 */

import { create } from "zustand";
import type { RLSceneState, RLConstraints } from "./middleware/types";

export interface EpisodeInfo {
  incidentType: string;
  constraints: Record<string, number | string>;
  startTime: number;
}

export interface EpisodeEnd {
  totalReward: number;
  steps: number;
}

interface RLStoreState {
  // Connection
  connected: boolean;
  paused: boolean;
  error: string | null;

  // Active RL mode
  rlModeActive: boolean;

  // Scene
  rlScene: RLSceneState | null;

  // Episode metadata
  episodeInfo: EpisodeInfo | null;
  episodeEnd: EpisodeEnd | null;
  rewardHistory: number[];   // last N step rewards for sparkline

  actions: {
    setConnected: (v: boolean) => void;
    setPaused: (v: boolean) => void;
    setError: (msg: string | null) => void;
    setRLModeActive: (v: boolean) => void;
    setRLScene: (scene: RLSceneState) => void;
    setEpisodeInfo: (info: EpisodeInfo) => void;
    setEpisodeEnd: (end: EpisodeEnd) => void;
    resetRewardHistory: () => void;
    appendReward: (r: number) => void;
  };
}

const MAX_REWARD_HISTORY = 200;

export const useRLStore = create<RLStoreState>((set) => ({
  connected: false,
  paused: false,
  error: null,
  rlModeActive: false,
  rlScene: null,
  episodeInfo: null,
  episodeEnd: null,
  rewardHistory: [],

  actions: {
    setConnected: (v) => set({ connected: v }),
    setPaused: (v) => set({ paused: v }),
    setError: (msg) => set({ error: msg }),
    setRLModeActive: (v) => set({ rlModeActive: v }),
    setRLScene: (scene) =>
      set((s) => {
        const newHistory = [...s.rewardHistory, scene.reward].slice(-MAX_REWARD_HISTORY);
        return { rlScene: scene, rewardHistory: newHistory };
      }),
    setEpisodeInfo: (info) => set({ episodeInfo: info, episodeEnd: null }),
    setEpisodeEnd: (end) => set({ episodeEnd: end }),
    resetRewardHistory: () => set({ rewardHistory: [] }),
    appendReward: (r) =>
      set((s) => ({ rewardHistory: [...s.rewardHistory, r].slice(-MAX_REWARD_HISTORY) })),
  },
}));
