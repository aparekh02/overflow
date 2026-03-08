/**
 * Global state store — Zustand-based, supporting both mock and real Waymo data.
 */

import { create } from "zustand";
import type { FrameData, SceneData } from "./mockData";

export type ColormapMode = "intensity" | "range" | "elongation";
export type BoxDisplayMode = "off" | "box" | "model";
export type DataSource = "mock" | "waymo" | "waymo-drop";
export type LoadStatus = "idle" | "loading" | "ready" | "error";

interface StoreState {
  // Data
  sceneData: SceneData | null;
  dataSource: DataSource;
  loadStatus: LoadStatus;
  loadMessage: string;
  loadProgress: number;
  loadError: string | null;

  // Playback
  currentFrameIndex: number;
  isPlaying: boolean;
  playbackSpeed: number;

  // Display
  colormapMode: ColormapMode;
  boxMode: BoxDisplayMode;
  pointOpacity: number;
  showGrid: boolean;

  // Computed
  currentFrame: FrameData | null;
  totalFrames: number;

  // Actions
  actions: {
    setSceneData: (data: SceneData) => void;
    setFrame: (index: number) => void;
    nextFrame: () => void;
    prevFrame: () => void;
    togglePlay: () => void;
    setPlaying: (v: boolean) => void;
    setPlaybackSpeed: (speed: number) => void;
    setColormapMode: (mode: ColormapMode) => void;
    setBoxMode: (mode: BoxDisplayMode) => void;
    setPointOpacity: (v: number) => void;
    toggleGrid: () => void;
    setDataSource: (source: DataSource) => void;
    setLoadStatus: (status: LoadStatus) => void;
    setLoadMessage: (msg: string) => void;
    setLoadProgress: (p: number) => void;
    setLoadError: (err: string | null) => void;
    reset: () => void;
  };
}

export const useStore = create<StoreState>((set, get) => ({
  sceneData: null,
  dataSource: "waymo",
  loadStatus: "idle",
  loadMessage: "",
  loadProgress: 0,
  loadError: null,
  currentFrameIndex: 0,
  isPlaying: false,
  playbackSpeed: 1,
  colormapMode: "intensity",
  boxMode: "box",
  pointOpacity: 0.85,
  showGrid: true,
  currentFrame: null,
  totalFrames: 0,

  actions: {
    setSceneData: (data) =>
      set({
        sceneData: data,
        totalFrames: data.totalFrames,
        currentFrameIndex: 0,
        currentFrame: data.frames[0] ?? null,
        loadStatus: "ready",
        loadError: null,
      }),

    setFrame: (index) => {
      const { sceneData } = get();
      if (!sceneData) return;
      const clamped = Math.max(0, Math.min(sceneData.totalFrames - 1, index));
      set({ currentFrameIndex: clamped, currentFrame: sceneData.frames[clamped] });
    },

    nextFrame: () => {
      const { sceneData, currentFrameIndex } = get();
      if (!sceneData) return;
      const next = (currentFrameIndex + 1) % sceneData.totalFrames;
      set({ currentFrameIndex: next, currentFrame: sceneData.frames[next] });
    },

    prevFrame: () => {
      const { sceneData, currentFrameIndex } = get();
      if (!sceneData) return;
      const prev = (currentFrameIndex - 1 + sceneData.totalFrames) % sceneData.totalFrames;
      set({ currentFrameIndex: prev, currentFrame: sceneData.frames[prev] });
    },

    togglePlay: () => set((s) => ({ isPlaying: !s.isPlaying })),
    setPlaying: (v) => set({ isPlaying: v }),
    setPlaybackSpeed: (speed) => set({ playbackSpeed: speed }),
    setColormapMode: (mode) => set({ colormapMode: mode }),
    setBoxMode: (mode) => set({ boxMode: mode }),
    setPointOpacity: (v) => set({ pointOpacity: v }),
    toggleGrid: () => set((s) => ({ showGrid: !s.showGrid })),
    setDataSource: (source) => set({ dataSource: source }),
    setLoadStatus: (status) => set({ loadStatus: status }),
    setLoadMessage: (msg) => set({ loadMessage: msg }),
    setLoadProgress: (p) => set({ loadProgress: p }),
    setLoadError: (err) => set({ loadError: err }),
    reset: () =>
      set({
        sceneData: null,
        currentFrameIndex: 0,
        currentFrame: null,
        totalFrames: 0,
        isPlaying: false,
        loadStatus: "idle",
        loadMessage: "",
        loadProgress: 0,
        loadError: null,
      }),
  },
}));
