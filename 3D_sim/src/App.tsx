/**
 * App — Full-viewport 3D perception viewer.
 * The 3D canvas IS the app. All UI floats on top as glass overlays.
 */

import { useEffect, useState, useCallback, useRef } from "react";
import Scene3D from "./components/Scene3D";
import ControlPanel from "./components/ControlPanel";
import Timeline from "./components/Timeline";
import InfoBar from "./components/InfoBar";
import IncidentPanel from "./components/IncidentPanel";
import RLStatsPanel from "./components/RLStatsPanel";
import { useStore } from "./store";
import { useRLStore } from "./rlStore";
import { wsClient } from "./middleware/wsClient";
import type { DataSource } from "./store";
import { generateSceneData } from "./mockData";
import {
  loadWaymoFromUrls,
  loadWaymoFromFiles,
  scanDroppedFiles,
  type WaymoLoadResult,
} from "./utils/waymoLoader";
import { colors, fonts } from "./theme";

// ---------------------------------------------------------------------------
// Auto-detect waymo data layout
// ---------------------------------------------------------------------------

async function detectWaymoLayout(
  basePath: string,
): Promise<{ basePath: string; segmentName?: string }> {
  try {
    const resp = await fetch(`${basePath}/manifest.json`);
    if (resp.ok) {
      const manifest = await resp.json();
      if (manifest.segment) {
        console.log(`[waymo] Manifest found — segment: ${manifest.segment}`);
        return { basePath, segmentName: manifest.segment };
      }
    }
  } catch { /* no manifest */ }

  try {
    const resp = await fetch(`${basePath}/vehicle_pose.parquet`, { method: "HEAD" });
    if (resp.ok) return { basePath };
  } catch { /* not flat */ }

  throw new Error("No Waymo data found. Place parquet files in public/waymo_data/ or drag & drop.");
}

// ---------------------------------------------------------------------------
// Data loading hook
// ---------------------------------------------------------------------------

function useDataLoader() {
  const dataSource = useStore((s) => s.dataSource);
  const loadStatus = useStore((s) => s.loadStatus);
  const actions = useStore((s) => s.actions);

  useEffect(() => {
    if (loadStatus !== "idle") return;
    if (dataSource === "waymo-drop") return;

    if (dataSource === "mock") {
      actions.setLoadStatus("loading");
      actions.setLoadMessage("Generating mock scene…");
      actions.setLoadProgress(0.5);
      setTimeout(() => {
        try {
          actions.setSceneData(generateSceneData());
        } catch (e) {
          actions.setLoadError(e instanceof Error ? e.message : String(e));
          actions.setLoadStatus("error");
        }
      }, 0);
    } else if (dataSource === "waymo") {
      actions.setLoadStatus("loading");
      actions.setLoadMessage("Detecting data layout…");
      actions.setLoadProgress(0);

      detectWaymoLayout("/waymo_data")
        .then(({ basePath, segmentName }) => {
          actions.setLoadMessage("Opening Parquet files…");
          return loadWaymoFromUrls(basePath, (step, progress) => {
            actions.setLoadMessage(step);
            actions.setLoadProgress(progress);
          }, segmentName);
        })
        .then((data: WaymoLoadResult) => {
          console.log("[loadWaymo] Success:", data.totalFrames, "frames");
          actions.setSceneData(data);
        })
        .catch((e) => {
          console.error("[loadWaymo] Error:", e);
          actions.setLoadError(e instanceof Error ? e.message : String(e));
          actions.setLoadStatus("error");
        });
    }
  }, [dataSource, loadStatus, actions]);
}

// ---------------------------------------------------------------------------
// Loading screen
// ---------------------------------------------------------------------------

function LoadingScreen() {
  const loadMessage = useStore((s) => s.loadMessage);
  const loadProgress = useStore((s) => s.loadProgress);

  return (
    <div style={{
      position: "fixed", inset: 0,
      display: "flex", alignItems: "center", justifyContent: "center",
      backgroundColor: colors.bgDeep, flexDirection: "column", gap: 20,
      fontFamily: fonts.sans, zIndex: 100,
    }}>
      <div style={{
        width: 32, height: 32,
        border: `2px solid ${colors.accent}`,
        borderTopColor: "transparent",
        borderRadius: "50%",
        animation: "spin 0.7s linear infinite",
      }} />
      <div style={{
        width: 200, height: 3,
        backgroundColor: "rgba(255,255,255,0.06)",
        borderRadius: 2, overflow: "hidden",
      }}>
        <div style={{
          height: "100%",
          width: `${Math.round(loadProgress * 100)}%`,
          background: `linear-gradient(90deg, ${colors.accent}, ${colors.accentBlue})`,
          borderRadius: 2, transition: "width 0.3s ease-out",
        }} />
      </div>
      <span style={{ color: colors.textDim, fontSize: 11, fontFamily: fonts.mono }}>
        {loadMessage}
      </span>
      <style>{`@keyframes spin { to { transform: rotate(360deg); } }`}</style>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Error screen
// ---------------------------------------------------------------------------

function ErrorScreen({ onOpenENV }: { onOpenENV?: () => void }) {
  const loadError = useStore((s) => s.loadError);
  const actions = useStore((s) => s.actions);

  return (
    <div style={{
      position: "fixed", inset: 0,
      display: "flex", alignItems: "center", justifyContent: "center",
      backgroundColor: colors.bgDeep, flexDirection: "column", gap: 16,
      fontFamily: fonts.sans,
    }}>
      <div style={{ fontSize: 28, opacity: 0.6 }}>⚠</div>
      <div style={{ maxWidth: 380, textAlign: "center", lineHeight: 1.5, fontSize: 13, color: "#FF6B6B" }}>
        {loadError || "Unknown error"}
      </div>
      <div style={{ display: "flex", gap: 10, marginTop: 8, flexWrap: "wrap", justifyContent: "center" }}>
        <GlassButton label="Retry Waymo" onClick={() => { actions.reset(); actions.setDataSource("waymo"); }} accent />
        <GlassButton label="Use Mock" onClick={() => { actions.reset(); actions.setDataSource("mock"); }} />
        {onOpenENV && (
          <GlassButton label="Launch OpenENV ●" onClick={onOpenENV} openenv />
        )}
      </div>
      {onOpenENV && (
        <div style={{ fontSize: 10, fontFamily: fonts.mono, color: colors.textDim, marginTop: 4 }}>
          Run <span style={{ color: colors.accent }}>python3 openenv/run_demo.py</span> first
        </div>
      )}
    </div>
  );
}

function GlassButton({ label, onClick, accent, openenv }: { label: string; onClick: () => void; accent?: boolean; openenv?: boolean }) {
  return (
    <button onClick={onClick} style={{
      padding: "7px 18px", fontSize: 12, fontFamily: fonts.sans, fontWeight: 500,
      background: openenv ? "rgba(0,255,136,0.12)" : accent ? "rgba(0,232,157,0.1)" : "rgba(255,255,255,0.04)",
      color: openenv ? "#00FF88" : accent ? colors.accent : colors.textSecondary,
      border: `1px solid ${openenv ? "rgba(0,255,136,0.35)" : accent ? "rgba(0,232,157,0.3)" : colors.border}`,
      borderRadius: 6, cursor: "pointer", backdropFilter: "blur(12px)",
      transition: "all 0.15s",
      boxShadow: openenv ? "0 0 14px rgba(0,255,136,0.15)" : "none",
    }}>
      {label}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Drop zone
// ---------------------------------------------------------------------------

function useDropZone() {
  const actions = useStore((s) => s.actions);
  const [dragging, setDragging] = useState(false);
  const dragCounter = useRef(0);

  const onDragEnter = useCallback((e: DragEvent) => { e.preventDefault(); dragCounter.current++; setDragging(true); }, []);
  const onDragLeave = useCallback((e: DragEvent) => { e.preventDefault(); dragCounter.current--; if (dragCounter.current <= 0) { setDragging(false); dragCounter.current = 0; } }, []);
  const onDragOver = useCallback((e: DragEvent) => { e.preventDefault(); }, []);
  const onDrop = useCallback(async (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    dragCounter.current = 0;
    if (!e.dataTransfer?.items) return;
    actions.reset();
    actions.setDataSource("waymo-drop");
    actions.setLoadStatus("loading");
    actions.setLoadMessage("Scanning files…");
    actions.setLoadProgress(0);
    try {
      const fileMap = await scanDroppedFiles(e.dataTransfer.items);
      if (!fileMap.has("vehicle_pose") || !fileMap.has("lidar")) throw new Error("Need vehicle_pose + lidar parquet files.");
      const data = await loadWaymoFromFiles(fileMap, (step, progress) => { actions.setLoadMessage(step); actions.setLoadProgress(progress); });
      actions.setSceneData(data);
    } catch (err) {
      actions.setLoadError(err instanceof Error ? err.message : String(err));
      actions.setLoadStatus("error");
    }
  }, [actions]);

  useEffect(() => {
    window.addEventListener("dragenter", onDragEnter);
    window.addEventListener("dragleave", onDragLeave);
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragenter", onDragEnter);
      window.removeEventListener("dragleave", onDragLeave);
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("drop", onDrop);
    };
  }, [onDragEnter, onDragLeave, onDragOver, onDrop]);

  return dragging;
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

/**
 * useWaymoFrameForwarding — streams current Waymo frame stats to Python env.
 * This is the key coupling that makes Waymo constraints live in the RL env.
 */
function useWaymoFrameForwarding() {
  const currentFrame = useStore((s) => s.currentFrame);
  const rlModeActive = useRLStore((s) => s.rlModeActive);

  useEffect(() => {
    if (!rlModeActive || !currentFrame) return;
    // Compute average intensity from point attributes (every 3rd float is intensity)
    let sum = 0;
    const stride = 3;
    const count = currentFrame.pointCount;
    for (let i = 0; i < count; i++) {
      sum += currentFrame.pointAttributes[i * stride] ?? 0;
    }
    const avgIntensity = count > 0 ? sum / count : 0.5;

    wsClient.sendWaymoFrame(currentFrame.pointCount, currentFrame.boxes, avgIntensity);
  }, [currentFrame, rlModeActive]);
}

export default function App() {
  useDataLoader();
  useWaymoFrameForwarding();
  const dragging = useDropZone();
  const loadStatus = useStore((s) => s.loadStatus);
  const sceneData = useStore((s) => s.sceneData);
  const dataSource = useStore((s) => s.dataSource);
  const actions = useStore((s) => s.actions);
  const rlModeActive = useRLStore((s) => s.rlModeActive);
  const rlActions = useRLStore((s) => s.actions);

  // When launched with ?openenv=1: load mock Waymo scene (always available)
  // then activate RL overlay on top of the live city simulation.
  useEffect(() => {
    const params = new URLSearchParams(window.location.search);
    if (params.has("openenv")) {
      actions.reset();
      actions.setDataSource("mock");   // real city sim, no parquet files needed
      rlActions.setRLModeActive(true);
      wsClient.connect();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Auto-play Waymo scene as soon as data is ready in RL mode
  useEffect(() => {
    if (rlModeActive && sceneData && loadStatus === "ready") {
      actions.setFrame(0);
      actions.setPlaying(true);
    }
  }, [rlModeActive, sceneData, loadStatus, actions]);

  // Toggle RL mode → connect/disconnect WebSocket
  const handleRLToggle = useCallback(() => {
    const next = !rlModeActive;
    rlActions.setRLModeActive(next);
    if (next) wsClient.connect();
    else wsClient.disconnect();
  }, [rlModeActive, rlActions]);

  if (loadStatus === "loading" || (!sceneData && loadStatus === "idle")) return <LoadingScreen />;
  if (loadStatus === "error") return (
    <ErrorScreen onOpenENV={() => {
      actions.reset();
      actions.setDataSource("mock");
      rlActions.setRLModeActive(true);
      wsClient.connect();
    }} />
  );

  return (
    <div style={{ position: "fixed", inset: 0, overflow: "hidden", backgroundColor: colors.bgDeep }}>
      {/* Full-viewport 3D canvas — THE center of the app */}
      <Scene3D />

      {/* Floating top info bar */}
      <InfoBar />

      {/* Floating control panel (top-left) */}
      <ControlPanel />

      {/* Data source + OpenENV mode pills (top-right) */}
      <div style={{
        position: "absolute", top: 52, right: 16, zIndex: 10,
        display: "flex", gap: 4, alignItems: "center",
      }}>
        {(["waymo", "mock"] as DataSource[]).map((src) => (
          <button key={src} onClick={() => { actions.reset(); actions.setDataSource(src); }} style={{
            padding: "4px 12px", fontSize: 10, fontFamily: fonts.mono, fontWeight: dataSource === src ? 600 : 400,
            color: dataSource === src ? colors.accent : colors.textDim,
            background: dataSource === src ? "rgba(0,232,157,0.12)" : "rgba(12,15,26,0.7)",
            border: `1px solid ${dataSource === src ? "rgba(0,232,157,0.3)" : "rgba(255,255,255,0.06)"}`,
            borderRadius: 4, cursor: "pointer", backdropFilter: "blur(12px)",
            textTransform: "uppercase", letterSpacing: "0.8px", transition: "all 0.15s",
          }}>
            {src === "waymo" ? "Waymo" : "Mock"}
          </button>
        ))}

        {/* OpenENV mode toggle */}
        <div style={{ width: 1, height: 16, background: "rgba(255,255,255,0.1)" }} />
        <button onClick={handleRLToggle} style={{
          padding: "4px 12px", fontSize: 10, fontFamily: fonts.mono, fontWeight: rlModeActive ? 600 : 400,
          color: rlModeActive ? "#00FF88" : colors.textDim,
          background: rlModeActive ? "rgba(0,255,136,0.12)" : "rgba(12,15,26,0.7)",
          border: `1px solid ${rlModeActive ? "rgba(0,255,136,0.35)" : "rgba(255,255,255,0.06)"}`,
          borderRadius: 4, cursor: "pointer", backdropFilter: "blur(12px)",
          textTransform: "uppercase", letterSpacing: "0.8px", transition: "all 0.15s",
          boxShadow: rlModeActive ? "0 0 12px rgba(0,255,136,0.15)" : "none",
        }}>
          OpenENV {rlModeActive ? "●" : "○"}
        </button>
        {/* Legend — only when RL active */}
        {rlModeActive && (
          <div style={{
            display: "flex", gap: 8, alignItems: "center",
            padding: "3px 10px",
            background: "rgba(8,11,20,0.7)", backdropFilter: "blur(12px)",
            border: "1px solid rgba(255,255,255,0.06)", borderRadius: 4,
          }}>
            {([
              { color: "#00FF88", label: "EGO (Review Agent)" },
              { color: "#FF2244", label: "Incident Actor" },
              { color: "#4488FF", label: "Traffic" },
            ] as const).map(({ color, label }) => (
              <div key={label} style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <span style={{ width: 7, height: 7, borderRadius: 1, background: color, flexShrink: 0 }} />
                <span style={{ fontSize: 8, fontFamily: fonts.mono, color: colors.textDim }}>{label}</span>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* RL Stats Panel — right side, always on top when active */}
      <RLStatsPanel />

      {/* Incident submission panel — bottom right */}
      <IncidentPanel />

      {/* Timeline — always rendered so playback RAF loop keeps ticking.
          Visible in normal mode; hidden but running in RL mode so the
          Waymo city clip advances in sync with the RL episode. */}
      <div style={{ visibility: rlModeActive ? "hidden" : "visible" }}>
        <Timeline />
      </div>

      {/* Drag overlay */}
      {dragging && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 9999,
          background: "rgba(0,232,157,0.06)",
          border: `2px dashed ${colors.accent}`,
          display: "flex", alignItems: "center", justifyContent: "center",
          flexDirection: "column", gap: 8,
        }}>
          <div style={{ fontSize: 40, opacity: 0.6 }}>📂</div>
          <div style={{ fontSize: 14, fontWeight: 600, fontFamily: fonts.sans, color: colors.accent }}>
            Drop Waymo parquet files
          </div>
          <div style={{ fontSize: 11, fontFamily: fonts.mono, color: colors.textDim }}>
            vehicle_pose + lidar + lidar_box + lidar_calibration
          </div>
        </div>
      )}
    </div>
  );
}
