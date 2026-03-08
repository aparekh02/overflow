/**
 * App — Full-viewport 3D perception viewer.
 * The 3D canvas IS the app. All UI floats on top as glass overlays.
 */

import { useEffect, useState, useCallback, useRef } from "react";
import Scene3D from "./components/Scene3D";
import ControlPanel from "./components/ControlPanel";
import Timeline from "./components/Timeline";
import InfoBar from "./components/InfoBar";
import IncidentAlert from "./components/IncidentAlert";
import ScenarioChat from "./components/ScenarioChat";
import CameraViews from "./components/CameraViews";
import KnowledgeGraph from "./components/KnowledgeGraph";
import TicketConsole from "./components/TicketConsole";
import DataBrowser from "./components/DataBrowser";
import { useStore } from "./store";
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
  overrideSegment?: string | null,
): Promise<{ basePath: string; segmentName?: string }> {
  // Use explicit segment from DataBrowser / store
  if (overrideSegment) {
    console.log(`[waymo] Explicit segment: ${overrideSegment}`);
    return { basePath, segmentName: overrideSegment };
  }

  try {
    const resp = await fetch(`${basePath}/manifest.json`);
    if (resp.ok) {
      const manifest = await resp.json();
      const segId = manifest.segment;
      if (segId) {
        console.log(`[waymo] Manifest found — segment: ${segId}`);
        return { basePath, segmentName: segId };
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
  const mockScenario = useStore((s) => s.mockScenario);
  const waymoSegment = useStore((s) => s.waymoSegment);
  const actions = useStore((s) => s.actions);

  useEffect(() => {
    if (loadStatus !== "idle") return;
    if (dataSource === "waymo-drop") return;

    if (dataSource === "mock") {
      actions.setLoadStatus("loading");
      actions.setLoadMessage(`Generating "${mockScenario}" scenario…`);
      actions.setLoadProgress(0.5);
      setTimeout(() => {
        try {
          actions.setSceneData(generateSceneData(mockScenario));
        } catch (e) {
          actions.setLoadError(e instanceof Error ? e.message : String(e));
          actions.setLoadStatus("error");
        }
      }, 0);
    } else if (dataSource === "waymo") {
      actions.setLoadStatus("loading");
      actions.setLoadMessage("Detecting data layout…");
      actions.setLoadProgress(0);

      detectWaymoLayout("/waymo_data", waymoSegment)
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
  }, [dataSource, loadStatus, mockScenario, waymoSegment, actions]);
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

function ErrorScreen() {
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
      <div style={{ display: "flex", gap: 10, marginTop: 8 }}>
        <GlassButton label="Retry Waymo" onClick={() => { actions.reset(); actions.setDataSource("waymo"); }} accent />
        <GlassButton label="Use Mock" onClick={() => { actions.reset(); actions.setDataSource("mock"); }} />
      </div>
    </div>
  );
}

function GlassButton({ label, onClick, accent }: { label: string; onClick: () => void; accent?: boolean }) {
  return (
    <button onClick={onClick} style={{
      padding: "7px 18px", fontSize: 12, fontFamily: fonts.sans, fontWeight: 500,
      background: accent ? "rgba(0,232,157,0.1)" : "rgba(255,255,255,0.04)",
      color: accent ? colors.accent : colors.textSecondary,
      border: `1px solid ${accent ? "rgba(0,232,157,0.3)" : colors.border}`,
      borderRadius: 6, cursor: "pointer", backdropFilter: "blur(12px)",
      transition: "all 0.15s",
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

export default function App() {
  useDataLoader();
  const dragging = useDropZone();
  const loadStatus = useStore((s) => s.loadStatus);
  const sceneData = useStore((s) => s.sceneData);
  const dataSource = useStore((s) => s.dataSource);
  const actions = useStore((s) => s.actions);

  if (loadStatus === "loading" || (!sceneData && loadStatus === "idle")) return <LoadingScreen />;
  if (loadStatus === "error") return <ErrorScreen />;

  return (
    <div style={{ position: "fixed", inset: 0, overflow: "hidden", backgroundColor: colors.bgDeep }}>
      {/* Full-viewport 3D canvas — THE center of the app */}
      <Scene3D />

      {/* Floating top info bar */}
      <InfoBar />
      <IncidentAlert />

      {/* Floating control panel (top-left) */}
      <ControlPanel />

      {/* Data source pills (top-right, above knowledge graph) */}
      <div style={{
        position: "absolute", top: 52, right: 8, zIndex: 12,
        display: "flex", gap: 4,
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
      </div>

      {/* Camera views (bottom-left) */}
      <CameraViews />

      {/* Knowledge graph (top-right) */}
      <KnowledgeGraph />

      {/* Ticket console (bottom-right) */}
      <TicketConsole />

      {/* Floating timeline (bottom) */}
      <Timeline />

      {/* Data browser sidebar */}
      <DataBrowser />

      {/* AI Scenario Chat */}
      <ScenarioChat />

      {/* Dataset browser drawer */}
      <DataBrowser />

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
