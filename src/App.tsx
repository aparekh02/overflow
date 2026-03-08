/**
 * App — Router + data loading + app shell.
 */

import { useEffect, useState, useCallback, useRef } from "react";
import { Routes, Route, Navigate } from "react-router-dom";
import { Toaster, toast } from "sonner";
import AppShell from "./components/ui/AppShell";
import SimPage from "./pages/SimPage";
import DashboardPage from "./pages/DashboardPage";
import GraphPage from "./pages/GraphPage";
import AnalyticsPage from "./pages/AnalyticsPage";
import { useStore } from "./store";
import type { DataSource } from "./store";
import { generateSceneData } from "./mockData";
import { generateTrajectoryMoments } from "./utils/trajectoryData";
import {
  loadWaymoFromUrls,
  loadWaymoFromFiles,
  scanDroppedFiles,
  type WaymoLoadResult,
} from "./utils/waymoLoader";
import { colors, fonts, typeScale } from "./theme";

// ---------------------------------------------------------------------------
// Auto-detect waymo data layout
// ---------------------------------------------------------------------------

async function detectWaymoLayout(
  basePath: string,
  overrideSegment?: string | null,
): Promise<{ basePath: string; segmentName?: string }> {
  if (overrideSegment) {
    return { basePath, segmentName: overrideSegment };
  }
  try {
    const resp = await fetch(`${basePath}/manifest.json`);
    if (resp.ok) {
      const manifest = await resp.json();
      const segId = manifest.segment;
      if (segId) return { basePath, segmentName: segId };
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
      actions.setLoadMessage(`Generating "${mockScenario}" scenario`);
      actions.setLoadProgress(0.5);
      setTimeout(() => {
        try {
          const sceneData = generateSceneData(mockScenario);
          actions.setSceneData(sceneData);
          const moments = generateTrajectoryMoments(sceneData);
          actions.setTrajectoryMoments(moments);
        } catch (e) {
          actions.setLoadError(e instanceof Error ? e.message : String(e));
          actions.setLoadStatus("error");
        }
      }, 0);
    } else if (dataSource === "waymo") {
      actions.setLoadStatus("loading");
      actions.setLoadMessage("Detecting data layout");
      actions.setLoadProgress(0);

      detectWaymoLayout("/waymo_data", waymoSegment)
        .then(({ basePath, segmentName }) => {
          actions.setLoadMessage("Opening Parquet files");
          return loadWaymoFromUrls(basePath, (step, progress) => {
            actions.setLoadMessage(step);
            actions.setLoadProgress(progress);
          }, segmentName);
        })
        .then((data: WaymoLoadResult) => {
          actions.setSceneData(data);
          const moments = generateTrajectoryMoments(data);
          actions.setTrajectoryMoments(moments);
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
        width: 28, height: 28,
        border: `2px solid ${colors.accent}`,
        borderTopColor: "transparent",
        borderRadius: "50%",
        animation: "spin 0.7s linear infinite",
      }} />
      <div style={{
        width: 200, height: 2,
        backgroundColor: "rgba(255,255,255,0.06)",
        borderRadius: 2, overflow: "hidden",
      }}>
        <div style={{
          height: "100%",
          width: `${Math.round(loadProgress * 100)}%`,
          background: colors.accent,
          borderRadius: 2, transition: "width 0.3s ease-out",
        }} />
      </div>
      <span style={{ color: colors.textDim, ...typeScale.mono }}>
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
      <div style={{ maxWidth: 380, textAlign: "center", lineHeight: 1.5, fontSize: 13, color: colors.error }}>
        {loadError || "Unknown error"}
      </div>
      <div style={{ display: "flex", gap: 10, marginTop: 8 }}>
        <button onClick={() => { actions.reset(); actions.setDataSource("waymo"); }} style={btnStyle(true)}>
          Retry Waymo
        </button>
        <button onClick={() => { actions.reset(); actions.setDataSource("mock"); }} style={btnStyle(false)}>
          Use Mock
        </button>
      </div>
    </div>
  );
}

function btnStyle(accent: boolean): React.CSSProperties {
  return {
    padding: "7px 18px", fontSize: 12, fontFamily: fonts.sans, fontWeight: 500,
    background: accent ? "rgba(0,232,157,0.08)" : "rgba(255,255,255,0.04)",
    color: accent ? colors.accent : colors.textSecondary,
    border: `1px solid ${accent ? colors.borderAccent : colors.border}`,
    borderRadius: 6, cursor: "pointer",
  };
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
    actions.setLoadMessage("Scanning files");
    actions.setLoadProgress(0);
    try {
      const fileMap = await scanDroppedFiles(e.dataTransfer.items);
      if (!fileMap.has("vehicle_pose") || !fileMap.has("lidar")) throw new Error("Need vehicle_pose + lidar parquet files.");
      const data = await loadWaymoFromFiles(fileMap, (step, progress) => { actions.setLoadMessage(step); actions.setLoadProgress(progress); });
      actions.setSceneData(data);
      const moments = generateTrajectoryMoments(data);
      actions.setTrajectoryMoments(moments);
      toast.success("Waymo data loaded successfully");
    } catch (err) {
      actions.setLoadError(err instanceof Error ? err.message : String(err));
      actions.setLoadStatus("error");
      toast.error("Failed to load Waymo data");
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

  if (loadStatus === "loading" || (!sceneData && loadStatus === "idle")) return <LoadingScreen />;
  if (loadStatus === "error") return <ErrorScreen />;

  return (
    <>
      <Toaster
        position="bottom-right"
        theme="dark"
        toastOptions={{
          style: {
            background: colors.bgCard,
            border: `1px solid ${colors.border}`,
            color: colors.textPrimary,
            fontFamily: fonts.sans,
            fontSize: 12,
          },
        }}
      />

      <AppShell>
        <Routes>
          <Route path="/" element={<Navigate to="/sim" replace />} />
          <Route path="/sim" element={<SimPage />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/graph" element={<GraphPage />} />
          <Route path="/analytics" element={<AnalyticsPage />} />
        </Routes>
      </AppShell>

      {/* Drag overlay */}
      {dragging && (
        <div style={{
          position: "fixed", inset: 0, zIndex: 9999,
          background: "rgba(0,232,157,0.04)",
          border: `2px dashed ${colors.accent}`,
          display: "flex", alignItems: "center", justifyContent: "center",
          flexDirection: "column", gap: 8,
        }}>
          <div style={{ fontSize: 14, fontWeight: 600, fontFamily: fonts.sans, color: colors.accent }}>
            Drop Waymo parquet files
          </div>
          <div style={{ fontSize: 11, fontFamily: fonts.mono, color: colors.textDim }}>
            vehicle_pose + lidar + lidar_box + lidar_calibration
          </div>
        </div>
      )}
    </>
  );
}
