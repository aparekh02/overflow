/**
 * DashboardPage — Camera-grid multi-sim.
 *
 * Layout:
 *   Left area: Ground Truth (3D) + up to 3 active counterfactual sims (3D)
 *   Right sidebar: scrollable list of finished runs, clickable to open detail
 *
 * All sims auto-play. Every 10s, when a slot opens (run finishes after 20 steps),
 * new counterfactuals spawn to fill back up to 3 active.
 *
 * Each counterfactual tile gets a full interactive 3D Scene3D, identical to
 * the ground truth view.
 */

import { useEffect, useRef, useState, useMemo } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import {
  Play,
  Pause,
  Maximize2,
  X,
  Zap,
  TrendingUp,
  TrendingDown,
  ChevronRight,
} from "lucide-react";
import Scene3D from "../components/Scene3D";
import Timeline from "../components/Timeline";
import Badge from "../components/ui/Badge";
import { useStore } from "../store";
import { useSimManager, startSimLoop, stopSimLoop } from "../lib/simManager";
import type { CounterfactualRun } from "../lib/simTypes";
import { colors, fonts, typeScale, spacing, glass, radius } from "../theme";

export default function DashboardPage() {
  const navigate = useNavigate();
  const runs = useSimManager((s) => s.runs);
  const mainState = useSimManager((s) => s.mainState);
  const totalSpawned = useSimManager((s) => s.totalSpawned);
  const [focusedRun, setFocusedRun] = useState<CounterfactualRun | null>(null);

  const isPlaying = useStore((s) => s.isPlaying);
  const actions = useStore((s) => s.actions);

  // Start sim loop (auto-plays everything)
  useEffect(() => {
    startSimLoop();
    toast("Dashboard active — all simulations running");
    return () => stopSimLoop();
  }, []);

  // Toast on new spawns
  const prevSpawnedRef = useRef(totalSpawned);
  useEffect(() => {
    if (totalSpawned > prevSpawnedRef.current && prevSpawnedRef.current > 0) {
      const diff = totalSpawned - prevSpawnedRef.current;
      toast(`Spawned ${diff} new counterfactual run${diff > 1 ? "s" : ""}`);
    }
    prevSpawnedRef.current = totalSpawned;
  }, [totalSpawned]);

  const activeRuns = runs.filter((r) => r.status === "running");
  const finishedRuns = runs
    .filter((r) => r.status === "finished" || r.status === "replaced")
    .sort((a, b) => b.createdAt - a.createdAt);

  const avgReward = activeRuns.length > 0
    ? activeRuns.reduce((s, r) => s + r.metrics.avgReward, 0) / activeRuns.length
    : 0;

  return (
    <div style={{
      height: "100%",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
      position: "relative",
    }}>
      {/* Header */}
      <div style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        padding: `${spacing.sm}px ${spacing.lg}px`,
        flexShrink: 0,
        borderBottom: `1px solid ${colors.border}`,
      }}>
        <div style={{ display: "flex", alignItems: "center", gap: spacing.md }}>
          <h1 style={{ ...typeScale.h2, color: colors.textPrimary, margin: 0 }}>
            Multi-Sim Dashboard
          </h1>
          <Badge variant="success" dot>
            {activeRuns.length} / 3 active
          </Badge>
        </div>
        <div style={{ display: "flex", gap: spacing.sm, alignItems: "center" }}>
          <StatMini label="Spawned" value={totalSpawned} />
          <StatMini label="Finished" value={finishedRuns.length} />
          <StatMini label="Avg R" value={avgReward.toFixed(2)} color={avgReward > 0.3 ? colors.success : colors.textSecondary} />
        </div>
      </div>

      {/* Main content area */}
      <div style={{ flex: 1, display: "flex", overflow: "hidden", paddingBottom: 68 }}>

        {/* Left: sim grid */}
        <div style={{
          flex: 1,
          display: "grid",
          gridTemplateColumns: "1fr 1fr",
          gridTemplateRows: "1fr 1fr",
          gap: spacing.sm,
          padding: spacing.sm,
          overflow: "hidden",
        }}>
          {/* Ground truth tile — top-left */}
          <div style={{
            background: colors.bgCard,
            border: `1px solid ${colors.borderAccent}`,
            borderRadius: radius.lg,
            overflow: "hidden",
            display: "flex",
            flexDirection: "column",
            position: "relative",
          }}>
            <TileHeader
              title="Ground Truth"
              badge={<Badge variant="success" dot>LIVE</Badge>}
              right={
                <div style={{ display: "flex", gap: 4 }}>
                  <button onClick={() => actions.togglePlay()} style={tileBtn}>
                    {isPlaying ? <Pause size={12} /> : <Play size={12} />}
                  </button>
                  <button onClick={() => navigate("/sim")} style={tileBtn}>
                    <Maximize2 size={12} />
                  </button>
                </div>
              }
            />
            <div style={{ flex: 1, position: "relative", background: colors.bgDeep, minHeight: 0 }}>
              <Scene3D />
            </div>
            <div style={{
              display: "flex", alignItems: "center",
              padding: `4px ${spacing.md}px`, gap: spacing.md,
              borderTop: `1px solid ${colors.border}`, flexShrink: 0,
            }}>
              <TileStat label="Reward" value={mainState.cumulativeReward.toFixed(2)} color={colors.accent} />
              <TileStat label="Action" value={mainState.lastAction?.action.replace(/_/g, " ") || "waiting"} />
              <TileStat label="Nearest" value={`${mainState.nearestObjectDist.toFixed(0)}m`} />
            </div>
          </div>

          {/* 3 counterfactual slots */}
          {[0, 1, 2].map((slot) => {
            const run = activeRuns[slot];
            if (!run) {
              return (
                <div key={`empty-${slot}`} style={{
                  background: colors.bgCard,
                  border: `1px dashed ${colors.border}`,
                  borderRadius: radius.lg,
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  flexDirection: "column",
                  gap: spacing.sm,
                }}>
                  <Zap size={20} color={colors.textMuted} />
                  <span style={{ ...typeScale.small, color: colors.textMuted }}>
                    Waiting for spawn...
                  </span>
                </div>
              );
            }
            return (
              <CounterfactualTile
                key={run.id}
                run={run}
                tileIndex={slot}
                onFocus={() => setFocusedRun(run)}
              />
            );
          })}
        </div>

        {/* Right sidebar: finished runs */}
        <div style={{
          width: 280,
          flexShrink: 0,
          borderLeft: `1px solid ${colors.border}`,
          display: "flex",
          flexDirection: "column",
          background: colors.bgBase,
        }}>
          <div style={{
            padding: `${spacing.md}px ${spacing.md}px`,
            borderBottom: `1px solid ${colors.border}`,
            flexShrink: 0,
          }}>
            <div style={{ ...typeScale.h3, color: colors.textPrimary }}>
              Completed Runs
            </div>
            <span style={{ ...typeScale.small, color: colors.textDim }}>
              {finishedRuns.length} run{finishedRuns.length !== 1 ? "s" : ""}
            </span>
          </div>

          <div style={{ flex: 1, overflow: "auto", padding: spacing.xs }}>
            {finishedRuns.length === 0 && (
              <div style={{
                textAlign: "center",
                padding: spacing.xl,
                color: colors.textMuted,
                ...typeScale.small,
              }}>
                Runs will appear here when they finish (after 20 steps)
              </div>
            )}
            {finishedRuns.map((run) => (
              <FinishedRunCard
                key={run.id}
                run={run}
                onClick={() => setFocusedRun(run)}
              />
            ))}
          </div>
        </div>
      </div>

      {/* Timeline — drives live playback */}
      <Timeline />

      {/* Focused run modal */}
      {focusedRun && (
        <RunDetailModal run={focusedRun} onClose={() => setFocusedRun(null)} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tile header
// ---------------------------------------------------------------------------

function TileHeader({ title, badge, right }: {
  title: string;
  badge?: React.ReactNode;
  right?: React.ReactNode;
}) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: `4px ${spacing.md}px`,
      borderBottom: `1px solid ${colors.border}`,
      flexShrink: 0,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <span style={{ ...typeScale.small, fontWeight: 600, color: colors.textPrimary }}>{title}</span>
        {badge}
      </div>
      {right}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Per-tile colours — each counterfactual gets a distinct hue
// ---------------------------------------------------------------------------

const TILE_COLORS = ["#FF6B6B", "#4ECDC4", "#FFD93D"];

// ---------------------------------------------------------------------------
// Action → visible world offset & trail computation
// ---------------------------------------------------------------------------

/**
 * Compute a visually meaningful world offset from the run's action stream.
 * Each action maps to a per-step displacement in ego-relative metres.
 * These are intentionally large enough to SEE in the 3D viewport.
 */
function computeActionOffset(run: CounterfactualRun): SceneOffset {
  let dx = 0;
  let dy = 0;
  for (const a of run.actionStream) {
    switch (a.action) {
      case "merge_left":   dy += 2.0;  break; // roughly one lane
      case "merge_right":  dy -= 2.0;  break;
      case "nudge_left":   dy += 0.7;  break;
      case "nudge_right":  dy -= 0.7;  break;
      case "brake_mild":   dx -= 1.5;  break; // fell behind ground truth
      case "brake_hard":   dx -= 3.5;  break;
      case "accelerate":   dx += 2.5;  break; // pulled ahead
      case "yield":        dx -= 2.0;  break;
      case "keep_lane":                break; // no deviation
    }
  }
  // Slight heading deviation proportional to lateral shift
  const dYaw = Math.atan2(dy, 30) * 0.35;
  return { dx, dy, dYaw };
}

/**
 * Build a 3D trail from the accumulated action offsets at each step.
 * Points are in ego-relative coords so they render around the ego.
 */
function buildTrail(run: CounterfactualRun): [number, number, number][] {
  const trail: [number, number, number][] = [[0, 0, 0.3]];
  let cx = 0;
  let cy = 0;
  for (const a of run.actionStream) {
    switch (a.action) {
      case "merge_left":   cy += 2.0;  break;
      case "merge_right":  cy -= 2.0;  break;
      case "nudge_left":   cy += 0.7;  break;
      case "nudge_right":  cy -= 0.7;  break;
      case "brake_mild":   cx -= 1.5;  break;
      case "brake_hard":   cx -= 3.5;  break;
      case "accelerate":   cx += 2.5;  break;
      case "yield":        cx -= 2.0;  break;
      case "keep_lane":                break;
    }
    trail.push([cx, cy, 0.3]);
  }
  return trail;
}

// ---------------------------------------------------------------------------
// Counterfactual tile — shows divergent 3D view + action overlay
// ---------------------------------------------------------------------------

function CounterfactualTile({
  run,
  onFocus,
  tileIndex,
}: {
  run: CounterfactualRun;
  onFocus: () => void;
  tileIndex: number;
}) {
  const delta = run.metrics.deltaVsMain;
  const isPositive = delta > 0;
  const lastAction = run.actionStream[run.actionStream.length - 1];
  const tileColor = TILE_COLORS[tileIndex % TILE_COLORS.length];

  // Compute offset & trail from the action stream (not physics positions)
  const offset = useMemo(() => {
    if (run.actionStream.length < 1) return undefined;
    const o = computeActionOffset(run);
    if (Math.abs(o.dx) < 0.01 && Math.abs(o.dy) < 0.01) return undefined;
    return o;
  }, [run.actionStream]);

  const trail = useMemo(
    () => (run.actionStream.length >= 1 ? buildTrail(run) : undefined),
    [run.actionStream],
  );

  // Latest action label color
  const actionColor = lastAction
    ? lastAction.action.includes("brake") ? colors.warning
    : lastAction.action.includes("merge") || lastAction.action.includes("nudge") ? colors.info
    : lastAction.action === "accelerate" ? colors.success
    : colors.accent
    : colors.accent;

  return (
    <div style={{
      background: colors.bgCard,
      border: `1px solid ${isPositive ? `${tileColor}33` : colors.border}`,
      borderRadius: radius.lg,
      overflow: "hidden",
      display: "flex",
      flexDirection: "column",
    }}>
      <TileHeader
        title={run.label}
        badge={<Badge variant="success" dot>RUNNING</Badge>}
        right={
          <button onClick={onFocus} style={tileBtn}>
            <Maximize2 size={11} />
          </button>
        }
      />

      {/* 3D scene — offset to reflect counterfactual ego divergence */}
      <div style={{ flex: 1, position: "relative", background: colors.bgDeep, minHeight: 0 }}>
        <Scene3D offset={offset} trail={trail} trailColor={tileColor} />

        {/* Colored border highlight so tiles are visually distinct */}
        <div style={{
          position: "absolute", inset: 0,
          border: `2px solid ${tileColor}30`,
          borderRadius: 0,
          pointerEvents: "none",
        }} />

        {/* Action + reward overlay */}
        {lastAction && (
          <div style={{
            position: "absolute", bottom: 6, left: 6, right: 6,
            display: "flex", flexDirection: "column", gap: 3,
            pointerEvents: "none",
          }}>
            <div style={{ display: "flex", gap: 6, alignItems: "center", flexWrap: "wrap" }}>
              <span style={{
                ...typeScale.mono, fontSize: 10, color: actionColor,
                background: "rgba(10,13,22,0.92)", padding: "2px 8px", borderRadius: 4,
                fontWeight: 600, letterSpacing: "0.3px",
              }}>
                ▸ {lastAction.action.replace(/_/g, " ")}
              </span>
              <span style={{
                ...typeScale.mono, fontSize: 10,
                color: lastAction.reward >= 0.5 ? colors.success : lastAction.reward >= 0 ? colors.textSecondary : colors.error,
                background: "rgba(10,13,22,0.92)", padding: "2px 6px", borderRadius: 4,
              }}>
                R: {lastAction.reward >= 0 ? "+" : ""}{lastAction.reward.toFixed(2)}
              </span>
              {offset && (
                <span style={{
                  ...typeScale.mono, fontSize: 8, color: tileColor,
                  background: "rgba(10,13,22,0.88)", padding: "1px 6px", borderRadius: 3,
                }}>
                  Δ {offset.dx > 0 ? "+" : ""}{offset.dx.toFixed(1)}m fwd, {offset.dy > 0 ? "+" : ""}{offset.dy.toFixed(1)}m lat
                </span>
              )}
            </div>
          </div>
        )}
      </div>

      {/* Metrics strip */}
      <div style={{
        display: "flex", alignItems: "center",
        padding: `4px ${spacing.md}px`, gap: spacing.sm,
        borderTop: `1px solid ${colors.border}`, flexShrink: 0,
      }}>
        <TileStat label="Reward" value={run.metrics.cumulativeReward.toFixed(2)}
          color={run.metrics.cumulativeReward > 0 ? colors.accent : colors.error} />
        <TileStat label="Delta" value={`${isPositive ? "+" : ""}${delta.toFixed(2)}`}
          color={isPositive ? colors.success : colors.error}
          icon={isPositive ? TrendingUp : TrendingDown} />
        <TileStat label="Steps" value={`${run.actionStream.length}`} />
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Finished run card (sidebar)
// ---------------------------------------------------------------------------

function FinishedRunCard({ run, onClick }: { run: CounterfactualRun; onClick: () => void }) {
  const delta = run.metrics.deltaVsMain;
  const isPositive = delta > 0;

  return (
    <button
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: spacing.sm,
        width: "100%",
        padding: `${spacing.sm}px ${spacing.md}px`,
        borderRadius: radius.md,
        background: "transparent",
        border: "none",
        cursor: "pointer",
        textAlign: "left",
        fontFamily: fonts.sans,
        color: colors.textPrimary,
        marginBottom: 2,
        transition: "background 0.1s",
      }}
      onMouseEnter={(e) => (e.currentTarget.style.background = "rgba(255,255,255,0.03)")}
      onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}
    >
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6, marginBottom: 2 }}>
          <span style={{ ...typeScale.small, fontWeight: 600, color: colors.textPrimary }}>
            {run.label}
          </span>
          <Badge variant={run.status === "finished" ? "info" : "default"}>
            {run.actionStream.length} steps
          </Badge>
        </div>
        <div style={{ display: "flex", gap: spacing.sm }}>
          <span style={{
            ...typeScale.mono, fontSize: 10,
            color: run.metrics.cumulativeReward > 0 ? colors.accent : colors.error,
          }}>
            R: {run.metrics.cumulativeReward.toFixed(2)}
          </span>
          <span style={{
            ...typeScale.mono, fontSize: 10,
            color: isPositive ? colors.success : colors.error,
          }}>
            {isPositive ? "+" : ""}{delta.toFixed(2)}
          </span>
          <span style={{ ...typeScale.mono, fontSize: 9, color: colors.textDim }}>
            {new Date(run.createdAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}
          </span>
        </div>
      </div>
      <ChevronRight size={14} color={colors.textDim} />
    </button>
  );
}

// ---------------------------------------------------------------------------
// Run detail modal
// ---------------------------------------------------------------------------

function RunDetailModal({ run, onClose }: { run: CounterfactualRun; onClose: () => void }) {
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed", inset: 0,
        background: "rgba(0,0,0,0.6)",
        display: "flex", alignItems: "center", justifyContent: "center",
        zIndex: 200,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: colors.bgCard,
          border: `1px solid ${colors.border}`,
          borderRadius: radius.xl,
          padding: spacing.xl,
          maxWidth: 600, width: "90%",
          maxHeight: "80vh", overflow: "auto",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: spacing.lg }}>
          <div>
            <h2 style={{ ...typeScale.h2, color: colors.textPrimary, margin: 0 }}>{run.label}</h2>
            <span style={{ ...typeScale.mono, color: colors.textDim }}>{run.branchId}</span>
          </div>
          <button onClick={onClose} style={{ cursor: "pointer", padding: 4, border: "none", background: "none" }}>
            <X size={18} color={colors.textDim} />
          </button>
        </div>

        <div style={{ display: "flex", gap: spacing.sm, marginBottom: spacing.lg }}>
          <Badge variant={run.status === "running" ? "success" : "info"} dot>
            {run.status}
          </Badge>
          <span style={{ ...typeScale.small, color: colors.textDim }}>
            Started at frame {run.startFrameIndex}
          </span>
          <span style={{ ...typeScale.small, color: colors.textDim }}>
            {run.actionStream.length} steps
          </span>
        </div>

        <div style={{
          display: "grid", gridTemplateColumns: "repeat(4, 1fr)",
          gap: spacing.sm, marginBottom: spacing.lg,
        }}>
          <MetricBox label="Cumulative Reward" value={run.metrics.cumulativeReward.toFixed(3)} color={colors.accent} />
          <MetricBox label="Avg Reward" value={run.metrics.avgReward.toFixed(3)} color={colors.info} />
          <MetricBox label="Delta vs Main" value={`${run.metrics.deltaVsMain >= 0 ? "+" : ""}${run.metrics.deltaVsMain.toFixed(3)}`} color={run.metrics.deltaVsMain >= 0 ? colors.success : colors.error} />
          <MetricBox label="Min TTC" value={run.metrics.minTTC < 100 ? `${run.metrics.minTTC.toFixed(1)}s` : "safe"} color={run.metrics.minTTC < 3 ? colors.warning : colors.textSecondary} />
        </div>

        <div>
          <div style={{ ...typeScale.caption, color: colors.textDim, marginBottom: spacing.sm }}>
            Action Stream
          </div>
          <div style={{ maxHeight: 200, overflow: "auto", display: "flex", flexDirection: "column", gap: 2 }}>
            {run.actionStream.map((a, i) => (
              <div key={i} style={{
                display: "flex", alignItems: "center", justifyContent: "space-between",
                padding: "4px 8px", borderRadius: 4,
                background: i % 2 === 0 ? "rgba(255,255,255,0.02)" : "transparent",
              }}>
                <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                  <span style={{ ...typeScale.mono, color: colors.textDim, width: 24 }}>{i + 1}</span>
                  <span style={{ ...typeScale.mono, color: colors.accent }}>{a.action}</span>
                </div>
                <span style={{
                  ...typeScale.mono,
                  color: a.reward >= 0.5 ? colors.success : a.reward >= 0 ? colors.textSecondary : colors.error,
                }}>
                  {a.reward >= 0 ? "+" : ""}{a.reward.toFixed(3)}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function TileStat({
  label, value, color = colors.textSecondary, icon: Icon,
}: {
  label: string; value: string; color?: string; icon?: typeof TrendingUp;
}) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 3 }}>
      {Icon && <Icon size={10} color={color} />}
      <span style={{ ...typeScale.caption, color: colors.textDim, fontSize: 8 }}>{label}</span>
      <span style={{ ...typeScale.mono, color, fontSize: 10, fontWeight: 500 }}>{value}</span>
    </div>
  );
}

function StatMini({ label, value, color = colors.textSecondary }: { label: string; value: string | number; color?: string }) {
  return (
    <div style={{
      ...glass, padding: "3px 8px",
      display: "flex", alignItems: "center", gap: 5,
    }}>
      <span style={{ ...typeScale.caption, color: colors.textDim, fontSize: 8 }}>{label}</span>
      <span style={{ ...typeScale.mono, color, fontSize: 11, fontWeight: 600 }}>{value}</span>
    </div>
  );
}

function MetricBox({ label, value, color }: { label: string; value: string; color: string }) {
  return (
    <div style={{
      background: "rgba(255,255,255,0.02)",
      borderRadius: radius.md, padding: spacing.sm,
    }}>
      <div style={{ ...typeScale.caption, color: colors.textDim, fontSize: 8, marginBottom: 4 }}>{label}</div>
      <div style={{ fontSize: 16, fontWeight: 700, color, fontFamily: fonts.mono }}>{value}</div>
    </div>
  );
}

const tileBtn: React.CSSProperties = {
  width: 24, height: 24,
  display: "flex", alignItems: "center", justifyContent: "center",
  borderRadius: 4,
  background: "rgba(255,255,255,0.04)",
  border: `1px solid ${colors.border}`,
  color: colors.textSecondary,
  cursor: "pointer", padding: 0,
};
