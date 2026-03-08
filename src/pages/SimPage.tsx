/**
 * SimPage — Main 3D simulator view.
 * Keeps the Three.js renderer intact; replaces surrounding UI with clean panels.
 */

import { useEffect, useState, useCallback } from "react";
import { useNavigate } from "react-router-dom";
import { toast } from "sonner";
import {
  LayoutGrid,
  GitBranch,
  Settings2,
  ChevronDown,
  Zap,
  Clock,
  ArrowRight,
  X,
} from "lucide-react";
import Scene3D from "../components/Scene3D";
import Timeline from "../components/Timeline";
import Badge from "../components/ui/Badge";
import { useStore } from "../store";
import { useSimManager } from "../lib/simManager";
import { getOpenEnvMode } from "../lib/openenvClient";
import { colors, fonts, typeScale, spacing, glass, radius } from "../theme";
import type { MockScenario } from "../mockData";
import { SCENARIO_INFO } from "../mockData";

const SCENARIOS: { id: MockScenario; label: string }[] = [
  { id: "normal", label: "Normal Traffic" },
  { id: "near_miss", label: "Near Miss" },
  { id: "rear_end", label: "Rear End" },
  { id: "jaywalker", label: "Jaywalker" },
  { id: "red_light_runner", label: "Red Light Runner" },
  { id: "swerving_vehicle", label: "Swerving Vehicle" },
];

export default function SimPage() {
  const navigate = useNavigate();
  const [showAutonomy, setShowAutonomy] = useState(true);
  const [showExplain, setShowExplain] = useState(false);
  const [showSettings, setShowSettings] = useState(false);

  const dataSource = useStore((s) => s.dataSource);
  const mockScenario = useStore((s) => s.mockScenario);
  const actions = useStore((s) => s.actions);
  const colormapMode = useStore((s) => s.colormapMode);
  const boxMode = useStore((s) => s.boxMode);
  const showGrid = useStore((s) => s.showGrid);
  const pointOpacity = useStore((s) => s.pointOpacity);
  const currentFrameIndex = useStore((s) => s.currentFrameIndex);
  const totalFrames = useStore((s) => s.totalFrames);
  const fps = useStore((s) => s.sceneData?.fps ?? 10);

  const mainState = useSimManager((s) => s.mainState);
  const openenvLastUpdate = useSimManager((s) => s.openenvLastUpdate);
  const simActions = useSimManager((s) => s.actions);

  // Periodic OpenEnv update — syncs real frame data from store automatically
  useEffect(() => {
    simActions.syncFromStore();
    simActions.updateMainFromOpenEnv();
    const interval = setInterval(() => {
      simActions.syncFromStore();
      simActions.updateMainFromOpenEnv();
    }, 3000);
    return () => clearInterval(interval);
  }, [simActions]);

  // Notify on OpenEnv updates
  useEffect(() => {
    if (mainState.lastAction) {
      toast(`OpenEnv: ${mainState.lastAction.action} (reward: ${mainState.lastAction.reward.toFixed(2)})`, {
        duration: 2000,
      });
    }
  }, [openenvLastUpdate]);

  const switchScenario = useCallback((id: MockScenario) => {
    actions.reset();
    actions.setDataSource("mock");
    actions.setMockScenario(id);
  }, [actions]);

  const currentTime = (currentFrameIndex / fps).toFixed(1);

  return (
    <div style={{ position: "absolute", inset: 0, overflow: "hidden" }}>
      {/* 3D Canvas — THE renderer, untouched */}
      <Scene3D />

      {/* ── Top Bar ── */}
      <div style={{
        position: "absolute",
        top: 48,
        left: spacing.md,
        right: spacing.md,
        zIndex: 20,
        display: "flex",
        alignItems: "center",
        gap: spacing.sm,
        pointerEvents: "none",
      }}>
        {/* Scenario selector */}
        <div style={{ pointerEvents: "auto", position: "relative" }}>
          <ScenarioSelector
            current={mockScenario}
            dataSource={dataSource}
            onSelect={switchScenario}
          />
        </div>

        {/* Timestamp */}
        <div style={{
          ...glass,
          padding: "6px 12px",
          display: "flex",
          alignItems: "center",
          gap: 6,
          pointerEvents: "auto",
        }}>
          <Clock size={12} color={colors.textDim} />
          <span style={{ ...typeScale.mono, color: colors.textSecondary }}>
            {currentTime}s
          </span>
          <span style={{ ...typeScale.mono, color: colors.textDim }}>
            F{currentFrameIndex + 1}/{totalFrames}
          </span>
        </div>

        <div style={{ flex: 1 }} />

        {/* Quick nav buttons */}
        <NavButton icon={LayoutGrid} label="Dashboard" onClick={() => navigate("/dashboard")} />
        <NavButton icon={GitBranch} label="Graph" onClick={() => navigate("/graph")} />

        {/* Settings toggle */}
        <div style={{ pointerEvents: "auto" }}>
          <button
            onClick={() => setShowSettings(!showSettings)}
            style={{
              ...glass,
              padding: "6px 10px",
              display: "flex",
              alignItems: "center",
              gap: 5,
              cursor: "pointer",
              border: `1px solid ${showSettings ? colors.borderAccent : colors.border}`,
            }}
          >
            <Settings2 size={13} color={showSettings ? colors.accent : colors.textSecondary} />
            <span style={{ ...typeScale.small, color: colors.textSecondary }}>View</span>
          </button>
        </div>
      </div>

      {/* ── Settings panel (small floating) ── */}
      {showSettings && (
        <div style={{
          position: "absolute",
          top: 92,
          right: spacing.md,
          zIndex: 25,
          ...glass,
          padding: spacing.md,
          width: 200,
          display: "flex",
          flexDirection: "column",
          gap: spacing.sm,
        }}>
          <div style={{ ...typeScale.caption, color: colors.textDim, marginBottom: 2 }}>Display</div>
          <SettingRow label="Colormap">
            <select
              value={colormapMode}
              onChange={(e) => actions.setColormapMode(e.target.value as "intensity" | "range" | "elongation")}
              style={selectStyle}
            >
              <option value="intensity">Intensity</option>
              <option value="range">Range</option>
              <option value="elongation">Elongation</option>
            </select>
          </SettingRow>
          <SettingRow label="Objects">
            <select
              value={boxMode}
              onChange={(e) => actions.setBoxMode(e.target.value as "off" | "box" | "model")}
              style={selectStyle}
            >
              <option value="box">Wireframe</option>
              <option value="model">3D Models</option>
              <option value="off">Hidden</option>
            </select>
          </SettingRow>
          <SettingRow label="Opacity">
            <input
              type="range"
              min={0}
              max={1}
              step={0.05}
              value={pointOpacity}
              onChange={(e) => actions.setPointOpacity(parseFloat(e.target.value))}
              style={{ width: 80 }}
            />
          </SettingRow>
          <SettingRow label="Grid">
            <button
              onClick={() => actions.toggleGrid()}
              style={{
                padding: "2px 8px",
                borderRadius: 4,
                background: showGrid ? "rgba(0,232,157,0.1)" : "rgba(255,255,255,0.04)",
                color: showGrid ? colors.accent : colors.textDim,
                border: `1px solid ${showGrid ? colors.borderAccent : colors.border}`,
                fontSize: 10,
                cursor: "pointer",
              }}
            >
              {showGrid ? "On" : "Off"}
            </button>
          </SettingRow>
          <SettingRow label="Source">
            <div style={{ display: "flex", gap: 2 }}>
              {(["waymo", "mock"] as const).map((src) => (
                <button
                  key={src}
                  onClick={() => { actions.reset(); actions.setDataSource(src); }}
                  style={{
                    padding: "2px 7px",
                    borderRadius: 3,
                    fontSize: 9,
                    fontFamily: fonts.mono,
                    background: dataSource === src ? "rgba(0,232,157,0.1)" : "transparent",
                    color: dataSource === src ? colors.accent : colors.textDim,
                    border: `1px solid ${dataSource === src ? colors.borderAccent : "transparent"}`,
                    cursor: "pointer",
                  }}
                >
                  {src}
                </button>
              ))}
            </div>
          </SettingRow>
        </div>
      )}

      {/* ── Right panel: Autonomy Stack ── */}
      {showAutonomy && (
        <AutonomyPanel
          mainState={mainState}
          onClose={() => setShowAutonomy(false)}
          onExplain={() => setShowExplain(true)}
        />
      )}

      {!showAutonomy && (
        <button
          onClick={() => setShowAutonomy(true)}
          style={{
            position: "absolute",
            right: spacing.md,
            top: 92,
            zIndex: 20,
            ...glass,
            padding: "6px 10px",
            cursor: "pointer",
            display: "flex",
            alignItems: "center",
            gap: 5,
            border: `1px solid ${colors.borderAccent}`,
          }}
        >
          <Zap size={12} color={colors.accent} />
          <span style={{ ...typeScale.small, color: colors.accent }}>OpenEnv</span>
        </button>
      )}

      {/* ── Explain modal ── */}
      {showExplain && mainState.lastAction && (
        <ExplainModal
          action={mainState.lastAction}
          onClose={() => setShowExplain(false)}
        />
      )}

      {/* Timeline (bottom) — reused from existing */}
      <Timeline />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function ScenarioSelector({
  current,
  dataSource,
  onSelect,
}: {
  current: MockScenario;
  dataSource: string;
  onSelect: (id: MockScenario) => void;
}) {
  const [open, setOpen] = useState(false);
  const label = dataSource === "mock"
    ? SCENARIOS.find((s) => s.id === current)?.label || current
    : "Waymo Data";

  return (
    <div style={{ position: "relative" }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          ...glass,
          padding: "6px 12px",
          display: "flex",
          alignItems: "center",
          gap: 6,
          cursor: "pointer",
          border: `1px solid ${open ? colors.borderAccent : colors.border}`,
        }}
      >
        <span style={{ ...typeScale.small, color: colors.textPrimary, fontWeight: 500 }}>
          {label}
        </span>
        <ChevronDown size={12} color={colors.textDim} />
      </button>
      {open && (
        <div style={{
          position: "absolute",
          top: "calc(100% + 4px)",
          left: 0,
          ...glass,
          padding: 4,
          minWidth: 180,
          zIndex: 100,
        }}>
          {SCENARIOS.map((s) => (
            <button
              key={s.id}
              onClick={() => { onSelect(s.id); setOpen(false); }}
              style={{
                display: "block",
                width: "100%",
                padding: "6px 10px",
                borderRadius: 6,
                fontSize: 12,
                textAlign: "left",
                color: s.id === current ? colors.accent : colors.textSecondary,
                background: s.id === current ? "rgba(0,232,157,0.06)" : "transparent",
                border: "none",
                cursor: "pointer",
                fontFamily: fonts.sans,
              }}
            >
              {s.label}
            </button>
          ))}
        </div>
      )}
    </div>
  );
}

function AutonomyPanel({
  mainState,
  onClose,
  onExplain,
}: {
  mainState: ReturnType<typeof useSimManager.getState>["mainState"];
  onClose: () => void;
  onExplain: () => void;
}) {
  const openenvConnected = useSimManager((s) => s.openenvConnected);
  const openenvLastUpdate = useSimManager((s) => s.openenvLastUpdate);
  const [countdown, setCountdown] = useState(3);

  useEffect(() => {
    const interval = setInterval(() => {
      const elapsed = (Date.now() - openenvLastUpdate) / 1000;
      setCountdown(Math.max(0, Math.round(3 - elapsed)));
    }, 500);
    return () => clearInterval(interval);
  }, [openenvLastUpdate]);

  const la = mainState.lastAction;
  const mode = getOpenEnvMode();

  return (
    <div style={{
      position: "absolute",
      right: spacing.md,
      top: 92,
      zIndex: 20,
      width: 260,
      ...glass,
      padding: spacing.md,
      display: "flex",
      flexDirection: "column",
      gap: spacing.sm,
    }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <Zap size={13} color={colors.accent} />
          <span style={{ ...typeScale.h3, color: colors.textPrimary }}>Autonomy Stack</span>
        </div>
        <button onClick={onClose} style={{ cursor: "pointer", padding: 2, border: "none", background: "none" }}>
          <X size={12} color={colors.textDim} />
        </button>
      </div>

      {/* Status */}
      <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
        <Badge variant={openenvConnected ? "success" : "warning"} dot>
          {mode === "real" ? "Connected" : "Mock Mode"}
        </Badge>
        <span style={{ ...typeScale.mono, color: colors.textDim }}>
          next: {countdown}s
        </span>
      </div>

      {/* Separator */}
      <div style={{ height: 1, background: colors.border }} />

      {/* Last action */}
      {la ? (
        <>
          <div>
            <div style={{ ...typeScale.caption, color: colors.textDim, marginBottom: 4 }}>
              Last Action
            </div>
            <div style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}>
              <span style={{
                ...typeScale.mono,
                fontSize: 13,
                fontWeight: 600,
                color: colors.accent,
              }}>
                {la.action.replace(/_/g, " ")}
              </span>
              <Badge variant={la.reward >= 0.5 ? "success" : la.reward >= 0 ? "info" : "error"}>
                R: {la.reward.toFixed(2)}
              </Badge>
            </div>
          </div>

          <div style={{ display: "flex", gap: spacing.sm }}>
            <MiniStat label="Latency" value={`${la.latencyMs}ms`} />
            <MiniStat label="Cumulative" value={mainState.cumulativeReward.toFixed(2)} />
          </div>

          <div style={{ display: "flex", gap: spacing.sm }}>
            <MiniStat label="Branch" value={la.branchId.slice(0, 10)} mono />
            <MiniStat label="Frame" value={`${mainState.frameIndex}`} />
          </div>

          {/* Explain button */}
          <button
            onClick={onExplain}
            style={{
              padding: "6px 10px",
              borderRadius: 6,
              background: "rgba(0,232,157,0.06)",
              border: `1px solid ${colors.borderAccent}`,
              color: colors.accent,
              fontSize: 11,
              fontWeight: 500,
              fontFamily: fonts.sans,
              cursor: "pointer",
              display: "flex",
              alignItems: "center",
              gap: 5,
            }}
          >
            Explain last decision
            <ArrowRight size={11} />
          </button>
        </>
      ) : (
        <div style={{ ...typeScale.small, color: colors.textDim, padding: "8px 0" }}>
          Waiting for first update...
        </div>
      )}
    </div>
  );
}

function ExplainModal({
  action,
  onClose,
}: {
  action: NonNullable<ReturnType<typeof useSimManager.getState>["mainState"]["lastAction"]>;
  onClose: () => void;
}) {
  return (
    <div
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.5)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        zIndex: 100,
      }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        style={{
          background: colors.bgCard,
          border: `1px solid ${colors.border}`,
          borderRadius: radius.xl,
          padding: spacing.xl,
          maxWidth: 480,
          width: "90%",
        }}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: spacing.lg }}>
          <span style={{ ...typeScale.h2, color: colors.textPrimary }}>Decision Explanation</span>
          <button onClick={onClose} style={{ cursor: "pointer", padding: 4, border: "none", background: "none" }}>
            <X size={16} color={colors.textDim} />
          </button>
        </div>
        <div style={{ marginBottom: spacing.md }}>
          <span style={{ ...typeScale.caption, color: colors.textDim }}>Action</span>
          <div style={{ ...typeScale.h3, color: colors.accent, marginTop: 4 }}>
            {action.action.replace(/_/g, " ")}
          </div>
        </div>
        <div style={{ marginBottom: spacing.md }}>
          <span style={{ ...typeScale.caption, color: colors.textDim }}>Reward</span>
          <div style={{ ...typeScale.h3, color: action.reward >= 0.5 ? colors.success : colors.warning, marginTop: 4 }}>
            {action.reward.toFixed(3)}
          </div>
        </div>
        <div style={{ marginBottom: spacing.md }}>
          <span style={{ ...typeScale.caption, color: colors.textDim }}>Reasoning</span>
          <div style={{ ...typeScale.body, color: colors.textSecondary, marginTop: 4, lineHeight: 1.6 }}>
            {action.explanation}
          </div>
        </div>
        <div>
          <span style={{ ...typeScale.caption, color: colors.textDim }}>Branch ID</span>
          <div style={{ ...typeScale.mono, color: colors.textDim, marginTop: 4 }}>
            {action.branchId}
          </div>
        </div>
      </div>
    </div>
  );
}

function NavButton({ icon: Icon, label, onClick }: { icon: typeof LayoutGrid; label: string; onClick: () => void }) {
  return (
    <div style={{ pointerEvents: "auto" }}>
      <button
        onClick={onClick}
        style={{
          ...glass,
          padding: "6px 10px",
          display: "flex",
          alignItems: "center",
          gap: 5,
          cursor: "pointer",
        }}
      >
        <Icon size={13} color={colors.textSecondary} />
        <span style={{ ...typeScale.small, color: colors.textSecondary }}>{label}</span>
      </button>
    </div>
  );
}

function MiniStat({ label, value, mono }: { label: string; value: string; mono?: boolean }) {
  return (
    <div style={{ flex: 1 }}>
      <div style={{ ...typeScale.caption, color: colors.textDim, fontSize: 8, marginBottom: 2 }}>{label}</div>
      <div style={{
        fontSize: 11,
        fontWeight: 500,
        fontFamily: mono ? fonts.mono : fonts.sans,
        color: colors.textSecondary,
      }}>
        {value}
      </div>
    </div>
  );
}

function SettingRow({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div style={{
      display: "flex",
      alignItems: "center",
      justifyContent: "space-between",
    }}>
      <span style={{ ...typeScale.small, color: colors.textSecondary }}>{label}</span>
      {children}
    </div>
  );
}

const selectStyle: React.CSSProperties = {
  background: "rgba(255,255,255,0.04)",
  border: `1px solid ${colors.border}`,
  borderRadius: 4,
  color: colors.textSecondary,
  fontSize: 10,
  fontFamily: fonts.mono,
  padding: "2px 6px",
  outline: "none",
};
