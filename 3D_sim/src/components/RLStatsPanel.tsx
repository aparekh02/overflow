/**
 * RLStatsPanel — Floating overlay showing live RL training stats.
 *
 * Shows: step reward, cumulative reward sparkline, constraint indicators,
 * review agent speed/action decomposition, and episode outcome.
 */

import { useMemo } from "react";
import { useRLStore } from "../rlStore";
import { colors, fonts } from "../theme";

const WEATHER_ICONS: Record<string, string> = {
  clear: "☀",
  overcast: "⛅",
  rain: "🌧",
  heavy_rain: "⛈",
  fog: "🌫",
  snow: "❄",
};

// ── Sparkline (reward history) ─────────────────────────────────────────────

function Sparkline({ data, width = 160, height = 32 }: { data: number[]; width?: number; height?: number }) {
  const path = useMemo(() => {
    if (data.length < 2) return "";
    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = Math.max(max - min, 1);
    const pts = data.map((v, i) => {
      const x = (i / (data.length - 1)) * width;
      const y = height - ((v - min) / range) * height;
      return `${x.toFixed(1)},${y.toFixed(1)}`;
    });
    return `M ${pts.join(" L ")}`;
  }, [data, width, height]);

  if (data.length < 2) return null;

  const lastVal = data[data.length - 1];
  const lineColor = lastVal >= 0 ? colors.accent : "#FF4466";

  return (
    <svg width={width} height={height} style={{ display: "block" }}>
      <path d={path} fill="none" stroke={lineColor} strokeWidth={1.5} opacity={0.85} />
      {/* Zero line */}
      <line x1={0} x2={width} y1={height / 2} y2={height / 2}
        stroke="rgba(255,255,255,0.12)" strokeWidth={0.5} strokeDasharray="3,3" />
    </svg>
  );
}

// ── Constraint bar ─────────────────────────────────────────────────────────

function ConstraintBar({ label, value, max = 1.0, color: barColor }: {
  label: string; value: number; max?: number; color?: string;
}) {
  const pct = Math.min(100, (value / max) * 100);
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <span style={{ fontSize: 8, color: colors.textDim, width: 56, flexShrink: 0 }}>{label}</span>
      <div style={{ flex: 1, height: 3, background: "rgba(255,255,255,0.08)", borderRadius: 2 }}>
        <div style={{
          width: `${pct}%`, height: "100%", borderRadius: 2,
          background: barColor ?? colors.accent, transition: "width 0.3s",
        }} />
      </div>
      <span style={{ fontSize: 8, fontFamily: fonts.mono, color: colors.textSecondary, width: 28, textAlign: "right" }}>
        {value.toFixed(2)}
      </span>
    </div>
  );
}

// ── Reward component breakdown ─────────────────────────────────────────────

function RewardBreakdown({ rc }: { rc: Record<string, number> }) {
  const entries = Object.entries(rc).filter(([k]) => k !== "total");
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {entries.map(([key, val]) => (
        <div key={key} style={{ display: "flex", justifyContent: "space-between" }}>
          <span style={{ fontSize: 8, color: colors.textDim }}>{key.replace("_", " ")}</span>
          <span style={{
            fontSize: 8, fontFamily: fonts.mono,
            color: val >= 0 ? colors.accent : "#FF6688",
          }}>
            {val >= 0 ? "+" : ""}{val.toFixed(3)}
          </span>
        </div>
      ))}
    </div>
  );
}

// ── Main panel ─────────────────────────────────────────────────────────────

export default function RLStatsPanel() {
  const rlModeActive = useRLStore((s) => s.rlModeActive);
  const rlScene = useRLStore((s) => s.rlScene);
  const rewardHistory = useRLStore((s) => s.rewardHistory);
  const episodeInfo = useRLStore((s) => s.episodeInfo);
  const episodeEnd = useRLStore((s) => s.episodeEnd);
  const connected = useRLStore((s) => s.connected);
  const paused = useRLStore((s) => s.paused);

  if (!rlModeActive) return null;

  const constraints = rlScene?.constraints;
  const ego = rlScene?.ego;
  const rc = rlScene?.reward_components;

  return (
    <div style={{
      position: "absolute",
      top: 56, right: 16,
      width: 210,
      background: "rgba(8,11,20,0.85)",
      backdropFilter: "blur(20px)",
      border: "1px solid rgba(255,255,255,0.07)",
      borderRadius: 10,
      padding: "10px 0",
      fontFamily: fonts.sans,
      zIndex: 10,
      display: "flex",
      flexDirection: "column",
      gap: 2,
    }}>
      {/* Header */}
      <div style={{ padding: "0 12px 6px", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
          <span style={{ fontSize: 9, fontWeight: 700, letterSpacing: "1.2px", textTransform: "uppercase", color: colors.textDim }}>
            OpenENV
          </span>
          <div style={{ display: "flex", gap: 5, alignItems: "center" }}>
            {paused && <span style={{ fontSize: 8, color: "#FFCC44" }}>⏸ PAUSED</span>}
            <span style={{
              width: 6, height: 6, borderRadius: "50%",
              background: connected ? colors.accent : "#FF4466",
              boxShadow: connected ? `0 0 6px ${colors.accent}` : "none",
            }} />
          </div>
        </div>

        {episodeInfo && (
          <div style={{ marginTop: 4, fontSize: 9, color: colors.textSecondary }}>
            <span style={{ color: colors.accentBlue, textTransform: "capitalize" }}>
              {episodeInfo.incidentType.replace("_", " ")}
            </span>
            {constraints && (
              <span style={{ marginLeft: 6, color: colors.textDim }}>
                {WEATHER_ICONS[constraints.weather] ?? ""} {constraints.weather}
              </span>
            )}
          </div>
        )}
      </div>

      {/* Episode / step stats */}
      {rlScene && (
        <div style={{ padding: "6px 12px", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <Stat label="Step" value={String(rlScene.step)} />
            <Stat label="Reward" value={(rlScene.reward >= 0 ? "+" : "") + rlScene.reward.toFixed(3)}
              valueColor={rlScene.reward >= 0 ? colors.accent : "#FF6688"} />
            <Stat label="Total" value={(rlScene.episode_reward >= 0 ? "+" : "") + rlScene.episode_reward.toFixed(1)}
              valueColor={rlScene.episode_reward >= 0 ? colors.accent : "#FF6688"} />
          </div>
          <Sparkline data={rewardHistory} width={186} height={28} />
        </div>
      )}

      {/* Ego speed + action */}
      {ego && (
        <div style={{ padding: "6px 12px", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
          <Label text="Review Agent" />
          <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
            <Stat label="Speed" value={`${(ego.speed * 2.237).toFixed(1)} mph`} />
            <Stat label="Accel" value={`${ego.accel >= 0 ? "+" : ""}${ego.accel.toFixed(2)} m/s²`} />
          </div>
        </div>
      )}

      {/* Constraint bars */}
      {constraints && (
        <div style={{ padding: "6px 12px", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
          <Label text="Waymo Constraints" />
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <ConstraintBar label="Friction" value={constraints.friction} color="#FFAA33" />
            <ConstraintBar label="Visibility" value={constraints.visibility_m} max={150} color={colors.accentBlue} />
            <ConstraintBar label="Traffic" value={constraints.traffic_density} color="#DD44FF" />
          </div>
        </div>
      )}

      {/* Reward decomposition */}
      {rc && (
        <div style={{ padding: "6px 12px", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
          <Label text="Reward Breakdown" />
          <RewardBreakdown rc={rc as Record<string, number>} />
        </div>
      )}

      {/* Episode end banner */}
      {episodeEnd && (
        <div style={{
          margin: "4px 10px",
          padding: "6px 10px",
          background: episodeEnd.totalReward > 0
            ? "rgba(0,232,157,0.08)"
            : "rgba(255,68,102,0.08)",
          border: `1px solid ${episodeEnd.totalReward > 0 ? "rgba(0,232,157,0.2)" : "rgba(255,68,102,0.2)"}`,
          borderRadius: 6,
        }}>
          <div style={{ fontSize: 9, fontWeight: 600, color: episodeEnd.totalReward > 0 ? colors.accent : "#FF6688" }}>
            Episode Complete
          </div>
          <div style={{ fontSize: 8, color: colors.textDim, marginTop: 2 }}>
            Total reward: {episodeEnd.totalReward.toFixed(2)} · {episodeEnd.steps} steps
          </div>
        </div>
      )}
    </div>
  );
}

// ── Sub-components ─────────────────────────────────────────────────────────

function Label({ text }: { text: string }) {
  return (
    <div style={{
      fontSize: 8, fontWeight: 700, letterSpacing: "1.2px",
      textTransform: "uppercase", color: colors.textDim, marginBottom: 4,
    }}>
      {text}
    </div>
  );
}

function Stat({
  label, value, valueColor,
}: {
  label: string; value: string; valueColor?: string;
}) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 1 }}>
      <span style={{ fontSize: 7, color: colors.textDim }}>{label}</span>
      <span style={{ fontSize: 10, fontFamily: fonts.mono, color: valueColor ?? colors.textPrimary }}>
        {value}
      </span>
    </div>
  );
}
