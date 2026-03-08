/**
 * IncidentPanel — Floating glass panel for submitting incident tickets to OpenENV.
 *
 * Allows operators to paste a real incident report, which is parsed by
 * the Python backend (LLM + rules) and used to seed a new RL episode
 * with the correct scene geometry, agents, and environmental constraints.
 */

import { useState, useRef } from "react";
import { useRLStore } from "../rlStore";
import { wsClient } from "../middleware/wsClient";
import { colors, fonts } from "../theme";

const EXAMPLE_TICKETS = [
  {
    id: "INC-001",
    label: "Jaywalker / Rain",
    text: "Waymo AV traveling at 28 mph on Market St in rainy conditions. A pedestrian jaywalked suddenly from between parked cars at 15m ahead. AV detected pedestrian and braked hard, narrowly avoiding contact. Speed limit: 30 mph. High traffic density.",
  },
  {
    id: "INC-002",
    label: "Intersection / Fog",
    text: "Vehicle approaching intersection at 35 mph, fog conditions, 35 mph zone. Cross traffic ran a red light at approximately 60 ft distance. Ego vehicle braked and swerved right, near-miss recorded. Two trailing vehicles within 20m.",
  },
  {
    id: "INC-003",
    label: "Lane Change / Clear",
    text: "On I-280 highway, vehicle in adjacent lane made abrupt lane change without signaling. Relative closing speed ~5 mph. Clear weather, 65 mph limit, light traffic. Ego decelerated and moved to shoulder to avoid sideswipe.",
  },
];

export default function IncidentPanel() {
  const connected = useRLStore((s) => s.connected);
  const rlModeActive = useRLStore((s) => s.rlModeActive);
  const episodeInfo = useRLStore((s) => s.episodeInfo);
  const error = useRLStore((s) => s.error);
  const paused = useRLStore((s) => s.paused);
  const actions = useRLStore((s) => s.actions);

  const [ticketText, setTicketText] = useState("");
  const [ticketId, setTicketId] = useState("INC-custom");
  const [submitting, setSubmitting] = useState(false);
  const [expanded, setExpanded] = useState(false);
  const textRef = useRef<HTMLTextAreaElement>(null);

  if (!rlModeActive) return null;

  const handleSubmit = () => {
    if (!ticketText.trim() || !connected) return;
    setSubmitting(true);
    wsClient.loadIncident(ticketId, ticketText.trim());
    setTimeout(() => setSubmitting(false), 800);
  };

  const handleExample = (ex: (typeof EXAMPLE_TICKETS)[0]) => {
    setTicketId(ex.id);
    setTicketText(ex.text);
    setExpanded(true);
    setTimeout(() => textRef.current?.focus(), 50);
  };

  const headerColor = connected ? colors.accent : "#FF6B6B";

  return (
    <div style={{
      position: "absolute",
      bottom: 80, right: 16,
      width: expanded ? 360 : 220,
      maxHeight: expanded ? 520 : 48,
      background: "rgba(8,11,20,0.88)",
      backdropFilter: "blur(20px)",
      border: `1px solid ${connected ? "rgba(0,232,157,0.2)" : "rgba(255,107,107,0.2)"}`,
      borderRadius: 12,
      overflow: "hidden",
      fontFamily: fonts.sans,
      zIndex: 20,
      transition: "all 0.25s cubic-bezier(0.4, 0, 0.2, 1)",
    }}>
      {/* Header */}
      <div
        onClick={() => setExpanded((v) => !v)}
        style={{
          display: "flex", alignItems: "center", gap: 8,
          padding: "12px 14px", cursor: "pointer",
          borderBottom: expanded ? "1px solid rgba(255,255,255,0.06)" : "none",
        }}
      >
        <span style={{
          width: 7, height: 7, borderRadius: "50%",
          background: headerColor,
          boxShadow: connected ? `0 0 8px ${colors.accent}` : "none",
          flexShrink: 0,
        }} />
        <span style={{
          flex: 1, fontSize: 11, fontWeight: 600,
          color: connected ? colors.textPrimary : "#FF6B6B",
          letterSpacing: "0.3px",
        }}>
          OpenENV {connected ? (episodeInfo ? `· ${episodeInfo.incidentType}` : "· Ready") : "· Offline"}
        </span>
        <span style={{ fontSize: 9, color: colors.textDim, transform: expanded ? "rotate(180deg)" : "none", transition: "0.2s" }}>▼</span>
      </div>

      {expanded && (
        <div style={{ padding: "10px 14px 14px", display: "flex", flexDirection: "column", gap: 10 }}>
          {/* Connection error */}
          {error && (
            <div style={{ fontSize: 9, color: "#FF6B6B", fontFamily: fonts.mono, background: "rgba(255,107,107,0.08)", padding: "5px 8px", borderRadius: 5 }}>
              {error}
            </div>
          )}

          {/* Episode status */}
          {episodeInfo && (
            <div style={{ fontSize: 9, color: colors.textDim, fontFamily: fonts.mono }}>
              <span style={{ color: colors.accent }}>{episodeInfo.incidentType.replace("_", " ")}</span>
              {" · "}
              <span>{(episodeInfo.constraints as { weather?: string }).weather ?? "clear"}</span>
            </div>
          )}

          {/* Controls */}
          <div style={{ display: "flex", gap: 5 }}>
            <CtrlBtn
              label={paused ? "Resume" : "Pause"}
              onClick={() => paused ? wsClient.resume() : wsClient.pause()}
              disabled={!connected}
              accent={!paused}
            />
            <CtrlBtn label="Reset" onClick={() => wsClient.reset()} disabled={!connected} />
          </div>

          <Sep />

          {/* Ticket ID */}
          <div>
            <Label text="Ticket ID" />
            <input
              value={ticketId}
              onChange={(e) => setTicketId(e.target.value)}
              style={inputStyle}
            />
          </div>

          {/* Ticket text */}
          <div>
            <Label text="Incident Description" />
            <textarea
              ref={textRef}
              value={ticketText}
              onChange={(e) => setTicketText(e.target.value)}
              placeholder="Describe the incident — weather, speed, involved actors, what happened…"
              rows={5}
              style={{ ...inputStyle, resize: "vertical", minHeight: 90, lineHeight: 1.45 }}
            />
          </div>

          <button
            onClick={handleSubmit}
            disabled={!connected || !ticketText.trim() || submitting}
            style={{
              padding: "8px 0", fontSize: 11, fontWeight: 600,
              fontFamily: fonts.sans, cursor: connected && ticketText.trim() ? "pointer" : "not-allowed",
              background: connected && ticketText.trim()
                ? `linear-gradient(135deg, rgba(0,232,157,0.2), rgba(0,200,219,0.15))`
                : "rgba(255,255,255,0.03)",
              color: connected && ticketText.trim() ? colors.accent : colors.textDim,
              border: `1px solid ${connected && ticketText.trim() ? "rgba(0,232,157,0.35)" : colors.border}`,
              borderRadius: 7, transition: "all 0.15s",
            }}
          >
            {submitting ? "Starting episode…" : "Load Incident → RL Episode"}
          </button>

          <Sep />

          {/* Quick examples */}
          <Label text="Quick Examples" />
          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            {EXAMPLE_TICKETS.map((ex) => (
              <button
                key={ex.id}
                onClick={() => handleExample(ex)}
                style={{
                  padding: "5px 9px", fontSize: 9, textAlign: "left",
                  fontFamily: fonts.sans,
                  background: "rgba(255,255,255,0.03)",
                  color: colors.textSecondary,
                  border: "1px solid rgba(255,255,255,0.06)",
                  borderRadius: 5, cursor: "pointer", transition: "all 0.1s",
                }}
              >
                <span style={{ color: colors.accentBlue, fontFamily: fonts.mono }}>{ex.id}</span>
                {" · "}{ex.label}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// ── Sub-components ────────────────────────────────────────────────────────────

function Label({ text }: { text: string }) {
  return (
    <div style={{
      fontSize: 8, fontWeight: 700, letterSpacing: "1.2px",
      textTransform: "uppercase", color: colors.textDim,
      marginBottom: 4,
    }}>
      {text}
    </div>
  );
}

function Sep() {
  return <div style={{ height: 1, background: "rgba(255,255,255,0.05)", margin: "2px 0" }} />;
}

function CtrlBtn({
  label, onClick, disabled, accent,
}: {
  label: string; onClick: () => void; disabled?: boolean; accent?: boolean;
}) {
  return (
    <button onClick={onClick} disabled={disabled} style={{
      flex: 1, padding: "5px 0", fontSize: 10, fontWeight: 500,
      fontFamily: fonts.sans, cursor: disabled ? "not-allowed" : "pointer",
      background: accent ? "rgba(0,232,157,0.1)" : "rgba(255,255,255,0.04)",
      color: accent ? colors.accent : colors.textSecondary,
      border: `1px solid ${accent ? "rgba(0,232,157,0.25)" : "rgba(255,255,255,0.07)"}`,
      borderRadius: 5, opacity: disabled ? 0.4 : 1, transition: "all 0.12s",
    }}>
      {label}
    </button>
  );
}

const inputStyle: React.CSSProperties = {
  width: "100%",
  padding: "6px 8px",
  fontSize: 10,
  fontFamily: fonts.mono,
  background: "rgba(255,255,255,0.04)",
  color: colors.textPrimary,
  border: "1px solid rgba(255,255,255,0.08)",
  borderRadius: 5,
  outline: "none",
  boxSizing: "border-box",
  resize: "none",
};
