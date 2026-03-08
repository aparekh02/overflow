/**
 * Timeline — Floating glass bottom bar.
 * Play/pause, step, scrub, speed. Positioned absolute over the 3D canvas.
 */

import { useEffect, useRef, useCallback } from "react";
import { colors, fonts } from "../theme";
import { useStore } from "../store";

export default function Timeline() {
  const currentFrameIndex = useStore((s) => s.currentFrameIndex);
  const totalFrames = useStore((s) => s.totalFrames);
  const isPlaying = useStore((s) => s.isPlaying);
  const playbackSpeed = useStore((s) => s.playbackSpeed);
  const setFrame = useStore((s) => s.actions.setFrame);
  const nextFrame = useStore((s) => s.actions.nextFrame);
  const prevFrame = useStore((s) => s.actions.prevFrame);
  const togglePlay = useStore((s) => s.actions.togglePlay);
  const setPlaybackSpeed = useStore((s) => s.actions.setPlaybackSpeed);
  const fps = useStore((s) => s.sceneData?.fps ?? 10);

  // Playback loop
  const rafRef = useRef(0);
  const lastTimeRef = useRef(0);
  useEffect(() => {
    if (!isPlaying) { if (rafRef.current) cancelAnimationFrame(rafRef.current); return; }
    const interval = 1000 / (fps * playbackSpeed);
    const tick = (now: number) => {
      if (now - lastTimeRef.current >= interval) { lastTimeRef.current = now; nextFrame(); }
      rafRef.current = requestAnimationFrame(tick);
    };
    lastTimeRef.current = performance.now();
    rafRef.current = requestAnimationFrame(tick);
    return () => { if (rafRef.current) cancelAnimationFrame(rafRef.current); };
  }, [isPlaying, fps, playbackSpeed, nextFrame]);

  // Keyboard
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === " ") { e.preventDefault(); togglePlay(); }
      if (e.key === "ArrowRight") nextFrame();
      if (e.key === "ArrowLeft") prevFrame();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [togglePlay, nextFrame, prevFrame]);

  // Scrub bar drag
  const barRef = useRef<HTMLDivElement>(null);
  const draggingRef = useRef(false);

  const scrubTo = useCallback((clientX: number) => {
    if (!barRef.current) return;
    const rect = barRef.current.getBoundingClientRect();
    const frac = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
    setFrame(Math.round(frac * (totalFrames - 1)));
  }, [setFrame, totalFrames]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => { if (draggingRef.current) scrubTo(e.clientX); };
    const onUp = () => { draggingRef.current = false; };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => { window.removeEventListener("mousemove", onMove); window.removeEventListener("mouseup", onUp); };
  }, [scrubTo]);

  const progress = totalFrames > 1 ? currentFrameIndex / (totalFrames - 1) : 0;
  const time = (currentFrameIndex / fps).toFixed(1);
  const totalTime = ((totalFrames - 1) / fps).toFixed(1);

  return (
    <div style={{
      position: "absolute",
      bottom: 12, left: 12, right: 12,
      height: 48,
      display: "flex", alignItems: "center",
      padding: "0 14px", gap: 10,
      background: "rgba(12,15,26,0.75)",
      backdropFilter: "blur(20px)",
      border: "1px solid rgba(255,255,255,0.06)",
      borderRadius: 12,
      fontFamily: fonts.sans,
      userSelect: "none",
      zIndex: 10,
    }}>
      {/* Step back */}
      <Btn onClick={prevFrame} title="← Previous">
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <path d="M4 3v10M13 3L6 8l7 5V3z" fill={colors.textSecondary} />
        </svg>
      </Btn>

      {/* Play/Pause */}
      <Btn onClick={togglePlay} title="Space" accent>
        {isPlaying ? (
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <rect x="3" y="2" width="4" height="12" rx="1" fill={colors.accent} />
            <rect x="9" y="2" width="4" height="12" rx="1" fill={colors.accent} />
          </svg>
        ) : (
          <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
            <path d="M4 2l10 6-10 6V2z" fill={colors.accent} />
          </svg>
        )}
      </Btn>

      {/* Step forward */}
      <Btn onClick={nextFrame} title="→ Next">
        <svg width="12" height="12" viewBox="0 0 16 16" fill="none">
          <path d="M12 3v10M3 3l7 5-7 5V3z" fill={colors.textSecondary} />
        </svg>
      </Btn>

      {/* Time */}
      <span style={{
        fontSize: 10, fontFamily: fonts.mono,
        color: colors.textSecondary, minWidth: 64, textAlign: "center",
      }}>
        {time}s / {totalTime}s
      </span>

      {/* Scrub bar */}
      <div
        ref={barRef}
        onMouseDown={(e) => { draggingRef.current = true; scrubTo(e.clientX); }}
        style={{
          flex: 1, height: 20,
          display: "flex", alignItems: "center",
          cursor: "pointer", position: "relative",
        }}
      >
        {/* Track */}
        <div style={{
          position: "absolute", left: 0, right: 0, height: 3,
          background: "rgba(255,255,255,0.06)", borderRadius: 2,
        }} />
        {/* Fill */}
        <div style={{
          position: "absolute", left: 0, height: 3,
          width: `${progress * 100}%`,
          background: `linear-gradient(90deg, ${colors.accent}, ${colors.accentBlue})`,
          borderRadius: 2, transition: draggingRef.current ? "none" : "width 0.06s linear",
        }} />
        {/* Handle */}
        <div style={{
          position: "absolute",
          left: `calc(${progress * 100}% - 6px)`,
          width: 12, height: 12, borderRadius: "50%",
          background: colors.accent,
          boxShadow: `0 0 8px ${colors.accentDim}`,
          transition: draggingRef.current ? "none" : "left 0.06s linear",
        }} />
      </div>

      {/* Frame count */}
      <span style={{
        fontSize: 9, fontFamily: fonts.mono,
        color: colors.textDim, minWidth: 48, textAlign: "right",
      }}>
        {currentFrameIndex + 1}/{totalFrames}
      </span>

      {/* Speed */}
      <div style={{ display: "flex", gap: 1 }}>
        {[0.5, 1, 2, 4].map((speed) => (
          <button key={speed} onClick={() => setPlaybackSpeed(speed)} style={{
            fontSize: 9, fontWeight: playbackSpeed === speed ? 600 : 400,
            fontFamily: fonts.mono, padding: "3px 5px",
            border: "none", borderRadius: 3, cursor: "pointer",
            background: playbackSpeed === speed ? "rgba(0,200,219,0.15)" : "transparent",
            color: playbackSpeed === speed ? colors.accentBlue : colors.textDim,
            transition: "all 0.12s",
          }}>
            {speed}×
          </button>
        ))}
      </div>
    </div>
  );
}

function Btn({ onClick, title, accent, children }: {
  onClick: () => void; title?: string; accent?: boolean; children: React.ReactNode;
}) {
  return (
    <button onClick={onClick} title={title} style={{
      width: 28, height: 28,
      display: "flex", alignItems: "center", justifyContent: "center",
      border: "none", borderRadius: 6, cursor: "pointer", padding: 0,
      background: accent ? "rgba(0,232,157,0.08)" : "transparent",
      transition: "background 0.12s",
    }}>
      {children}
    </button>
  );
}
