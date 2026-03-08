/**
 * AgentOverlay — Injects OpenENV RL agents into the live Waymo 3D scene.
 *
 * These render using the EXACT same pipeline as BoundingBoxes.tsx (wireframe
 * edges + transparent fill + useFrame with no React re-renders) so the injected
 * boxes look identical to real Waymo-detected objects in the scene.
 *
 * Coordinate space: The Waymo scene is always in ego-relative space — the ego
 * vehicle sits at (0,0). We convert RL world-space positions → ego-relative by
 * subtracting the RL ego position, so injected boxes slot right into the
 * existing scene geometry.
 *
 * Agent visual identity (matching Waymo palette):
 *   REVIEW AGENT   → cyan  #00E5FF  (the HIGHLIGHT_COLOR used elsewhere)
 *   INCIDENT ACTOR → bright white/red with pulsing glow
 *   RL TRAFFIC     → same orange as Waymo vehicles, slightly lighter fill
 */

import { useRef, useMemo } from "react";
import * as THREE from "three";
import { useFrame } from "@react-three/fiber";
import { useRLStore } from "../rlStore";

// ── Constants (must match BoundingBoxes.tsx) ─────────────────────────────

const MAX_RL_AGENTS = 20;

// Colour scheme chosen to sit harmoniously with Waymo's orange/lime palette
const COLOR_REVIEW   = new THREE.Color("#00E5FF");   // cyan — matches HIGHLIGHT_COLOR
const COLOR_INCIDENT = new THREE.Color("#FFFFFF");   // bright white — unmistakably synthetic
const COLOR_TRAFFIC  = new THREE.Color("#FFAA33");   // amber — same family as Waymo orange

// Unit cube edge pairs (identical to BoundingBoxes.tsx)
const UNIT_CUBE_EDGES: [number, number, number, number, number, number][] = [
  [-0.5,-0.5,-0.5, 0.5,-0.5,-0.5], [0.5,-0.5,-0.5, 0.5,0.5,-0.5],
  [0.5,0.5,-0.5, -0.5,0.5,-0.5],   [-0.5,0.5,-0.5, -0.5,-0.5,-0.5],
  [-0.5,-0.5,0.5, 0.5,-0.5,0.5],   [0.5,-0.5,0.5, 0.5,0.5,0.5],
  [0.5,0.5,0.5, -0.5,0.5,0.5],     [-0.5,0.5,0.5, -0.5,-0.5,0.5],
  [-0.5,-0.5,-0.5, -0.5,-0.5,0.5], [0.5,-0.5,-0.5, 0.5,-0.5,0.5],
  [0.5,0.5,-0.5, 0.5,0.5,0.5],     [-0.5,0.5,-0.5, -0.5,0.5,0.5],
];

const EDGE_VERTS_PER_BOX = 24;
const _mat4   = new THREE.Matrix4();
const _pos    = new THREE.Vector3();
const _quat   = new THREE.Quaternion();
const _scale  = new THREE.Vector3();
const _euler  = new THREE.Euler();
const _color  = new THREE.Color();
const _boxGeo = new THREE.BoxGeometry(1, 1, 1);

// ── Colour lookup ─────────────────────────────────────────────────────────

function agentColor(type: string): THREE.Color {
  if (type === "review")   return COLOR_REVIEW;
  if (type === "incident" || type === "pedestrian" || type === "cyclist") return COLOR_INCIDENT;
  return COLOR_TRAFFIC;
}

// ── Predicted trajectory for review agent ────────────────────────────────

function ReviewTrajectory() {
  const geoRef   = useRef<THREE.BufferGeometry>(null);
  const positions = useMemo(() => new Float32Array(32 * 3), []);

  useFrame(() => {
    const geo = geoRef.current;
    if (!geo) return;
    const s = useRLStore.getState();
    if (!s.rlModeActive || !s.rlScene) { geo.setDrawRange(0, 0); return; }

    const { ego, agents } = s.rlScene;
    const traj = ego.trajectory;
    if (!traj || traj.length < 2) { geo.setDrawRange(0, 0); return; }

    // Convert trajectory to ego-relative
    let vi = 0;
    const len = Math.min(traj.length, 32);
    for (let i = 0; i < len; i++) {
      positions[vi++] = traj[i][0] - ego.x;
      positions[vi++] = traj[i][1] - ego.y;
      positions[vi++] = 0.3;
    }

    const attr = geo.getAttribute("position") as THREE.BufferAttribute;
    (attr.array as Float32Array).set(positions.subarray(0, vi));
    attr.needsUpdate = true;
    geo.setDrawRange(0, len);
    void agents; // suppress unused warning
  });

  return (
    <line>
      <bufferGeometry ref={geoRef}>
        <bufferAttribute
          attach="attributes-position"
          args={[new Float32Array(32 * 3), 3]}
          usage={THREE.DynamicDrawUsage}
        />
      </bufferGeometry>
      <lineBasicMaterial color="#00E5FF" transparent opacity={0.65} linewidth={1} />
    </line>
  );
}

// ── Main: wireframe + fill boxes (same technique as BoundingBoxes.tsx) ───

export default function AgentOverlay() {
  const fillMeshRef  = useRef<THREE.InstancedMesh>(null);
  const edgeGeoRef   = useRef<THREE.BufferGeometry>(null);

  const fillMat = useMemo(() => new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.18, depthWrite: false, side: THREE.DoubleSide,
  }), []);

  const edgeMat = useMemo(() => new THREE.LineBasicMaterial({
    vertexColors: true, linewidth: 1,
  }), []);

  const edgePositions = useMemo(() => new Float32Array(MAX_RL_AGENTS * EDGE_VERTS_PER_BOX * 3), []);
  const edgeColors    = useMemo(() => new Float32Array(MAX_RL_AGENTS * EDGE_VERTS_PER_BOX * 3), []);
  const fillColors    = useMemo(() => new Float32Array(MAX_RL_AGENTS * 3), []);

  // Pulse state for incident actor
  const pulseRef = useRef(0);

  useFrame(({ clock }) => {
    const fillMesh = fillMeshRef.current;
    const edgeGeo  = edgeGeoRef.current;
    if (!fillMesh || !edgeGeo) return;

    const s = useRLStore.getState();
    if (!s.rlModeActive || !s.rlScene) {
      fillMesh.count = 0;
      edgeGeo.setDrawRange(0, 0);
      return;
    }

    pulseRef.current = 0.55 + 0.45 * Math.abs(Math.sin(clock.elapsedTime * 5));

    const { ego, agents } = s.rlScene;

    // Build agent list: review agent first, then others
    // All positions converted to ego-relative space
    interface AgentEntry { relX: number; relY: number; relZ: number; sx: number; sy: number; sz: number; yaw: number; type: string; }
    const entries: AgentEntry[] = [];

    // Review agent sits at ego position → always (0, 0) relative to itself
    entries.push({
      relX: 0, relY: 0, relZ: ego.height / 2,
      sx: ego.length, sy: ego.width, sz: ego.height,
      yaw: 0,   // ego is always heading forward in its own frame
      type: "review",
    });

    // All other RL agents (incident + traffic) — subtract ego world pos
    for (const ag of agents) {
      entries.push({
        relX: ag.x - ego.x,
        relY: ag.y - ego.y,
        relZ: ag.height / 2,
        sx: ag.length, sy: ag.width, sz: ag.height,
        yaw: ag.yaw - ego.yaw,   // relative heading
        type: ag.type,
      });
    }

    const count = Math.min(entries.length, MAX_RL_AGENTS);
    let edgeVI = 0;

    for (let i = 0; i < count; i++) {
      const e = entries[i];
      const baseColor = agentColor(e.type);

      // Pulse incident actor fill
      const isIncident = e.type === "incident" || e.type === "pedestrian" || e.type === "cyclist";
      _color.copy(baseColor);
      if (isIncident) _color.multiplyScalar(pulseRef.current);

      _pos.set(e.relX, e.relY, e.relZ);
      _euler.set(0, 0, e.yaw);
      _quat.setFromEuler(_euler);
      _scale.set(e.sx, e.sy, e.sz);
      _mat4.compose(_pos, _quat, _scale);
      fillMesh.setMatrixAt(i, _mat4);

      fillColors[i * 3]     = _color.r;
      fillColors[i * 3 + 1] = _color.g;
      fillColors[i * 3 + 2] = _color.b;

      // Wire edges
      for (const edge of UNIT_CUBE_EDGES) {
        for (let v = 0; v < 2; v++) {
          _pos.set(edge[v * 3], edge[v * 3 + 1], edge[v * 3 + 2]);
          _pos.applyMatrix4(_mat4);
          const dst = edgeVI * 3;
          edgePositions[dst]     = _pos.x;
          edgePositions[dst + 1] = _pos.y;
          edgePositions[dst + 2] = _pos.z;
          edgeColors[dst]        = _color.r;
          edgeColors[dst + 1]    = _color.g;
          edgeColors[dst + 2]    = _color.b;
          edgeVI++;
        }
      }
    }

    fillMesh.count = count;
    fillMesh.instanceMatrix.needsUpdate = true;
    if (!fillMesh.instanceColor) {
      fillMesh.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(MAX_RL_AGENTS * 3), 3);
    }
    (fillMesh.instanceColor.array as Float32Array).set(fillColors.subarray(0, count * 3));
    fillMesh.instanceColor.needsUpdate = true;

    const posAttr = edgeGeo.getAttribute("position") as THREE.BufferAttribute;
    const colAttr = edgeGeo.getAttribute("color") as THREE.BufferAttribute;
    (posAttr.array as Float32Array).set(edgePositions.subarray(0, edgeVI * 3));
    (colAttr.array as Float32Array).set(edgeColors.subarray(0, edgeVI * 3));
    posAttr.needsUpdate = true;
    colAttr.needsUpdate = true;
    edgeGeo.setDrawRange(0, edgeVI);
  });

  return (
    <>
      {/* Fill instances */}
      <instancedMesh
        ref={fillMeshRef}
        args={[_boxGeo, fillMat, MAX_RL_AGENTS]}
        frustumCulled={false}
      />

      {/* Edge wireframes */}
      <lineSegments frustumCulled={false} material={edgeMat}>
        <bufferGeometry ref={edgeGeoRef}>
          <bufferAttribute
            attach="attributes-position"
            args={[new Float32Array(MAX_RL_AGENTS * EDGE_VERTS_PER_BOX * 3), 3]}
            usage={THREE.DynamicDrawUsage}
          />
          <bufferAttribute
            attach="attributes-color"
            args={[new Float32Array(MAX_RL_AGENTS * EDGE_VERTS_PER_BOX * 3), 3]}
            usage={THREE.DynamicDrawUsage}
          />
        </bufferGeometry>
      </lineSegments>

      {/* Predicted trajectory for review agent (cyan arc ahead of ego) */}
      <ReviewTrajectory />
    </>
  );
}
