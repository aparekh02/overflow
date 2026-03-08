/**
 * Scene3D — Full-viewport R3F Canvas. This IS the background of the entire app.
 *
 * Accepts an optional `offset` prop for counterfactual tiles:
 *   When provided, the world (point cloud + bounding boxes) is shifted
 *   to simulate viewing from a different ego position, while the ego
 *   vehicle stays centered. This makes each dashboard tile look distinct.
 *
 * Also accepts an optional `trail` — an array of [x,y] points to draw
 * the counterfactual ego trajectory as a colored line.
 */

import { useRef, useMemo } from "react";
import * as THREE from "three";
import { Canvas, useFrame } from "@react-three/fiber";
import { OrbitControls, GizmoHelper, GizmoViewport, Line } from "@react-three/drei";
import PointCloud from "./PointCloud";
import BoundingBoxes from "./BoundingBoxes";
import EgoVehicle from "./EgoVehicle";
import { useStore } from "../store";
import { colors } from "../theme";
import type { FrameData } from "../mockData";
import { FrameOverrideContext } from "./FrameOverrideContext";

// ---------------------------------------------------------------------------
// Scene offset type — used by counterfactual tiles in the dashboard
// ---------------------------------------------------------------------------

export interface SceneOffset {
  dx: number; // world shift X (meters)
  dy: number; // world shift Y (meters)
  dYaw: number; // world rotation Z (radians)
}

// ---------------------------------------------------------------------------
// Counterfactual trail — renders the divergent ego path
// ---------------------------------------------------------------------------

function CounterfactualTrail({
  trail,
  trailColor,
}: {
  trail: [number, number, number][];
  trailColor: string;
}) {
  if (trail.length < 2) return null;

  return (
    <>
      {/* Trajectory line */}
      <Line
        points={trail}
        color={trailColor}
        lineWidth={3}
        transparent
        opacity={0.8}
      />
      {/* Ghost ego marker at the end of the trail */}
      <group position={trail[trail.length - 1]}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <torusGeometry args={[0.8, 0.08, 8, 24]} />
          <meshBasicMaterial color={trailColor} transparent opacity={0.7} />
        </mesh>
        <mesh>
          <sphereGeometry args={[0.2, 12, 12]} />
          <meshBasicMaterial color={trailColor} />
        </mesh>
      </group>
    </>
  );
}

// ---------------------------------------------------------------------------
// Animated offset group — smoothly interpolates the world offset
// ---------------------------------------------------------------------------

function OffsetGroup({
  offset,
  children,
}: {
  offset: SceneOffset | null;
  children: React.ReactNode;
}) {
  const groupRef = useRef<THREE.Group>(null);
  const targetPos = useMemo(() => new THREE.Vector3(), []);
  const currentPos = useMemo(() => new THREE.Vector3(), []);
  const targetYaw = useRef(0);
  const currentYaw = useRef(0);

  useFrame(() => {
    if (!groupRef.current) return;
    // Target position
    targetPos.set(
      offset ? -offset.dx : 0,
      offset ? -offset.dy : 0,
      0,
    );
    targetYaw.current = offset ? -offset.dYaw : 0;

    // Smooth interpolation
    currentPos.lerp(targetPos, 0.08);
    currentYaw.current += (targetYaw.current - currentYaw.current) * 0.08;

    groupRef.current.position.copy(currentPos);
    groupRef.current.rotation.set(0, 0, currentYaw.current);
  });

  return <group ref={groupRef}>{children}</group>;
}

// ---------------------------------------------------------------------------
// Scene content — accepts offset + trail as direct props (not context!)
// R3F Canvas creates a separate React tree, so context doesn't cross.
// ---------------------------------------------------------------------------

function SceneContent({
  offset,
  trail,
  trailColor,
  frameOverride,
}: {
  offset: SceneOffset | null;
  trail: [number, number, number][] | null;
  trailColor: string;
  frameOverride: FrameData | null;
}) {
  const showGrid = useStore((s) => s.showGrid);

  return (
    <FrameOverrideContext.Provider value={frameOverride}>
      <ambientLight intensity={0.35} />
      <directionalLight position={[50, -30, 80]} intensity={0.9} />
      <directionalLight position={[-30, 40, 20]} intensity={0.35} />

      {/* World group — smoothly shifted for counterfactual perspective */}
      <OffsetGroup offset={offset}>
        <PointCloud />
        <BoundingBoxes />
      </OffsetGroup>

      {/* Ego stays at origin regardless of offset */}
      <EgoVehicle />

      {/* Counterfactual trajectory trail */}
      {trail && trail.length >= 2 && (
        <CounterfactualTrail trail={trail} trailColor={trailColor} />
      )}

      {/* Ground grid */}
      {showGrid && (
        <gridHelper
          args={[300, 60, "#1E2440", "#161A30"]}
          rotation={[Math.PI / 2, 0, 0]}
        />
      )}

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={0.08}
        minDistance={3}
        maxDistance={250}
        target={[8, 0, 0]}
        zoomSpeed={1.2}
        rotateSpeed={0.6}
      />

      <GizmoHelper alignment="bottom-right" margin={[56, 72]}>
        <GizmoViewport
          axisColors={[colors.gizmoX, colors.gizmoY, colors.gizmoZ]}
          labelColor="white"
        />
      </GizmoHelper>
    </FrameOverrideContext.Provider>
  );
}

// ---------------------------------------------------------------------------
// Exported component
// ---------------------------------------------------------------------------

export default function Scene3D({
  offset,
  trail,
  trailColor = "#00e89d",
  frameOverride,
}: {
  offset?: SceneOffset;
  trail?: [number, number, number][];
  trailColor?: string;
  frameOverride?: FrameData | null;
}) {
  return (
    <div style={{ position: "absolute", inset: 0 }}>
      <Canvas
        camera={{
          position: [-20, -8, 18],
          fov: 55,
          near: 0.1,
          far: 600,
          up: [0, 0, 1],
        }}
        gl={{ antialias: false, powerPreference: "high-performance" }}
        style={{ width: "100%", height: "100%" }}
        onCreated={({ gl }) => {
          gl.setClearColor("#080B14");
        }}
      >
        <SceneContent
          offset={offset ?? null}
          trail={trail ?? null}
          trailColor={trailColor}
          frameOverride={frameOverride ?? null}
        />
      </Canvas>
    </div>
  );
}
