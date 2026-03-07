/**
 * Scene3D — Full-viewport R3F Canvas. This IS the background of the entire app.
 */

import { Canvas } from "@react-three/fiber";
import { OrbitControls, GizmoHelper, GizmoViewport } from "@react-three/drei";
import PointCloud from "./PointCloud";
import BoundingBoxes from "./BoundingBoxes";
import EgoVehicle from "./EgoVehicle";
import { useStore } from "../store";
import { colors } from "../theme";

function SceneContent() {
  const showGrid = useStore((s) => s.showGrid);

  return (
    <>
      <ambientLight intensity={0.35} />
      <directionalLight position={[50, -30, 80]} intensity={0.9} />
      <directionalLight position={[-30, 40, 20]} intensity={0.35} />

      {/* Scene group — vehicle coordinate frame */}
      <group>
        <PointCloud />
        <BoundingBoxes />
        <EgoVehicle />
      </group>

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
    </>
  );
}

export default function Scene3D() {
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
        <SceneContent />
      </Canvas>
    </div>
  );
}
