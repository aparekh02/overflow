/**
 * EgoVehicle — Subtle origin marker with forward direction indicator.
 */

import { colors } from "../theme";

export default function EgoVehicle() {
  return (
    <group position={[0, 0, 0]}>
      {/* Origin ring */}
      <mesh rotation={[Math.PI / 2, 0, 0]}>
        <torusGeometry args={[0.4, 0.06, 8, 24]} />
        <meshBasicMaterial color={colors.accent} transparent opacity={0.8} />
      </mesh>
      {/* Center dot */}
      <mesh>
        <sphereGeometry args={[0.12, 12, 12]} />
        <meshBasicMaterial color={colors.accent} />
      </mesh>
      {/* Forward arrow (pointing +X) */}
      <mesh position={[1.3, 0, 0]} rotation={[0, 0, -Math.PI / 2]}>
        <coneGeometry args={[0.15, 0.5, 6]} />
        <meshBasicMaterial color={colors.accent} transparent opacity={0.6} />
      </mesh>
    </group>
  );
}
