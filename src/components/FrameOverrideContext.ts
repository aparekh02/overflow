/**
 * When provided inside an R3F Canvas tree, overrides the global store's
 * currentFrame for PointCloud and BoundingBoxes. This allows dashboard
 * tiles to render independent scene data.
 */

import { createContext } from "react";
import type { FrameData } from "../mockData";

export const FrameOverrideContext = createContext<FrameData | null>(null);
