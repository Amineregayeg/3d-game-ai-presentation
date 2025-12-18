"use client";

import { Suspense, useRef, useState, useEffect } from "react";
import { Canvas, useFrame, useThree } from "@react-three/fiber";
import {
  OrbitControls,
  Environment,
  Grid,
  useGLTF,
  Center,
  Html,
  PerspectiveCamera,
} from "@react-three/drei";
import { motion } from "framer-motion";
import {
  Box,
  RotateCcw,
  ZoomIn,
  ZoomOut,
  Download,
  Maximize2,
  Camera,
  Loader2,
} from "lucide-react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Button } from "@/components/ui/button";
import { Badge } from "@/components/ui/badge";
import * as THREE from "three";

// Loading component for Suspense
function Loader() {
  return (
    <Html center>
      <div className="flex flex-col items-center gap-2">
        <Loader2 className="w-8 h-8 text-cyan-400 animate-spin" />
        <span className="text-sm text-slate-400">Loading model...</span>
      </div>
    </Html>
  );
}

// Placeholder cube when no model is loaded
function PlaceholderCube() {
  const meshRef = useRef<THREE.Mesh>(null);

  useFrame((state) => {
    if (meshRef.current) {
      meshRef.current.rotation.y = state.clock.elapsedTime * 0.5;
    }
  });

  return (
    <mesh ref={meshRef}>
      <boxGeometry args={[1, 1, 1]} />
      <meshStandardMaterial
        color="#06b6d4"
        metalness={0.5}
        roughness={0.5}
        wireframe
      />
    </mesh>
  );
}

// Model loader component
function Model({ url }: { url: string }) {
  const { scene } = useGLTF(url);
  const groupRef = useRef<THREE.Group>(null);

  useEffect(() => {
    // Clone the scene to avoid modifying the cached original
    const clonedScene = scene.clone();

    // Calculate bounding box
    const box = new THREE.Box3().setFromObject(clonedScene);
    const center = box.getCenter(new THREE.Vector3());
    const size = box.getSize(new THREE.Vector3());
    const maxDim = Math.max(size.x, size.y, size.z);

    console.log("[Model] Loaded model - size:", size, "maxDim:", maxDim, "center:", center);

    // Handle edge cases
    if (maxDim === 0 || !isFinite(maxDim)) {
      console.warn("[Model] Invalid model dimensions, using default scale");
      return;
    }

    // Scale to fit nicely in view (target size ~2 units)
    const targetSize = 2.5;
    const scale = targetSize / maxDim;

    // Apply transformations to the original scene
    scene.scale.setScalar(scale);
    scene.position.set(-center.x * scale, -center.y * scale, -center.z * scale);

    console.log("[Model] Applied scale:", scale);
  }, [scene]);

  useFrame((state) => {
    if (groupRef.current) {
      groupRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.3) * 0.1;
    }
  });

  return (
    <group ref={groupRef}>
      <primitive object={scene} />
    </group>
  );
}

// Scene content
function SceneContent({
  modelUrl,
  showGrid,
  autoRotate,
}: {
  modelUrl?: string;
  showGrid: boolean;
  autoRotate: boolean;
}) {
  return (
    <>
      <PerspectiveCamera makeDefault position={[4, 3, 4]} fov={50} />
      <OrbitControls
        enableDamping
        dampingFactor={0.05}
        autoRotate={autoRotate}
        autoRotateSpeed={1}
        minDistance={1}
        maxDistance={10}
      />

      {/* Lighting */}
      <ambientLight intensity={0.5} />
      <directionalLight position={[5, 5, 5]} intensity={1} castShadow />
      <directionalLight position={[-5, 5, -5]} intensity={0.5} />

      {/* Environment for reflections */}
      <Environment preset="city" />

      {/* Grid */}
      {showGrid && (
        <Grid
          args={[10, 10]}
          cellSize={0.5}
          cellThickness={0.5}
          cellColor="#1e293b"
          sectionSize={2}
          sectionThickness={1}
          sectionColor="#334155"
          fadeDistance={10}
          fadeStrength={1}
          followCamera={false}
          infiniteGrid
        />
      )}

      {/* Model or Placeholder */}
      <Center>
        <Suspense fallback={<Loader />}>
          {modelUrl ? <Model url={modelUrl} /> : <PlaceholderCube />}
        </Suspense>
      </Center>
    </>
  );
}

// Screenshot capture component
function ScreenshotCapture({
  onCapture,
}: {
  onCapture: (dataUrl: string) => void;
}) {
  const { gl, scene, camera } = useThree();

  useEffect(() => {
    // Expose capture function
    (window as unknown as { captureThreeJS: () => void }).captureThreeJS = () => {
      gl.render(scene, camera);
      const dataUrl = gl.domElement.toDataURL("image/png");
      onCapture(dataUrl);
    };
  }, [gl, scene, camera, onCapture]);

  return null;
}

// Main ThreeJS Viewport Component
interface ThreeJSViewportProps {
  modelUrl?: string;
  isLoading?: boolean;
  onScreenshot?: (dataUrl: string) => void;
  onExport?: (format: "glb" | "fbx") => void;
}

export function ThreeJSViewport({
  modelUrl,
  isLoading = false,
  onScreenshot,
  onExport,
}: ThreeJSViewportProps) {
  const [showGrid, setShowGrid] = useState(true);
  const [autoRotate, setAutoRotate] = useState(false);
  const [isFullscreen, setIsFullscreen] = useState(false);
  const containerRef = useRef<HTMLDivElement>(null);

  const handleScreenshot = () => {
    if ((window as unknown as { captureThreeJS: () => void }).captureThreeJS) {
      (window as unknown as { captureThreeJS: () => void }).captureThreeJS();
    }
  };

  const toggleFullscreen = () => {
    if (!containerRef.current) return;

    if (!isFullscreen) {
      containerRef.current.requestFullscreen?.();
      setIsFullscreen(true);
    } else {
      document.exitFullscreen?.();
      setIsFullscreen(false);
    }
  };

  return (
    <Card className="bg-slate-800/50 border-white/10 backdrop-blur-sm overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="text-base text-white flex items-center gap-2">
          <Box className="w-4 h-4 text-cyan-400" />
          3D Preview
          {isLoading && (
            <Badge variant="outline" className="ml-2 text-[10px] border-cyan-500/50 text-cyan-400">
              <Loader2 className="w-3 h-3 mr-1 animate-spin" />
              Loading
            </Badge>
          )}
          {modelUrl && (
            <Badge variant="outline" className="ml-auto text-[10px] border-emerald-500/50 text-emerald-400">
              Model Loaded
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent className="p-0">
        <div ref={containerRef} className="relative">
          {/* Three.js Canvas */}
          <div className="aspect-video bg-slate-900">
            <Canvas
              shadows
              gl={{ preserveDrawingBuffer: true }}
              style={{ background: "linear-gradient(to bottom, #0f172a, #1e293b)" }}
            >
              <SceneContent
                modelUrl={modelUrl}
                showGrid={showGrid}
                autoRotate={autoRotate}
              />
              {onScreenshot && <ScreenshotCapture onCapture={onScreenshot} />}
            </Canvas>
          </div>

          {/* Controls Overlay */}
          <div className="absolute bottom-0 left-0 right-0 p-3 bg-gradient-to-t from-black/80 to-transparent">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setAutoRotate(!autoRotate)}
                  className={`h-8 w-8 ${autoRotate ? "text-cyan-400" : "text-white"} hover:bg-white/20`}
                  title="Toggle auto-rotate"
                >
                  <RotateCcw className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setShowGrid(!showGrid)}
                  className={`h-8 w-8 ${showGrid ? "text-cyan-400" : "text-white"} hover:bg-white/20`}
                  title="Toggle grid"
                >
                  <svg className="w-4 h-4" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M3 3h18v18H3zM3 9h18M3 15h18M9 3v18M15 3v18" />
                  </svg>
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={handleScreenshot}
                  className="h-8 w-8 text-white hover:bg-white/20"
                  title="Take screenshot"
                >
                  <Camera className="w-4 h-4" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={toggleFullscreen}
                  className="h-8 w-8 text-white hover:bg-white/20"
                  title="Toggle fullscreen"
                >
                  <Maximize2 className="w-4 h-4" />
                </Button>
              </div>

              {onExport && modelUrl && (
                <div className="flex items-center gap-1">
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onExport("glb")}
                    className="h-7 text-xs border-white/20 text-white hover:bg-white/10"
                  >
                    <Download className="w-3 h-3 mr-1" />
                    GLB
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    onClick={() => onExport("fbx")}
                    className="h-7 text-xs border-white/20 text-white hover:bg-white/10"
                  >
                    <Download className="w-3 h-3 mr-1" />
                    FBX
                  </Button>
                </div>
              )}
            </div>
          </div>

          {/* Loading Overlay */}
          {isLoading && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="absolute inset-0 flex items-center justify-center bg-slate-900/80 backdrop-blur-sm"
            >
              <div className="text-center">
                <Loader2 className="w-10 h-10 text-cyan-400 animate-spin mx-auto mb-3" />
                <p className="text-slate-400">Loading 3D model...</p>
              </div>
            </motion.div>
          )}
        </div>
      </CardContent>
    </Card>
  );
}
