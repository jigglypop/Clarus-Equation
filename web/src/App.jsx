import { useEffect, useState } from "react";
import init, { run_viewer, load_gltf } from "./wasm_viewer_web/viewer.js";

export function App() {
  const [ready, setReady] = useState(false);
  const [modelLoaded, setModelLoaded] = useState(false);

  useEffect(() => {
    let cancelled = false;
    const run = async () => {
      await init();
      if (cancelled) return;
      run_viewer("sfe-canvas");
      setReady(true);
    };
    run();
    return () => {
      cancelled = true;
    };
  }, []);

  const loadModel = async (url) => {
    try {
      const response = await fetch(url);
      const arrayBuffer = await response.arrayBuffer();
      const data = new Uint8Array(arrayBuffer);
      load_gltf(data);
      setModelLoaded(true);
    } catch (e) {
      console.error("Failed to load model:", e);
    }
  };

  return (
    <div style={{ width: "100vw", height: "100vh", position: "relative", background: "#000" }}>
      <canvas
        id="sfe-canvas"
        width={1280}
        height={720}
        style={{ width: "100%", height: "100%", display: "block" }}
      />
      {ready && (
        <div style={{
          position: "absolute",
          top: 10,
          left: 10,
          display: "flex",
          gap: 8,
          flexWrap: "wrap",
        }}>
          <button 
            onClick={() => loadModel("/models/Box.glb")}
            style={{
              padding: "8px 16px",
              background: "#48f",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 14,
            }}
          >
            Box
          </button>
          <button 
            onClick={() => loadModel("/models/Fox.glb")}
            style={{
              padding: "8px 16px",
              background: modelLoaded ? "#2a5" : "#48f",
              color: "#fff",
              border: "none",
              borderRadius: 4,
              cursor: "pointer",
              fontSize: 14,
            }}
          >
            Fox
          </button>
          <span style={{ color: "#fff", fontSize: 12, alignSelf: "center" }}>
            SFE Engine v0.2 | Drag to rotate | Scroll to zoom
          </span>
        </div>
      )}
    </div>
  );
}
