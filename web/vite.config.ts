import { defineConfig } from "vite";

// base "./" so the built site works from any static host path (GitHub Pages, etc.)
// three.js is its own lazily-loaded chunk (only fetched when the live experiment is, and only if WebGL works), hence the raised warning limit.
export default defineConfig({ base: "./", build: { chunkSizeWarningLimit: 650 } });
