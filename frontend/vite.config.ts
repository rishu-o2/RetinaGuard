import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 3000,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
        rewrite: (path) => path.replace(/^\/api/, ""),
      },
      "^/(auth|screen|screenings|model|config|referral|voice|offline|health)": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
});
