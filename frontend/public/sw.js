/**
 * RetinaGuard Offline Service Worker.
 * Caches static shell, active ONNX model, and model metadata for offline operation.
 */

const CACHE_NAME = "retinaguard-model-cache-v1";
const STATIC_CACHE = "retinaguard-static-v1";

const ASSETS_TO_CACHE = [
  "/",
  "/index.html",
  "/manifest.json",
];

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(STATIC_CACHE).then((cache) => cache.addAll(ASSETS_TO_CACHE).catch(() => {}))
  );
  self.skipWaiting();
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(
        keys.map((key) => {
          if (key !== CACHE_NAME && key !== STATIC_CACHE) {
            return caches.delete(key);
          }
        })
      )
    )
  );
  self.clients.claim();
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);

  // Model artifacts cache strategy: Cache-first with network background revalidation
  if (url.pathname.startsWith("/model/")) {
    event.respondWith(
      caches.open(CACHE_NAME).then(async (cache) => {
        const cached = await cache.match(event.request);
        if (cached) {
          // Revalidate in background if online
          fetch(event.request)
            .then((fresh) => {
              if (fresh.ok) cache.put(event.request, fresh);
            })
            .catch(() => {});
          return cached;
        }
        try {
          const fresh = await fetch(event.request);
          if (fresh.ok) {
            cache.put(event.request, fresh.clone());
          }
          return fresh;
        } catch (err) {
          if (cached) return cached;
          throw err;
        }
      })
    );
    return;
  }

  // General request strategy
  event.respondWith(
    caches.match(event.request).then((response) => {
      return response || fetch(event.request).catch(() => {
        // Offline fallback
        if (event.request.mode === "navigate") {
          return caches.match("/");
        }
      });
    })
  );
});
