/* Liya Voice Assistant Service Worker */
const CACHE_NAME = 'liya-cache-v3';
const APP_SHELL = [
  '/manifest.webmanifest',
  '/sw.js',
  'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css',
  'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
  'https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap'
];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => cache.addAll(APP_SHELL)).then(() => self.skipWaiting())
  );
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) => Promise.all(keys.filter(k => k !== CACHE_NAME).map(k => caches.delete(k)))).then(() => self.clients.claim())
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Bypass for non-GET and dev endpoints
  if (request.method !== 'GET') return;

  // Network-first for navigations (HTML pages) to avoid stale UI
  const acceptsHtml = request.headers.get('accept') || '';
  if (request.mode === 'navigate' || acceptsHtml.includes('text/html')) {
    event.respondWith(
      fetch(request)
        .then((resp) => {
          const clone = resp.clone();
          caches.open(CACHE_NAME).then((cache) => cache.put(request, clone));
          return resp;
        })
        .catch(() => caches.match(request).then((cached) => cached || caches.match('/'))) // fallback to shell
    );
    return;
  }

  // Network-first for API calls
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(
      fetch(request).catch(() => caches.match(request))
    );
    return;
  }

  // Cache-first for same-origin static/resources
  if (url.origin === self.location.origin) {
    event.respondWith(
      caches.match(request).then((cached) => cached || fetch(request).then((resp) => {
        const respClone = resp.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(request, respClone));
        return resp;
      }))
    );
    return;
  }

  // For cross-origin resources, try cache, then network
  event.respondWith(
    caches.match(request).then((cached) => cached || fetch(request))
  );
});
