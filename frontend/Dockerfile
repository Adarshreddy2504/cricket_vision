# syntax=docker/dockerfile:1.7

# CricketVision frontend image: build the Vite app, then serve it with an
# unprivileged nginx. nginx also proxies /api to the backend service so the
# browser only needs one origin and uploads keep the same-origin path.

FROM node:22-alpine AS build

WORKDIR /app

COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./

# Empty means "same origin": the browser calls /api/... and nginx proxies it.
ARG VITE_API_URL=""
ENV VITE_API_URL=${VITE_API_URL}
RUN npm run build

FROM nginxinc/nginx-unprivileged:1.27-alpine AS runtime

COPY frontend/nginx.conf /etc/nginx/conf.d/default.conf
COPY --from=build /app/dist /usr/share/nginx/html

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=5 \
    CMD wget -q -O /dev/null http://127.0.0.1:8080/healthz || exit 1

CMD ["nginx", "-g", "daemon off;"]
