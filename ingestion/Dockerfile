FROM golang:1.22-alpine AS builder

WORKDIR /app

# Copy source first so go mod tidy can resolve all imports,
# then tidy downloads deps AND generates go.sum in one step.
COPY go.mod ./
COPY . .
RUN go mod tidy
RUN CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -o /lydian-ingestion ./cmd/

# ── Minimal runtime image ──────────────────────────────────────────────────
FROM alpine:3.19
RUN apk add --no-cache ca-certificates tzdata
WORKDIR /app
COPY --from=builder /lydian-ingestion .
COPY testdata/ ./testdata/

# Non-root user for security
RUN addgroup -S sentinel && adduser -S sentinel -G sentinel
USER sentinel

ENTRYPOINT ["/app/lydian-ingestion"]
