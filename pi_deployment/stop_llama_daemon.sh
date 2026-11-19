#!/usr/bin/env bash
# ==============================================================================
# Stop Llama In-Memory Persistent Daemon
# ==============================================================================

PORT=8080
PID_FILE="/tmp/llama_daemon.pid"
CACHE_FILE="/dev/shm/llama_prompt_cache.bin"

STOPPED=false

if [ -f "$PID_FILE" ]; then
    PID="$(cat "$PID_FILE")"
    if kill -0 "$PID" 2>/dev/null; then
        echo "Stopping Llama Daemon (PID: $PID)..."
        kill "$PID" 2>/dev/null || true
        for i in $(seq 1 10); do
            if ! kill -0 "$PID" 2>/dev/null; then
                break
            fi
            sleep 0.3
        done
        # Force kill if still running
        if kill -0 "$PID" 2>/dev/null; then
            kill -9 "$PID" 2>/dev/null || true
        fi
        STOPPED=true
    fi
    rm -f "$PID_FILE"
fi

# Fallback: check if any process is still listening on port 8080
if command -v fuser &>/dev/null; then
    fuser -k "${PORT}/tcp" &>/dev/null || true
elif command -v pkill &>/dev/null; then
    pkill -f llama-server 2>/dev/null || true
fi

# Remove prompt cache file to free up /dev/shm
rm -f "$CACHE_FILE"

if [ "$STOPPED" = true ]; then
    echo "✅ Llama In-Memory Persistent Daemon stopped successfully."
else
    echo "ℹ️ No running Llama Daemon was found on port ${PORT}."
fi
