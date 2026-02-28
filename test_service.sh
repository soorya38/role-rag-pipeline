#!/bin/bash

# --- Configuration ---
API_BASE="http://localhost:8000"
USER1_FILE="user1_roles.txt"
USER2_FILE="user2_roles.txt"
APP_DIR="backend/services/rag"

# --- Colors for Output ---
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}=== Role RAG Multi-User Test Script ===${NC}\n"

# 0. Check GROQ_API_KEY
if [ -z "$GROQ_API_KEY" ]; then
    echo -e "${RED}Error: GROQ_API_KEY is not set.${NC}"
    echo -e "Please run: ${BLUE}export GROQ_API_KEY=your_key_here${NC} before running this script."
    exit 1
fi
echo -e "${GREEN}GROQ_API_KEY is set.${NC}"

# 1. Check if dummy files exist
for f in "$USER1_FILE" "$USER2_FILE"; do
    if [ ! -f "$f" ]; then
        echo -e "${RED}Error: $f not found in current directory.${NC}"
        exit 1
    fi
done

ROOT_DIR=$(pwd)

# --- Start Uvicorn Server ---
echo -e "${BLUE}Starting Uvicorn Server in background...${NC}"

LOG_FILE="$ROOT_DIR/uvicorn.log"
rm -f "$LOG_FILE"

# Identify where uvicorn is (prefer local venv)
UVICORN_CMD="uvicorn"
if [ -f "$ROOT_DIR/venv/bin/uvicorn" ]; then
    UVICORN_CMD="$ROOT_DIR/venv/bin/uvicorn"
elif command -v python3 &>/dev/null; then
    UVICORN_CMD="python3 -m uvicorn"
fi

cd "$APP_DIR" || exit 1
# Start uvicorn and redirect output to a log file
$UVICORN_CMD app.main:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
UVICORN_PID=$!
cd "$ROOT_DIR" || exit 1

# Ensure uvicorn is killed on script exit
cleanup() {
    echo -e "\n${BLUE}Cleaning up... Killing Uvicorn (PID: $UVICORN_PID)${NC}"
    kill "$UVICORN_PID" 2>/dev/null
}
trap cleanup EXIT

# --- Wait for Server to be Healthy ---
echo -n -e "${BLUE}Waiting for server to become healthy...${NC}"
MAX_RETRIES=30
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s "$API_BASE/health" | grep -q "ok"; then
        echo -e "\n${GREEN}Service is UP and Healthy!${NC}\n"
        break
    fi
    echo -n "."
    sleep 1
    RETRY_COUNT=$((RETRY_COUNT + 1))
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "\n${RED}Error: Server failed to become healthy within $MAX_RETRIES seconds.${NC}"
    cat "$LOG_FILE"
    exit 1
fi

# --- Helper: Check HTTP Status ---
check_status() {
    local status=$1
    local expected=$2
    local msg=$3
    if [ "$status" -ne "$expected" ]; then
        echo -e "${RED}FAIL: $msg (HTTP $status)${NC}"
        echo -e "Check ${BLUE}uvicorn.log${NC} for server-side error details."
        exit 1
    fi
}

# --- Run Tests ---

# 2. Ingest User 1
echo -e "${BLUE}[2/5] Ingesting roles for User 1 (Backend/DevOps)...${NC}"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/ingest" \
  -F "user_id=user1" \
  -F "file=@$USER1_FILE")
check_status "$HTTP_STATUS" 200 "User 1 Ingestion"
echo -e "${GREEN}User 1 Ingested${NC}\n"

# 3. Ingest User 2
echo -e "${BLUE}[3/5] Ingesting roles for User 2 (Frontend/Design)...${NC}"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/ingest" \
  -F "user_id=user2" \
  -F "file=@$USER2_FILE")
check_status "$HTTP_STATUS" 200 "User 2 Ingestion"
echo -e "${GREEN}User 2 Ingested${NC}\n"

# 4. Match Test (Isolation Check)
echo -e "${BLUE}[4/5] Testing Isolation: Searching User 2 for a Backend Role...${NC}"
RESPONSE_DATA=$(curl -s -w "\n%{http_code}" -X POST "$API_BASE/api/match" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user2",
    "query": "Senior Python Developer with Docker",
    "top_k": 3
  }')

HTTP_STATUS=$(echo "$RESPONSE_DATA" | tail -n 1)
RESPONSE_BODY=$(echo "$RESPONSE_DATA" | head -n -1)

check_status "$HTTP_STATUS" 200 "Search for User 2"

echo -e "Query: 'Senior Python Developer with Docker'"
if echo "$RESPONSE_BODY" | grep -q "Senior Backend Engineer"; then
    echo -e "${RED}FAIL: User 2 accessed User 1's data!${NC}"
    exit 1
else
    echo -e "${GREEN}PASS: Isolation verified. User 1's data not found in User 2's search.${NC}"
fi
echo ""

# 5. Success Check
echo -e "${BLUE}[5/5] Testing Success: Searching User 1 for Backend Role...${NC}"
RESPONSE_DATA=$(curl -s -w "\n%{http_code}" -X POST "$API_BASE/api/match" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user1",
    "query": "Senior Python Developer with Docker",
    "top_k": 1
  }')

HTTP_STATUS=$(echo "$RESPONSE_DATA" | tail -n 1)
RESPONSE_BODY=$(echo "$RESPONSE_DATA" | head -n -1)

check_status "$HTTP_STATUS" 200 "Search for User 1"

if echo "$RESPONSE_BODY" | grep -q "Senior Backend Engineer"; then
    echo -e "${GREEN}PASS: Successfully matched Backend Engineer for User 1.${NC}"
else
    echo -e "${RED}FAIL: Could not find matching role in User 1's index response.${NC}"
    exit 1
fi

echo -e "\n${BLUE}=== All Tests Passed Successfully! ===${NC}"
