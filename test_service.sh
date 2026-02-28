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
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

PASS_COUNT=0
FAIL_COUNT=0

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
GROQ_API_KEY="$GROQ_API_KEY" $UVICORN_CMD app.main:app --host 0.0.0.0 --port 8000 > "$LOG_FILE" 2>&1 &
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

# --- Test Helpers ---

pass() {
    echo -e "  ${GREEN}✅ PASS: $1${NC}"
    PASS_COUNT=$((PASS_COUNT + 1))
}

fail() {
    echo -e "  ${RED}❌ FAIL: $1${NC}"
    FAIL_COUNT=$((FAIL_COUNT + 1))
}

make_request() {
    # $1=method, $2=url, $3=extra curl args (optional body/form)
    # Sets globals: HTTP_STATUS, RESPONSE_BODY
    RESPONSE_DATA=$(eval curl -s -w "\\n%{http_code}" "$1" "$2" "$3" 2>/dev/null)
    RESPONSE_BODY=$(echo "$RESPONSE_DATA" | sed '$d')
    HTTP_STATUS=$(echo "$RESPONSE_DATA" | tail -n 1)
}

match_request() {
    # $1=user_id, $2=query, $3=top_k
    RESPONSE_DATA=$(curl -s -w "\n%{http_code}" -X POST "$API_BASE/api/match" \
      -H "Content-Type: application/json" \
      -d "{\"user_id\": \"$1\", \"query\": \"$2\", \"top_k\": $3}")
    RESPONSE_BODY=$(echo "$RESPONSE_DATA" | sed '$d')
    HTTP_STATUS=$(echo "$RESPONSE_DATA" | tail -n 1)
}

# =========================================================
# GROUP A: INGESTION TESTS
# =========================================================
echo -e "${BLUE}━━━━ GROUP A: Ingestion Tests ━━━━${NC}"

# A1. Ingest User 1
echo -e "${YELLOW}[A1] Ingest roles for User 1 (Backend/DevOps)${NC}"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/ingest" \
  -F "user_id=user1" -F "file=@$USER1_FILE")
[ "$HTTP_STATUS" -eq 200 ] && pass "User 1 ingest returns HTTP 200" || fail "User 1 ingest should return 200, got $HTTP_STATUS"

# A2. Ingest User 2
echo -e "${YELLOW}[A2] Ingest roles for User 2 (Frontend/Design)${NC}"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/ingest" \
  -F "user_id=user2" -F "file=@$USER2_FILE")
[ "$HTTP_STATUS" -eq 200 ] && pass "User 2 ingest returns HTTP 200" || fail "User 2 ingest should return 200, got $HTTP_STATUS"

# A3. Re-ingest User 1 (should overwrite, not error)
echo -e "${YELLOW}[A3] Re-ingest User 1 (overwrite)${NC}"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/ingest" \
  -F "user_id=user1" -F "file=@$USER1_FILE")
[ "$HTTP_STATUS" -eq 200 ] && pass "Re-ingest returns HTTP 200 (not error)" || fail "Re-ingest should return 200, got $HTTP_STATUS"

# A4. Ingest with missing file (should 422)
echo -e "${YELLOW}[A4] Ingest with no file attached (expect 422)${NC}"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/ingest" \
  -F "user_id=user_no_file")
[ "$HTTP_STATUS" -eq 422 ] && pass "Missing file returns HTTP 422" || fail "Missing file should return 422, got $HTTP_STATUS"

# A5. Ingest with missing user_id (should 422)
echo -e "${YELLOW}[A5] Ingest with no user_id (expect 422)${NC}"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/ingest" \
  -F "file=@$USER1_FILE")
[ "$HTTP_STATUS" -eq 422 ] && pass "Missing user_id returns HTTP 422" || fail "Missing user_id should return 422, got $HTTP_STATUS"

echo ""

# =========================================================
# GROUP B: MATCH — ACCURACY TESTS
# =========================================================
echo -e "${BLUE}━━━━ GROUP B: Retrieval Accuracy Tests ━━━━${NC}"

# B1. User 1 correctly matches Backend role
echo -e "${YELLOW}[B1] User 1: Python+Docker query → Backend Engineer${NC}"
match_request "user1" "Senior Python Developer with Docker" 1
[ "$HTTP_STATUS" -eq 200 ] && pass "Returns HTTP 200" || fail "Should return 200, got $HTTP_STATUS"
echo "$RESPONSE_BODY" | grep -q "Senior Backend Engineer" \
    && pass "Top result is 'Senior Backend Engineer'" \
    || fail "Expected 'Senior Backend Engineer' in results"

# B2. User 2 correctly matches Frontend role
echo -e "${YELLOW}[B2] User 2: React+Next.js query → Frontend Developer${NC}"
match_request "user2" "Looking for a React developer with Next.js skills" 1
[ "$HTTP_STATUS" -eq 200 ] && pass "Returns HTTP 200" || fail "Should return 200, got $HTTP_STATUS"
echo "$RESPONSE_BODY" | grep -q "Lead Frontend Developer" \
    && pass "Top result is 'Lead Frontend Developer'" \
    || fail "Expected 'Lead Frontend Developer' in results"

# B3. User 2 correctly matches UI/UX Designer
echo -e "${YELLOW}[B3] User 2: Figma+design query → UI/UX Designer${NC}"
match_request "user2" "We need a UI designer experienced in Figma and design systems" 2
[ "$HTTP_STATUS" -eq 200 ] && pass "Returns HTTP 200" || fail "Should return 200, got $HTTP_STATUS"
echo "$RESPONSE_BODY" | grep -q "UI/UX Designer" \
    && pass "Results contain 'UI/UX Designer'" \
    || fail "Expected 'UI/UX Designer' in results"

# B4. User 1: AWS/Kubernetes query → DevOps Specialist
echo -e "${YELLOW}[B4] User 1: Kubernetes+AWS query → DevOps Specialist${NC}"
match_request "user1" "Seeking a DevOps engineer skilled in Kubernetes and AWS infrastructure" 1
[ "$HTTP_STATUS" -eq 200 ] && pass "Returns HTTP 200" || fail "Should return 200, got $HTTP_STATUS"
echo "$RESPONSE_BODY" | grep -q "DevOps Specialist" \
    && pass "Top result is 'DevOps Specialist'" \
    || fail "Expected 'DevOps Specialist' in results"

# B5. top_k=1 returns exactly one result
echo -e "${YELLOW}[B5] top_k=1 constraint${NC}"
match_request "user1" "Python developer" 1
RESULT_COUNT=$(echo "$RESPONSE_BODY" | grep -o '"role"' | wc -l)
[ "$HTTP_STATUS" -eq 200 ] && pass "Returns HTTP 200" || fail "Should return 200, got $HTTP_STATUS"
[ "$RESULT_COUNT" -le 1 ] \
    && pass "Returned at most 1 unique role (top_k=1)" \
    || fail "Expected 1 role, got $RESULT_COUNT"

echo ""

# =========================================================
# GROUP C: ISOLATION TESTS
# =========================================================
echo -e "${BLUE}━━━━ GROUP C: Multi-User Isolation Tests ━━━━${NC}"

# C1. User 2 cannot see User 1's Backend Engineer
echo -e "${YELLOW}[C1] User 2 cannot access User 1's Backend Engineer${NC}"
match_request "user2" "Senior Python Developer with Docker" 5
echo "$RESPONSE_BODY" | grep -q "Senior Backend Engineer" \
    && fail "ISOLATION BREACH: User 2 found User 1's Backend Engineer!" \
    || pass "User 2 cannot see User 1's Backend Engineer"

# C2. User 1 cannot see User 2's Frontend Developer
echo -e "${YELLOW}[C2] User 1 cannot access User 2's Frontend Developer${NC}"
match_request "user1" "React developer with Next.js" 5
echo "$RESPONSE_BODY" | grep -q "Lead Frontend Developer" \
    && fail "ISOLATION BREACH: User 1 found User 2's Frontend Developer!" \
    || pass "User 1 cannot see User 2's Lead Frontend Developer"

# C3. User 1 cannot see User 2's UI/UX Designer
echo -e "${YELLOW}[C3] User 1 cannot access User 2's UI/UX Designer${NC}"
match_request "user1" "Figma UI designer" 5
echo "$RESPONSE_BODY" | grep -q "UI/UX Designer" \
    && fail "ISOLATION BREACH: User 1 found User 2's UI/UX Designer!" \
    || pass "User 1 cannot see User 2's UI/UX Designer"

echo ""

# =========================================================
# GROUP D: EDGE CASES
# =========================================================
echo -e "${BLUE}━━━━ GROUP D: Edge Case Tests ━━━━${NC}"

# D1. Match for non-existent user → 404
echo -e "${YELLOW}[D1] Match for non-existent user (expect 404)${NC}"
match_request "ghost_user_999" "any query" 5
[ "$HTTP_STATUS" -eq 404 ] \
    && pass "Non-existent user returns HTTP 404" \
    || fail "Expected 404, got $HTTP_STATUS"

# D2. Match with empty query (expect 422)
echo -e "${YELLOW}[D2] Match with empty query string (expect 422)${NC}"
RESPONSE_DATA=$(curl -s -w "\n%{http_code}" -X POST "$API_BASE/api/match" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "query": "", "top_k": 5}')
RESPONSE_BODY=$(echo "$RESPONSE_DATA" | sed '$d')
HTTP_STATUS=$(echo "$RESPONSE_DATA" | tail -n 1)
[ "$HTTP_STATUS" -eq 422 ] \
    && pass "Empty query returns HTTP 422" \
    || fail "Expected 422 for empty query, got $HTTP_STATUS"

# D3. Match with missing body fields (expect 422)
echo -e "${YELLOW}[D3] Match with missing user_id (expect 422)${NC}"
RESPONSE_DATA=$(curl -s -w "\n%{http_code}" -X POST "$API_BASE/api/match" \
  -H "Content-Type: application/json" \
  -d '{"query": "test query", "top_k": 5}')
HTTP_STATUS=$(echo "$RESPONSE_DATA" | tail -n 1)
[ "$HTTP_STATUS" -eq 422 ] \
    && pass "Missing user_id returns HTTP 422" \
    || fail "Expected 422 for missing user_id, got $HTTP_STATUS"

# D4. Match with malformed JSON body (expect 422)
echo -e "${YELLOW}[D4] Match with malformed JSON (expect 422)${NC}"
HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/match" \
  -H "Content-Type: application/json" \
  -d 'NOT_VALID_JSON')
[ "$HTTP_STATUS" -eq 422 ] \
    && pass "Malformed JSON returns HTTP 422" \
    || fail "Expected 422 for malformed JSON, got $HTTP_STATUS"

# D5. top_k=0 (edge boundary — expect 422 or graceful handling)
echo -e "${YELLOW}[D5] Match with top_k=0 (boundary value)${NC}"
RESPONSE_DATA=$(curl -s -w "\n%{http_code}" -X POST "$API_BASE/api/match" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user1", "query": "Python developer", "top_k": 0}')
HTTP_STATUS=$(echo "$RESPONSE_DATA" | tail -n 1)
RESPONSE_BODY=$(echo "$RESPONSE_DATA" | sed '$d')
([ "$HTTP_STATUS" -eq 200 ] || [ "$HTTP_STATUS" -eq 422 ]) \
    && pass "top_k=0 handled gracefully (HTTP $HTTP_STATUS)" \
    || fail "top_k=0 crashed — got HTTP $HTTP_STATUS"

echo ""

# =========================================================
# SUMMARY
# =========================================================
TOTAL=$((PASS_COUNT + FAIL_COUNT))
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BLUE}  Test Summary: ${GREEN}$PASS_COUNT passed${NC} / ${RED}$FAIL_COUNT failed${NC} / $TOTAL total"
echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

if [ $FAIL_COUNT -eq 0 ]; then
    echo -e "\n${GREEN}=== All $TOTAL Tests Passed Successfully! ===${NC}"
else
    echo -e "\n${RED}=== $FAIL_COUNT test(s) failed. Check uvicorn.log for server-side details. ===${NC}"
    exit 1
fi
