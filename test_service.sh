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
# GROUP E: RESPONSE STRUCTURE TESTS
# =========================================================
echo -e "${BLUE}━━━━ GROUP E: Response Structure Tests ━━━━${NC}"

# E1. Health response has correct structure
echo -e "${YELLOW}[E1] Health response structure${NC}"
RESPONSE_BODY=$(curl -s "$API_BASE/health")
echo "$RESPONSE_BODY" | grep -q '"status"' && echo "$RESPONSE_BODY" | grep -q '"ok"' \
    && pass "Health response has {\"status\": \"ok\"}" \
    || fail "Health response missing expected fields"

# E2. Ingest response has required fields
echo -e "${YELLOW}[E2] Ingest response has user_id, chunks_indexed, message${NC}"
INGEST_RESP=$(curl -s -X POST "$API_BASE/api/ingest" -F "user_id=user1" -F "file=@$USER1_FILE")
echo "$INGEST_RESP" | grep -q '"user_id"' \
    && echo "$INGEST_RESP" | grep -q '"chunks_indexed"' \
    && echo "$INGEST_RESP" | grep -q '"message"' \
    && pass "Ingest response has user_id, chunks_indexed, message" \
    || fail "Ingest response missing required fields"

# E3. Ingest chunks_indexed is a positive number
echo -e "${YELLOW}[E3] chunks_indexed is a positive integer${NC}"
CHUNKS=$(echo "$INGEST_RESP" | grep -o '"chunks_indexed":[0-9]*' | grep -o '[0-9]*$')
[ -n "$CHUNKS" ] && [ "$CHUNKS" -gt 0 ] \
    && pass "chunks_indexed=$CHUNKS (positive integer)" \
    || fail "chunks_indexed is missing or zero"

# E4. Match response has required top-level fields
echo -e "${YELLOW}[E4] Match response structure: user_id, results, analysis${NC}"
match_request "user1" "Python developer" 3
echo "$RESPONSE_BODY" | grep -q '"user_id"' \
    && echo "$RESPONSE_BODY" | grep -q '"results"' \
    && pass "Match response has user_id and results fields" \
    || fail "Match response missing required top-level fields"

# E5. Each result item has role, tools, projects, score fields
echo -e "${YELLOW}[E5] Each result item has role, tools, projects, score${NC}"
echo "$RESPONSE_BODY" | grep -q '"role"' \
    && echo "$RESPONSE_BODY" | grep -q '"tools"' \
    && echo "$RESPONSE_BODY" | grep -q '"projects"' \
    && echo "$RESPONSE_BODY" | grep -q '"score"' \
    && pass "Each result item has role, tools, projects, score" \
    || fail "Result items missing expected fields"

# E6. Score values are present and numeric (match float pattern)
echo -e "${YELLOW}[E6] Score values are valid floats${NC}"
SCORES=$(echo "$RESPONSE_BODY" | grep -o '"score":[0-9.]*')
[ -n "$SCORES" ] \
    && pass "Score fields are present and numeric" \
    || fail "Score field missing or non-numeric"

# E7. user_id in response matches user_id in request
echo -e "${YELLOW}[E7] Response user_id matches request user_id${NC}"
echo "$RESPONSE_BODY" | grep -q '"user_id":"user1"' \
    && pass "Response user_id matches request user_id (user1)" \
    || fail "Response user_id does not match request"

echo ""

# =========================================================
# GROUP F: USER3 LIFECYCLE TESTS
# =========================================================
echo -e "${BLUE}━━━━ GROUP F: User3 Lifecycle Tests ━━━━${NC}"

USER3_FILE="user3_roles.txt"

# F1. user3 match before ingest → 404
echo -e "${YELLOW}[F1] Match user3 before ingestion (expect 404)${NC}"
# Clean any leftover user3 index from a previous run
rm -rf backend/services/rag/vector_store/user3 vector_store/user3 2>/dev/null
match_request "user3" "machine learning model training" 3
[ "$HTTP_STATUS" -eq 404 ] \
    && pass "user3 match before ingest returns 404" \
    || fail "Expected 404 before ingest, got $HTTP_STATUS"

# F2. Ingest user3
echo -e "${YELLOW}[F2] Ingest user3 (Data/ML/QA roles)${NC}"
if [ ! -f "$USER3_FILE" ]; then
    echo -e "  ${YELLOW}SKIP: $USER3_FILE not found${NC}"
else
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$API_BASE/api/ingest" \
      -F "user_id=user3" -F "file=@$USER3_FILE")
    [ "$HTTP_STATUS" -eq 200 ] && pass "user3 ingest returns 200" || fail "user3 ingest failed: $HTTP_STATUS"
fi

# F3. user3 match after ingest → 200 and correct role
echo -e "${YELLOW}[F3] user3 ML query → Data Scientist after ingest${NC}"
match_request "user3" "machine learning model training with scikit-learn and pandas" 3
[ "$HTTP_STATUS" -eq 200 ] && pass "user3 match after ingest returns 200" || fail "Expected 200, got $HTTP_STATUS"
echo "$RESPONSE_BODY" | grep -iE 'Data Scientist|ML Engineer' \
    && pass "user3 results contain ML-related role" \
    || fail "Expected Data Scientist or ML Engineer in results"

# F4. user3 cannot see user1's Backend Engineer
echo -e "${YELLOW}[F4] user3 isolation: cannot see user1's Backend Engineer${NC}"
match_request "user3" "Python FastAPI Docker backend developer" 5
echo "$RESPONSE_BODY" | grep -q "Senior Backend Engineer" \
    && fail "ISOLATION BREACH: user3 found user1's Backend Engineer!" \
    || pass "user3 cannot see user1's Senior Backend Engineer"

# F5. user1 cannot see user3's Data Scientist
echo -e "${YELLOW}[F5] user1 isolation: cannot see user3's Data Scientist${NC}"
match_request "user1" "machine learning model training scikit-learn" 5
echo "$RESPONSE_BODY" | grep -q "Data Scientist" \
    && fail "ISOLATION BREACH: user1 found user3's Data Scientist!" \
    || pass "user1 cannot see user3's Data Scientist"

# F6. user3 self-query: QA Engineer
echo -e "${YELLOW}[F6] user3: Selenium+Pytest query → QA Engineer${NC}"
match_request "user3" "QA tester experienced in Selenium and Pytest" 2
echo "$RESPONSE_BODY" | grep -q "QA Engineer" \
    && pass "user3 correctly retrieves QA Engineer role" \
    || fail "Expected QA Engineer in user3 results"

echo ""

# =========================================================
# GROUP G: SCORING AND ORDERING TESTS
# =========================================================
echo -e "${BLUE}━━━━ GROUP G: Scoring & Ordering Tests ━━━━${NC}"

# G1. Same query returns different top roles for user1 vs user2
echo -e "${YELLOW}[G1] Same query → different roles for user1 vs user2${NC}"
match_request "user1" "developer experienced with databases and containers" 1
TOP_ROLE_U1=$(echo "$RESPONSE_BODY" | grep -o '"role":"[^"]*"' | head -1)
match_request "user2" "developer experienced with databases and containers" 1
TOP_ROLE_U2=$(echo "$RESPONSE_BODY" | grep -o '"role":"[^"]*"' | head -1)
[ "$TOP_ROLE_U1" != "$TOP_ROLE_U2" ] \
    && pass "Same query returns different top roles (user1: $TOP_ROLE_U1, user2: $TOP_ROLE_U2)" \
    || fail "Same query returned identical top role — isolation may be broken"

# G2. top_k=2 returns at most 2 roles
echo -e "${YELLOW}[G2] top_k=2 returns ≤2 roles${NC}"
match_request "user1" "engineer" 2
ROLE_COUNT=$(echo "$RESPONSE_BODY" | grep -o '"role":' | wc -l | tr -d ' ')
[ "$ROLE_COUNT" -le 2 ] \
    && pass "top_k=2 returned $ROLE_COUNT role(s) (within limit)" \
    || fail "Expected ≤2 roles, got $ROLE_COUNT"

# G3. top_k=5 can return up to 5 roles when data supports it
echo -e "${YELLOW}[G3] top_k=5 on user3 (has 3 roles) returns ≤5${NC}"
match_request "user3" "technology specialist" 5
ROLE_COUNT=$(echo "$RESPONSE_BODY" | grep -o '"role":' | wc -l | tr -d ' ')
[ "$ROLE_COUNT" -le 5 ] && [ "$ROLE_COUNT" -ge 1 ] \
    && pass "top_k=5 returned $ROLE_COUNT role(s)" \
    || fail "top_k=5 returned unexpected count: $ROLE_COUNT"

# G4. Scores are non-negative numbers
echo -e "${YELLOW}[G4] All score values are non-negative${NC}"
match_request "user1" "Python developer" 5
NEG_SCORES=$(echo "$RESPONSE_BODY" | grep -o '"score":-[0-9]' | wc -l | tr -d ' ')
[ "$NEG_SCORES" -eq 0 ] \
    && pass "All scores are non-negative" \
    || fail "Found $NEG_SCORES negative score value(s)"

# G5. Results array is non-empty for a valid query
echo -e "${YELLOW}[G5] Results array is non-empty for valid query${NC}"
match_request "user2" "design tools visual interface" 3
RESULT_COUNT=$(echo "$RESPONSE_BODY" | grep -o '"role":' | wc -l | tr -d ' ')
[ "$RESULT_COUNT" -ge 1 ] \
    && pass "Results array is non-empty ($RESULT_COUNT result(s))" \
    || fail "Results array was empty for a valid query"

# G6. Highly specific query returns the most relevant role as top result
echo -e "${YELLOW}[G6] Highly specific query nails exact role for user1${NC}"
match_request "user1" "DevOps engineer AWS Terraform Kubernetes CI/CD" 1
echo "$RESPONSE_BODY" | grep -q "DevOps Specialist" \
    && pass "DevOps Specialist is top result for specific DevOps query" \
    || fail "Expected DevOps Specialist as top result"

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
