#!/usr/bin/env bash
# One-shot: build + push the whisperx-worker image with the chunked-whisper
# source changes, update the RunPod `podquery-transcription` template to
# point at the new digest, and flip WHISPER_CHUNKED_MODE_ENABLED on the
# production tenant container.
#
# Run with HF_TOKEN set (used only at build time by download_models.sh):
#
#     HF_TOKEN=hf_... bash deploy-chunked.sh
#
# Everything is idempotent. Rerun is safe.
set -euo pipefail

cd "$(dirname "$0")"

: "${HF_TOKEN:?HF_TOKEN must be set (your HuggingFace read token)}"
: "${RUNPOD_API_KEY:?RUNPOD_API_KEY must be set (read from the tenant compose env)}"
TEMPLATE_ID="tmo14wwxp5"
VPS_HOST="${VPS_HOST:-212.227.20.188}"
VPS_USER="${VPS_USER:-timniko}"
PROD_TENANT_SLUG="1cdeadb36460"
PROD_COMPOSE="/home/timniko/web/podquery.ai/clients/${PROD_TENANT_SLUG}/docker-compose.yaml"

echo "=== 1/4 build + push whisperx-worker ==="
export HF_TOKEN
make build

DIGEST_LINE="$(cat .last-digest)"
NEW_IMAGE="${DIGEST_LINE}"
echo "new image: ${NEW_IMAGE}"
[[ -n "$NEW_IMAGE" ]] || { echo "no digest captured"; exit 1; }

echo
echo "=== 2/4 update RunPod template ${TEMPLATE_ID} ==="
curl -s -X POST "https://api.runpod.io/graphql?api_key=${RUNPOD_API_KEY}" \
  -H "Content-Type: application/json" \
  -d @- <<JSON | tee /tmp/template-update.json
{"query":"mutation { saveTemplate(input: { id: \"${TEMPLATE_ID}\", name: \"podquery-transcription\", imageName: \"${NEW_IMAGE}\", containerDiskInGb: 60, volumeInGb: 0, dockerArgs: \"\", volumeMountPath: \"/workspace\", isServerless: true, env: [{ key: \"HF_TOKEN\", value: \"{{ RUNPOD_SECRET_HF_TOKEN }}\" }] }) { id name imageName } }"}
JSON
echo

if grep -q '"errors"' /tmp/template-update.json; then
  echo "RunPod template update FAILED — see /tmp/template-update.json"
  exit 1
fi
echo "template updated."

echo
echo "=== 3/4 flip WHISPER_CHUNKED_MODE_ENABLED on prod tenant ${PROD_TENANT_SLUG} ==="
ssh "${VPS_USER}@${VPS_HOST}" bash -s "${PROD_COMPOSE}" "${PROD_TENANT_SLUG}" <<'ENDSSH'
set -euo pipefail
compose="$1"; slug="$2"
if ! grep -q "WHISPER_CHUNKED_MODE_ENABLED" "$compose"; then
  # Add the env var right after RUNPOD_API_KEY so it sits with the other
  # worker-facing env in the same block.
  sed -i "/RUNPOD_API_KEY:/a \      WHISPER_CHUNKED_MODE_ENABLED: '1'" "$compose"
  echo "added WHISPER_CHUNKED_MODE_ENABLED=1 to compose"
else
  sed -i "s|WHISPER_CHUNKED_MODE_ENABLED:.*|WHISPER_CHUNKED_MODE_ENABLED: '1'|" "$compose"
  echo "flipped WHISPER_CHUNKED_MODE_ENABLED to 1 in compose"
fi
docker compose -f "$compose" up -d --no-deps "php-${slug}"
echo "prod tenant php-${slug} recreated."
ENDSSH

echo
echo "=== 4/4 verify ==="
ssh "${VPS_USER}@${VPS_HOST}" "docker exec php-${PROD_TENANT_SLUG} printenv WHISPER_CHUNKED_MODE_ENABLED 2>&1; docker exec php-${PROD_TENANT_SLUG} grep -c 'transcribeChunked' /app/src/Service/PodcastTranscribeService.php || true"

echo
echo "=== DEPLOY COMPLETE ==="
echo "Next episode > 30 min on the prod tenant will exercise the chunked path."
echo "Monitor the pipeline log for [WHISPER] Chunked mode — splitting …-min episode … lines."
