GHCR_IMAGE = ghcr.io/timtegtmeyer/whisperx-worker
BUILDER    = ghcr-builder

.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: build
build: ## Build, push to GHCR, capture digest, tag
	@PREV=$$(git tag --sort=-v:refname | head -1 | sed 's/^v//'); \
	if [ -z "$$PREV" ]; then PREV="0.0.0"; fi; \
	NEW=$$(echo $$PREV | awk -F. '{printf "%d.%d.%d", $$1, $$2, $$3+1}'); \
	echo "Building v$$NEW..."; \
	docker buildx build --builder $(BUILDER) --platform linux/amd64 --push \
		-t $(GHCR_IMAGE):v$$NEW \
		-t $(GHCR_IMAGE):latest \
		. && \
	DIGEST=$$(docker buildx imagetools inspect $(GHCR_IMAGE):v$$NEW --format '{{.Manifest.Digest}}'); \
	echo "$(GHCR_IMAGE)@$$DIGEST" > .last-digest; \
	echo ""; \
	echo "Digest: $(GHCR_IMAGE)@$$DIGEST"; \
	echo ""; \
	git add -A && git commit -m "build: v$$NEW" && git tag v$$NEW

.PHONY: digest
digest: ## Show current image digest for RunPod endpoint config
	@cat .last-digest 2>/dev/null || echo "No digest — run 'make build' first"
