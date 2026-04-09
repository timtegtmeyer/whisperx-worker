.DEFAULT_GOAL := help

.PHONY: help
help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

.PHONY: release
release: ## Create a versioned release (prompts for version, e.g. 1.0.1)
	@echo ""; \
	echo "Current tags:"; \
	git tag --sort=-v:refname | head -5 || echo "  (none)"; \
	echo ""; \
	read -p "Version to release (e.g. 1.0.1): " version; \
	if [ -z "$$version" ]; then echo "Aborted."; exit 1; fi; \
	tag="v$$version"; \
	echo ""; \
	echo "This will:"; \
	echo "  1. Create git tag $$tag"; \
	echo "  2. Push to origin (triggers Docker image build)"; \
	echo "  3. Publish ghcr.io/timtegtmeyer/whisperx-worker:$$version"; \
	echo ""; \
	read -p "Continue? [y/N] " confirm; \
	if [ "$$confirm" != "y" ] && [ "$$confirm" != "Y" ]; then echo "Aborted."; exit 1; fi; \
	git tag "$$tag" && \
	git push origin "$$tag" && \
	echo "" && \
	echo "✓ Tag $$tag pushed. Image will be available at:" && \
	echo "  ghcr.io/timtegtmeyer/whisperx-worker:$$version"
