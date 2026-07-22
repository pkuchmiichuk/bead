# ============================================================================
# Argument Structure Pipeline - Makefile (Ukrainian)
# ============================================================================
#
# Quick start:
#   make help          - List targets
#   make data          - Run the experiment
#   make data-quick    - Smoke test over a few verbs
#   make deployment    - Build the jsPsych/JATOS experiments
#
# ============================================================================

PROJECT_ROOT := $(shell cd ../../.. && pwd)
PYTHON       := uv run --project $(PROJECT_ROOT) python
# build_frequencies.py is the only script needing wordfreq, supplied per call
# so the pipeline itself carries no such dependency.
PYTHON_FREQ  := uv run --project $(PROJECT_ROOT) --with wordfreq python
JSPSYCH_DIR  := $(PROJECT_ROOT)/bead/deployment/jspsych

LEXICONS_DIR := lexicons
TEMPLATES_DIR := templates
ITEMS_DIR    := items
LISTS_DIR    := lists
CACHE_DIR    := .cache

# How many verbs the experiment tests, chosen by frequency since VESUM is
# ordered alphabetically. At 100 each verb still contributes its six case
# contrasts across the configured lists.
VERBS        := 100
QUICK_VERBS  := 20

BLUE   := \033[0;34m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m

# ============================================================================
# Main targets
# ============================================================================

.PHONY: help
help: ## List available targets
	@echo "$(BLUE)Argument Structure Pipeline (Ukrainian)$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z0-9_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-22s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(BLUE)Examples:$(NC)"
	@echo "  make data                 # the experiment ($(VERBS) verbs)"
	@echo "  make data-quick           # smoke test ($(QUICK_VERBS) verbs)"
	@echo "  make deployment           # every configured list"
	@echo "  make deployment-quick     # 2 lists, for local testing"
	@echo "  make frequencies          # regenerate the committed frequency table"
	@echo ""
	@echo "$(RED)clean-cache and clean-all discard the cached model scores,$(NC)"
	@echo "$(RED)so the next 2afc-pairs run rescores every sentence.$(NC)"
	@echo ""

.PHONY: all
all: data deployment ## Run the pipeline end to end
	@echo "$(GREEN)Pipeline complete$(NC)"

.PHONY: data
data: lexicons templates fill-templates 2afc-pairs lists ## Generate all data for the experiment
	@echo "$(GREEN)Data generated$(NC)"

.PHONY: data-quick
data-quick: lexicons templates fill-templates-quick 2afc-pairs lists ## Generate all data over a few verbs
	@echo "$(GREEN)Data generated$(NC)"

# ============================================================================
# Pipeline stages
# ============================================================================

.PHONY: lexicons
lexicons: ## [1/5] Build the verb and bleached noun lexicons from VESUM
	@echo "$(BLUE)[1/5] Lexicons$(NC)"
	$(PYTHON) generate_lexicons.py

.PHONY: frequencies
frequencies: ## Regenerate resources/verb_frequencies.csv (needs wordfreq, run rarely)
	@echo "$(BLUE)Verb frequencies$(NC)"
	$(PYTHON_FREQ) build_frequencies.py

.PHONY: templates
templates: ## [2/5] Build the five sentence frames
	@echo "$(BLUE)[2/5] Templates$(NC)"
	$(PYTHON) generate_templates.py

.PHONY: fill-templates
fill-templates: ## [3/5] Fill the frames over the experiment verbs
	@echo "$(BLUE)[3/5] Filling frames ($(VERBS) verbs)$(NC)"
	$(PYTHON) fill_templates.py --limit $(VERBS) --by-frequency

.PHONY: fill-templates-quick
fill-templates-quick: ## [3/5] Fill the frames over a few verbs
	@echo "$(BLUE)[3/5] Filling frames ($(QUICK_VERBS) verbs)$(NC)"
	$(PYTHON) fill_templates.py --limit $(QUICK_VERBS) --by-frequency

.PHONY: 2afc-pairs
2afc-pairs: ## [4/5] Score the sentences and build the minimal pairs
	@echo "$(BLUE)[4/5] 2AFC pairs$(NC)"
	@echo "$(YELLOW)Scoring loads a 12B model and is slow on first run; scores are cached$(NC)"
	$(PYTHON) create_2afc_pairs.py

.PHONY: lists
lists: ## [5/5] Partition the pairs into experiment lists
	@echo "$(BLUE)[5/5] Experiment lists$(NC)"
	$(PYTHON) generate_lists.py

# ============================================================================
# Deployment
# ============================================================================

.PHONY: jspsych-build
jspsych-build: ## Compile the jsPsych assets the generator needs
	@echo "$(BLUE)Building jsPsych assets$(NC)"
	cd $(JSPSYCH_DIR) && npm install --silent && npm run build && npm run build:gallery

.PHONY: deployment
deployment: jspsych-build ## Build every configured list and package for JATOS
	@echo "$(BLUE)Deployment (all configured lists)$(NC)"
	$(PYTHON) generate_deployment.py

.PHONY: deployment-quick
deployment-quick: jspsych-build ## Build 2 experiments for local testing
	@echo "$(BLUE)Deployment (2 lists)$(NC)"
	$(PYTHON) generate_deployment.py --n-lists 2
	@echo "  Open deployment/local/list_01/index.html"

.PHONY: deployment-local
deployment-local: jspsych-build ## Build the local version only, skipping JATOS packaging
	@echo "$(BLUE)Deployment (local only)$(NC)"
	$(PYTHON) generate_deployment.py --no-jatos

# ============================================================================
# Quality
# ============================================================================

.PHONY: lint
lint: ## Lint the pipeline scripts
	@echo "$(BLUE)Linting$(NC)"
	$(PYTHON) -m ruff check \
		build_frequencies.py \
		generate_lexicons.py \
		generate_templates.py \
		fill_templates.py \
		create_2afc_pairs.py \
		generate_lists.py \
		generate_deployment.py \
		utils/

.PHONY: check
check: lint ## Run all quality checks

# ============================================================================
# Inspection
# ============================================================================

.PHONY: show-stats
show-stats: ## Show counts for the generated files
	@echo "$(BLUE)Generated data$(NC)"
	@for f in $(LEXICONS_DIR)/*.jsonl $(TEMPLATES_DIR)/*.jsonl \
	          $(ITEMS_DIR)/*.jsonl $(LISTS_DIR)/*.jsonl; do \
		if [ -f "$$f" ]; then \
			printf "  %-34s %8s lines  %6s\n" "$$f" \
				"$$(wc -l < $$f | tr -d ' ')" \
				"$$(ls -lh $$f | awk '{print $$5}')"; \
		fi \
	done
	@if [ -d deployment/jatos ]; then \
		printf "  %-34s %8s archives\n" "deployment/jatos" \
			"$$(ls deployment/jatos/*.jzip 2>/dev/null | wc -l | tr -d ' ')"; \
	fi

.PHONY: show-config
show-config: ## Show the key configuration values
	@$(PYTHON) -c "import yaml; c = yaml.safe_load(open('config.yaml')); \
		m = [x for x in c['items']['models'] if x.get('use_for_scoring')][0]; \
		print('language:      ', c['project']['language_code']); \
		print('scoring model: ', m['name'], '(' + m['type'] + ', ' + m['device'] + ')'); \
		print('frames:        ', 5); \
		print('lists:         ', c['lists']['n_lists'], 'x', c['lists']['items_per_list']); \
		print('deploying:     ', c['deployment']['n_lists_to_deploy'], 'lists')"

# ============================================================================
# Cleaning
# ============================================================================

.PHONY: clean
clean: ## Remove generated items, lists and deployment
	@rm -f $(ITEMS_DIR)/*.jsonl $(LISTS_DIR)/*.jsonl
	@rm -rf deployment
	@echo "$(GREEN)Cleaned items, lists and deployment$(NC)"

.PHONY: clean-cache
clean-cache: ## Discard the cached scores, forcing a full rescore next run
	@rm -rf $(CACHE_DIR)
	@echo "$(RED)Cached scores discarded; the next run rescores every sentence$(NC)"

.PHONY: clean-all
clean-all: clean clean-cache ## Remove every generated file AND the cached scores
	@rm -f $(LEXICONS_DIR)/*.jsonl $(TEMPLATES_DIR)/*.jsonl
	@echo "$(RED)All generated data removed (resources/ kept)$(NC)"

.DEFAULT_GOAL := help
