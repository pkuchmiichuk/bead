# ============================================================================
# Argument Structure Pipeline - Makefile (Korean)
# ============================================================================
#
# Quick Start:
#   make help              - Show available targets
#   make data              - Run full data generation pipeline
#   make show-stats        - Show statistics about generated data
#   make clean             - Clean generated items and cache
#
# ============================================================================

# Python interpreter (use uv run with project root for dependency management)
PROJECT_ROOT := $(shell cd ../../.. && pwd)
PYTHON := uv run --project $(PROJECT_ROOT) python

# Directories
LEXICONS_DIR     := lexicons
TEMPLATES_DIR    := templates
FILLED_DIR       := filled_templates
ITEMS_DIR        := items
LISTS_DIR        := lists
CACHE_DIR        := .cache

# Default limits for quick/test runs
DEFAULT_FILL_LIMIT := 100
DEFAULT_2AFC_LIMIT := 50
DEFAULT_PER_TEMPLATE_LIMIT := 100

# Colors for output
BLUE   := \033[0;34m
GREEN  := \033[0;32m
YELLOW := \033[0;33m
RED    := \033[0;31m
NC     := \033[0m

# ============================================================================
# Main Targets
# ============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)============================================================================$(NC)"
	@echo "$(BLUE)Argument Structure Pipeline (Korean) - Available Targets$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@echo ""
	@echo "$(GREEN)Main Targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "(all|data|help|show-stats|show-config)"
	@echo ""
	@echo "$(GREEN)Data Generation:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "(lexicons|templates|fill|2afc|lists|data-full)" | \
		grep -v -E "(deployment|clean)"
	@echo ""
	@echo "$(GREEN)Deployment:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "deployment" | \
		grep -v "clean"
	@echo ""
	@echo "$(GREEN)Quality:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "(lint|check)"
	@echo ""
	@echo "$(GREEN)Cleaning:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep "clean"
	@echo ""
	@echo "$(BLUE)Examples:$(NC)"
	@echo "  make data                   # Quick test run (limited verbs + pairs)"
	@echo "  make data-full              # Production run (all verbs + all pairs)"
	@echo "  make fill-templates-full    # Fill templates with all verbs (no limit)"
	@echo "  make 2afc-pairs-full        # Generate 2AFC pairs without limit"
	@echo "  make deployment             # Generate deployment for 2 lists (local test)"
	@echo "  make deployment-full        # Generate deployment for all 16 lists"
	@echo "  make show-stats             # Show counts for all generated files"
	@echo ""

.PHONY: all
all: data-full deployment-full ## Run full production pipeline (all verbs, all pairs, all lists)
	@echo "$(GREEN)✓ Full pipeline executed successfully$(NC)"

.PHONY: data
data: lexicons templates fill-templates 2afc-pairs lists ## Generate all data files with limits (quick test run)
	@echo "$(GREEN)✓ All data files generated$(NC)"

.PHONY: data-full
data-full: lexicons templates-full fill-templates-full 2afc-pairs-full lists ## Generate all data files without limits (production run)
	@echo "$(GREEN)✓ Full data pipeline executed$(NC)"

# ============================================================================
# Data Generation Targets
# ============================================================================

.PHONY: lexicons
lexicons: ## Generate lexicons (UniMorph verbs + bleached resources)
	@echo "$(BLUE)[1/5] Generating lexicons...$(NC)"
	$(PYTHON) generate_lexicons.py
	@echo "$(GREEN)✓ Lexicons generated$(NC)"

.PHONY: templates
templates: ## Generate base templates only (intransitive + transitive, 2 frames) — quick test
	@echo "$(BLUE)[2/5] Generating base templates...$(NC)"
	$(PYTHON) generate_templates.py --include
	@echo "$(GREEN)✓ Templates generated$(NC)"

.PHONY: templates-full
templates-full: ## Generate all templates (base + adjuncts + progressive + complement)
	@echo "$(BLUE)[2/5] Generating all templates...$(NC)"
	$(PYTHON) generate_templates.py
	@echo "$(GREEN)✓ Templates generated$(NC)"

.PHONY: fill-templates
fill-templates: ## Fill templates (limit=$(DEFAULT_FILL_LIMIT) verbs, cap=$(DEFAULT_PER_TEMPLATE_LIMIT)/template) — quick test run
	@echo "$(BLUE)[3/5] Filling templates (limit=$(DEFAULT_FILL_LIMIT) verbs, max $(DEFAULT_PER_TEMPLATE_LIMIT)/template)...$(NC)"
	$(PYTHON) fill_templates.py --limit $(DEFAULT_FILL_LIMIT) --max-per-template $(DEFAULT_PER_TEMPLATE_LIMIT)
	@echo "$(GREEN)✓ Templates filled$(NC)"

.PHONY: fill-templates-full
fill-templates-full: ## Fill all 47 templates with all verbs — production run
	@echo "$(BLUE)[3/5] Filling all 47 templates (all verbs)...$(NC)"
	@echo "$(YELLOW)⚠ This step may take a while$(NC)"
	$(PYTHON) fill_templates.py
	@echo "$(GREEN)✓ Templates filled$(NC)"

.PHONY: 2afc-pairs
2afc-pairs: ## Generate 2AFC pairs (limit=$(DEFAULT_2AFC_LIMIT)) — quick test run
	@echo "$(BLUE)[4/5] Generating 2AFC pairs (limit=$(DEFAULT_2AFC_LIMIT))...$(NC)"
	$(PYTHON) create_2afc_pairs.py --limit $(DEFAULT_2AFC_LIMIT)
	@echo "$(GREEN)✓ 2AFC pairs generated$(NC)"

.PHONY: 2afc-pairs-full
2afc-pairs-full: ## Generate 2AFC pairs from all filled templates (no limit)
	@echo "$(BLUE)[4/5] Generating 2AFC pairs (no limit)...$(NC)"
	@echo "$(YELLOW)⚠ This step uses EleutherAI/polyglot-ko-1.3b and may take a while$(NC)"
	$(PYTHON) create_2afc_pairs.py
	@echo "$(GREEN)✓ All 2AFC pairs generated$(NC)"

.PHONY: lists
lists: ## Partition 2AFC pairs into 16 balanced experiment lists
	@echo "$(BLUE)[5/5] Generating experiment lists...$(NC)"
	$(PYTHON) generate_lists.py
	@echo "$(GREEN)✓ Experiment lists generated$(NC)"

.PHONY: deployment
deployment: ## Generate jsPsych deployment for 2 lists (local testing)
	@echo "$(BLUE)Generating deployment (2 lists, local + jatos)...$(NC)"
	$(PYTHON) generate_deployment.py --n-lists 2
	@echo "$(GREEN)✓ Deployment generated$(NC)"
	@echo "  Open: deployment/local/list_01/index.html"

.PHONY: deployment-full
deployment-full: ## Generate deployment for all 16 lists
	@echo "$(BLUE)Generating full deployment (16 lists)...$(NC)"
	$(PYTHON) generate_deployment.py --n-lists 16
	@echo "$(GREEN)✓ Full deployment generated$(NC)"

.PHONY: deployment-local-only
deployment-local-only: ## Generate local-only deployment (no JATOS .jzip export)
	@echo "$(BLUE)Generating local-only deployment...$(NC)"
	$(PYTHON) generate_deployment.py --n-lists 16 --no-jatos
	@echo "$(GREEN)✓ Local deployment generated$(NC)"

# ============================================================================
# Quality Assurance Targets
# ============================================================================

.PHONY: lint
lint: ## Run ruff linting on all pipeline scripts
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(PYTHON) -m ruff check \
		generate_lexicons.py \
		generate_templates.py \
		fill_templates.py \
		create_2afc_pairs.py \
		generate_lists.py \
		generate_deployment.py
	@echo "$(GREEN)✓ Linting checks passed$(NC)"

.PHONY: check
check: lint ## Run static analysis
	@echo "$(GREEN)✓ Static analysis passed$(NC)"

# ============================================================================
# Data Inspection Targets
# ============================================================================

.PHONY: show-stats
show-stats: ## Show counts and sizes for all generated data files
	@echo "$(BLUE)============================================================================$(NC)"
	@echo "$(BLUE)Data Statistics$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@echo ""
	@echo "$(GREEN)Lexicons:$(NC)"
	@if [ -d "$(LEXICONS_DIR)" ]; then \
		for file in $(LEXICONS_DIR)/*.jsonl; do \
			if [ -f "$$file" ]; then \
				count=$$(wc -l < "$$file"); \
				size=$$(ls -lh "$$file" | awk '{print $$5}'); \
				echo "  $$(basename $$file): $$count entries ($$size)"; \
			fi \
		done; \
	else \
		echo "  $(RED)No lexicons found. Run 'make lexicons' first.$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Templates:$(NC)"
	@if [ -f "$(TEMPLATES_DIR)/generic_frames.jsonl" ]; then \
		count=$$(wc -l < "$(TEMPLATES_DIR)/generic_frames.jsonl"); \
		size=$$(ls -lh "$(TEMPLATES_DIR)/generic_frames.jsonl" | awk '{print $$5}'); \
		echo "  generic_frames.jsonl: $$count templates ($$size)"; \
	else \
		echo "  $(RED)No templates found. Run 'make templates' first.$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Filled Templates:$(NC)"
	@if [ -d "$(FILLED_DIR)" ]; then \
		for file in $(FILLED_DIR)/*.jsonl; do \
			if [ -f "$$file" ]; then \
				count=$$(wc -l < "$$file"); \
				size=$$(ls -lh "$$file" | awk '{print $$5}'); \
				echo "  $$(basename $$file): $$count items ($$size)"; \
			fi \
		done; \
	else \
		echo "  $(RED)No filled templates found. Run 'make fill-templates' first.$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Items:$(NC)"
	@if [ -f "$(ITEMS_DIR)/cross_product_items.jsonl" ]; then \
		count=$$(wc -l < "$(ITEMS_DIR)/cross_product_items.jsonl"); \
		size=$$(ls -lh "$(ITEMS_DIR)/cross_product_items.jsonl" | awk '{print $$5}'); \
		echo "  cross_product_items.jsonl: $$count items ($$size)"; \
	else \
		echo "  $(RED)cross_product_items.jsonl not found.$(NC)"; \
	fi
	@if [ -f "$(ITEMS_DIR)/2afc_pairs.jsonl" ]; then \
		count=$$(wc -l < "$(ITEMS_DIR)/2afc_pairs.jsonl"); \
		size=$$(ls -lh "$(ITEMS_DIR)/2afc_pairs.jsonl" | awk '{print $$5}'); \
		echo "  2afc_pairs.jsonl: $$count pairs ($$size)"; \
	else \
		echo "  $(RED)2afc_pairs.jsonl not found. Run 'make 2afc-pairs-full' first.$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Lists:$(NC)"
	@if [ -f "$(LISTS_DIR)/experiment_lists.jsonl" ]; then \
		count=$$(wc -l < "$(LISTS_DIR)/experiment_lists.jsonl"); \
		size=$$(ls -lh "$(LISTS_DIR)/experiment_lists.jsonl" | awk '{print $$5}'); \
		echo "  experiment_lists.jsonl: $$count lists ($$size)"; \
	else \
		echo "  $(RED)No lists found. Run 'make lists' first.$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)Deployment:$(NC)"
	@if [ -d "deployment/local" ]; then \
		n=$$(ls -d deployment/local/list_* 2>/dev/null | wc -l); \
		echo "  local: $$n lists"; \
	else \
		echo "  $(RED)No local deployment found. Run 'make deployment' first.$(NC)"; \
	fi
	@if [ -d "deployment/jatos" ]; then \
		n=$$(ls deployment/jatos/*.jzip 2>/dev/null | wc -l); \
		echo "  jatos: $$n .jzip packages"; \
	else \
		echo "  $(RED)No jatos deployment found.$(NC)"; \
	fi
	@echo ""

.PHONY: show-config
show-config: ## Show key pipeline configuration values
	@echo "$(BLUE)============================================================================$(NC)"
	@echo "$(BLUE)Pipeline Configuration$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@echo ""
	@$(PYTHON) -c "import yaml; c = yaml.safe_load(open('config.yaml')); \
		print('Project:', c['project']['name']); \
		print('Language:', c['project']['language_code']); \
		print(''); \
		print('MLM model:', c['template']['mlm']['model_name']); \
		print('LM model: ', c['items']['models'][0]['name']); \
		print(''); \
		print('Lists:', c['lists']['n_lists']); \
		print('Items per list:', c['lists']['items_per_list']); \
		print('Quantile bins:', c['lists']['quantile_bins']); \
		print(''); \
		print('Active learning strategy:', c['active_learning']['strategy']); \
		print('Budget per iteration:', c['active_learning']['budget_per_iteration']); \
		print('Max iterations:', c['active_learning']['max_iterations']); \
		print('Convergence metric:', c['training']['convergence']['metric']); \
		print('Convergence threshold:', c['training']['convergence']['threshold'])"
	@echo ""

# ============================================================================
# Cleaning Targets
# ============================================================================

.PHONY: clean
clean: clean-items clean-cache ## Clean generated items and model cache
	@echo "$(GREEN)✓ Cleaned$(NC)"

.PHONY: clean-items
clean-items: ## Clean cross-product items and 2AFC pairs
	@echo "$(BLUE)Cleaning generated items...$(NC)"
	@rm -f $(ITEMS_DIR)/cross_product_items.jsonl
	@rm -f $(ITEMS_DIR)/2afc_pairs.jsonl
	@echo "$(GREEN)✓ Items cleaned$(NC)"

.PHONY: clean-cache
clean-cache: ## Clean MLM/LM model cache
	@echo "$(BLUE)Cleaning model cache...$(NC)"
	@rm -rf $(CACHE_DIR)
	@echo "$(GREEN)✓ Cache cleaned$(NC)"

.PHONY: clean-filled-templates
clean-filled-templates: ## Clean filled templates
	@echo "$(BLUE)Cleaning filled templates...$(NC)"
	@rm -f $(FILLED_DIR)/*.jsonl
	@echo "$(GREEN)✓ Filled templates cleaned$(NC)"

.PHONY: clean-templates
clean-templates: ## Clean generated templates
	@echo "$(BLUE)Cleaning templates...$(NC)"
	@rm -f $(TEMPLATES_DIR)/generic_frames.jsonl
	@echo "$(GREEN)✓ Templates cleaned$(NC)"

.PHONY: clean-lexicons
clean-lexicons: ## Clean generated lexicons
	@echo "$(BLUE)Cleaning lexicons...$(NC)"
	@rm -f $(LEXICONS_DIR)/*.jsonl
	@echo "$(GREEN)✓ Lexicons cleaned$(NC)"

.PHONY: clean-lists
clean-lists: ## Clean experiment lists
	@echo "$(BLUE)Cleaning experiment lists...$(NC)"
	@rm -f $(LISTS_DIR)/*.jsonl
	@echo "$(GREEN)✓ Lists cleaned$(NC)"

.PHONY: clean-deployment
clean-deployment: ## Clean deployment files
	@echo "$(BLUE)Cleaning deployment files...$(NC)"
	@rm -rf deployment/local deployment/jatos
	@echo "$(GREEN)✓ Deployment cleaned$(NC)"

.PHONY: clean-all
clean-all: clean-items clean-cache clean-filled-templates clean-templates clean-lexicons clean-lists clean-deployment ## Clean all generated files
	@echo "$(RED)⚠ All generated data removed$(NC)"

# ============================================================================
# Special Targets
# ============================================================================

.DEFAULT_GOAL := help
.SECONDARY:
.PHONY: all data help clean check lint
