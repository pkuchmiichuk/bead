# ============================================================================
# Argument Structure Active Learning Pipeline - Makefile
# ============================================================================
#
# This Makefile provides convenient targets for running all components of the
# argument structure active learning pipeline.
#
# Quick Start:
#   make help           - Show available targets
#   make all            - Run full pipeline (generate data + run pipeline)
#   make test           - Run all tests
#   make clean          - Clean generated files
#
# ============================================================================

# Python interpreter (use uv run with project root for dependency management)
PROJECT_ROOT := $(shell cd ../../.. && pwd)
PYTHON := uv run --project $(PROJECT_ROOT) python
BEAD := uv run --project $(PROJECT_ROOT) bead

# Directories
ITEMS_DIR := items
TEMPLATES_DIR := templates
LEXICONS_DIR := lexicons
RESULTS_DIR := results
CACHE_DIR := .cache

# Default limits for testing
DEFAULT_CROSS_PRODUCT_LIMIT := 100
DEFAULT_2AFC_LIMIT := 50
DEFAULT_INITIAL_SIZE := 50
DEFAULT_UNLABELED_SIZE := 100

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m # No Color

# ============================================================================
# Main Targets
# ============================================================================

.PHONY: help
help: ## Show this help message
	@echo "$(BLUE)============================================================================$(NC)"
	@echo "$(BLUE)Argument Structure Active Learning Pipeline - Available Targets$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@echo ""
	@echo "$(GREEN)Main Targets:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "(all|test|clean|help|pipeline|data)"
	@echo ""
	@echo "$(GREEN)Data Generation:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "(lexicons|templates|fill-templates|cross-product|2afc|lists|deployment)"
	@echo ""
	@echo "$(GREEN)Pipeline Execution:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "(run|dry-run)"
	@echo ""
	@echo "$(GREEN)Simulation:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "simulate"
	@echo ""
	@echo "$(GREEN)Testing & Quality:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "(test|lint|check)"
	@echo ""
	@echo "$(GREEN)Cleaning:$(NC)"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(YELLOW)%-25s$(NC) %s\n", $$1, $$2}' | \
		grep -E "clean"
	@echo ""
	@echo "$(BLUE)Examples:$(NC)"
	@echo "  make all                    # Run complete pipeline with test data"
	@echo "  make data                   # Generate all data files"
	@echo "  make pipeline-dry-run       # Test pipeline without training"
	@echo "  make test                   # Run all tests"
	@echo ""

.PHONY: all
all: data pipeline-dry-run ## Run complete pipeline with test data
	@echo "$(GREEN)✓ Complete pipeline executed successfully$(NC)"

.PHONY: data
data: validate-protocol lexicons templates fill-templates cross-product 2afc-pairs ## Generate all data files
	@echo "$(GREEN)✓ All data files generated$(NC)"

.PHONY: validate-protocol
validate-protocol: ## Validate the protocol section of config.yaml builds cleanly
	@echo "$(BLUE)Validating protocol declaration...$(NC)"
	@$(PYTHON) -c "from protocol import build_protocol; \
		p = build_protocol('config.yaml'); \
		assert p.families, 'No families declared in protocol'; \
		anchor = p.families[0].anchor; \
		print('  protocol:', p.name); \
		print('  family:', anchor.name); \
		print('  prompt:', anchor.canonical_prompt); \
		print('  scale_type:', anchor.response_space.scale_type)"
	@echo "$(GREEN)✓ Protocol validated$(NC)"

# ============================================================================
# Data Generation Targets
# ============================================================================

.PHONY: lexicons
lexicons: ## Generate lexicons from VerbNet and bleached resources
	@echo "$(BLUE)[1/7] Generating lexicons...$(NC)"
	$(PYTHON) generate_lexicons.py
	@echo "$(GREEN)✓ Lexicons generated$(NC)"

.PHONY: verbnet-templates
verbnet-templates: ## Generate verb-specific VerbNet templates
	@echo "$(BLUE)[2/7] Generating verb-specific templates...$(NC)"
	$(PYTHON) generate_templates.py -y
	@echo "$(GREEN)✓ Verb-specific templates generated$(NC)"

.PHONY: templates
templates: verbnet-templates ## Extract generic frame templates from verb-specific templates
	@echo "$(BLUE)[3/7] Extracting generic templates...$(NC)"
	$(PYTHON) extract_generic_templates.py -y
	@echo "$(GREEN)✓ Generic templates extracted$(NC)"

.PHONY: fill-templates
fill-templates: ## Fill templates using MLM strategy
	@echo "$(BLUE)[4/7] Filling templates with MLM strategy...$(NC)"
	$(PYTHON) fill_templates.py --dry-run -y
	@echo "$(GREEN)✓ Templates filled$(NC)"

.PHONY: fill-templates-full
fill-templates-full: ## Fill all templates (no limit)
	@echo "$(BLUE)[4/7] Filling all templates (this may take a while)...$(NC)"
	$(PYTHON) fill_templates.py -y
	@echo "$(GREEN)✓ All templates filled$(NC)"

.PHONY: cross-product
cross-product: ## Generate cross-product items (verb × template)
	@echo "$(BLUE)[5/7] Generating cross-product items (limit=$(DEFAULT_CROSS_PRODUCT_LIMIT))...$(NC)"
	$(PYTHON) generate_cross_product.py --limit $(DEFAULT_CROSS_PRODUCT_LIMIT) -y
	@echo "$(GREEN)✓ Cross-product items generated$(NC)"

.PHONY: cross-product-full
cross-product-full: ## Generate full cross-product (all 74,880 combinations)
	@echo "$(BLUE)[5/7] Generating FULL cross-product items (this may take a while)...$(NC)"
	$(PYTHON) generate_cross_product.py -y
	@echo "$(GREEN)✓ Full cross-product items generated$(NC)"

.PHONY: 2afc-pairs
2afc-pairs: ## Generate 2AFC pairs from cross-product items
	@echo "$(BLUE)[6/7] Generating 2AFC pairs (limit=$(DEFAULT_2AFC_LIMIT))...$(NC)"
	$(PYTHON) create_2afc_pairs.py --limit $(DEFAULT_2AFC_LIMIT) -y
	@echo "$(GREEN)✓ 2AFC pairs generated$(NC)"

.PHONY: 2afc-pairs-full
2afc-pairs-full: ## Generate 2AFC pairs from all cross-product items
	@echo "$(BLUE)[6/7] Generating 2AFC pairs (no limit)...$(NC)"
	$(PYTHON) create_2afc_pairs.py -y
	@echo "$(GREEN)✓ All 2AFC pairs generated$(NC)"

.PHONY: lists
lists: ## Partition 2AFC pairs into experiment lists
	@echo "$(BLUE)[7/7] Generating experiment lists...$(NC)"
	$(PYTHON) generate_lists.py
	@echo "$(GREEN)✓ Experiment lists generated$(NC)"

# ============================================================================
# Acceptability Model Targets (MegaAcceptability -> 2AFC classifier)
# ============================================================================

.PHONY: acceptability-data
acceptability-data: ## Map MegaAcceptability Likert ratings to 2AFC training pairs
	@echo "$(BLUE)Preparing MegaAcceptability 2AFC pairs...$(NC)"
	$(PYTHON) prepare_megaacceptability.py -y
	@echo "$(GREEN)✓ Acceptability training pairs prepared$(NC)"

.PHONY: acceptability-train
acceptability-train: ## Train the acceptability model (needs torch; slow on CPU)
	@echo "$(BLUE)Training acceptability model...$(NC)"
	$(PYTHON) train_acceptability_model.py -y
	@echo "$(GREEN)✓ Acceptability model trained$(NC)"

.PHONY: acceptability
acceptability: acceptability-data acceptability-train ## Prepare data and train the acceptability model
	@echo "$(GREEN)✓ Acceptability model ready for stratification$(NC)"

# ============================================================================
# Layers / lairs Interop Targets
# ============================================================================

.PHONY: layers-encode
layers-encode: ## Encode the 2AFC item collection as a layers fragment
	@echo "$(BLUE)Encoding 2AFC items as a layers fragment...$(NC)"
	$(BEAD) layers encode items/2afc_pairs.jsonl --out items/2afc_pairs.layers.json
	@echo "$(GREEN)✓ Layers fragment written$(NC)"

.PHONY: layers-materialize
layers-materialize: ## Materialize the 2AFC items as an Arrow/Parquet corpus
	@echo "$(BLUE)Materializing 2AFC items as a layers corpus...$(NC)"
	$(BEAD) layers materialize items/2afc_pairs.jsonl --out items/2afc_corpus
	@echo "$(GREEN)✓ Layers corpus materialized$(NC)"

.PHONY: layers
layers: layers-encode layers-materialize ## Produce all layers artifacts for the 2AFC items
	@echo "$(GREEN)✓ Layers artifacts produced$(NC)"

.PHONY: deployment
deployment: ## Generate jsPsych/JATOS deployment files
	@echo "$(BLUE)Generating jsPsych/JATOS deployment...$(NC)"
	$(PYTHON) generate_deployment.py --n-lists 2
	@echo "$(GREEN)✓ Deployment files generated$(NC)"

.PHONY: deployment-full
deployment-full: ## Generate deployment for all lists
	@echo "$(BLUE)Generating full deployment (20 lists)...$(NC)"
	$(PYTHON) generate_deployment.py --n-lists 20
	@echo "$(GREEN)✓ Full deployment generated$(NC)"

# ============================================================================
# Pipeline Execution Targets
# ============================================================================

.PHONY: pipeline-dry-run
pipeline-dry-run: ## Run pipeline in dry-run mode (test configuration)
	@echo "$(BLUE)Running pipeline in dry-run mode...$(NC)"
	$(PYTHON) run_pipeline.py \
		--dry-run \
		--initial-size $(DEFAULT_INITIAL_SIZE) \
		--unlabeled-size $(DEFAULT_UNLABELED_SIZE)
	@echo "$(GREEN)✓ Dry run completed$(NC)"

.PHONY: pipeline
pipeline: ## Run full active learning pipeline
	@echo "$(BLUE)Running active learning pipeline...$(NC)"
	@echo "$(YELLOW)⚠ Warning: This will run the full training loop$(NC)"
	$(PYTHON) run_pipeline.py \
		--initial-size $(DEFAULT_INITIAL_SIZE) \
		--unlabeled-size $(DEFAULT_UNLABELED_SIZE)
	@echo "$(GREEN)✓ Pipeline completed$(NC)"

.PHONY: pipeline-full
pipeline-full: ## Run pipeline with production settings
	@echo "$(BLUE)Running pipeline with production settings...$(NC)"
	@echo "$(YELLOW)⚠ Warning: This will use large datasets and may take a long time$(NC)"
	$(PYTHON) run_pipeline.py \
		--initial-size 1000 \
		--unlabeled-size 5000
	@echo "$(GREEN)✓ Pipeline completed$(NC)"

# ============================================================================
# Testing & Quality Assurance Targets
# ============================================================================

.PHONY: test
test: test-unit test-lint ## Run all tests (unit + linting)
	@echo "$(GREEN)✓ All tests passed$(NC)"

.PHONY: test-unit
test-unit: ## Run unit tests for evaluation module
	@echo "$(BLUE)Running unit tests...$(NC)"
	cd ../../.. && $(PYTHON) -m pytest tests/evaluation/ -v
	@echo "$(GREEN)✓ Unit tests passed$(NC)"

.PHONY: test-lint
test-lint: ## Run linting checks on pipeline scripts
	@echo "$(BLUE)Running linting checks...$(NC)"
	$(PYTHON) -m ruff check \
		generate_lexicons.py \
		generate_templates.py \
		extract_generic_templates.py \
		fill_templates.py \
		generate_cross_product.py \
		create_2afc_pairs.py \
		generate_lists.py \
		generate_deployment.py \
		run_pipeline.py \
		simulate_pipeline.py
	@echo "$(GREEN)✓ Linting checks passed$(NC)"

.PHONY: test-types
test-types: ## Run type checking with pyright
	@echo "$(BLUE)Running type checks...$(NC)"
	cd ../../.. && $(PYTHON) -m pyright \
		gallery/eng/argument_structure/create_2afc_pairs.py \
		gallery/eng/argument_structure/run_pipeline.py
	@echo "$(GREEN)✓ Type checks passed$(NC)"

.PHONY: check
check: test-lint test-types ## Run static analysis (lint + types)
	@echo "$(GREEN)✓ Static analysis passed$(NC)"

# ============================================================================
# Data Inspection Targets
# ============================================================================

.PHONY: show-stats
show-stats: ## Show statistics about generated data
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
		done \
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
	@echo "$(GREEN)Cross-Product Items:$(NC)"
	@if [ -f "$(ITEMS_DIR)/cross_product_items.jsonl" ]; then \
		count=$$(wc -l < "$(ITEMS_DIR)/cross_product_items.jsonl"); \
		size=$$(ls -lh "$(ITEMS_DIR)/cross_product_items.jsonl" | awk '{print $$5}'); \
		echo "  cross_product_items.jsonl: $$count items ($$size)"; \
	else \
		echo "  $(RED)No cross-product items found. Run 'make cross-product' first.$(NC)"; \
	fi
	@echo ""
	@echo "$(GREEN)2AFC Pairs:$(NC)"
	@if [ -f "$(ITEMS_DIR)/2afc_pairs.jsonl" ]; then \
		count=$$(wc -l < "$(ITEMS_DIR)/2afc_pairs.jsonl"); \
		size=$$(ls -lh "$(ITEMS_DIR)/2afc_pairs.jsonl" | awk '{print $$5}'); \
		echo "  2afc_pairs.jsonl: $$count pairs ($$size)"; \
	else \
		echo "  $(RED)No 2AFC pairs found. Run 'make 2afc-pairs' first.$(NC)"; \
	fi
	@echo ""

.PHONY: show-config
show-config: ## Show pipeline configuration
	@echo "$(BLUE)============================================================================$(NC)"
	@echo "$(BLUE)Pipeline Configuration$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@echo ""
	@$(PYTHON) -c "import yaml; config = yaml.safe_load(open('config.yaml')); \
		print('Project:', config['project']['name']); \
		print('Language:', config['project']['language_code']); \
		print(''); \
		print('Active Learning:'); \
		print('  Strategy:', config['active_learning']['strategy']); \
		print('  Method:', config['active_learning']['method']); \
		print('  Max iterations:', config['active_learning']['max_iterations']); \
		print('  Budget per iteration:', config['active_learning']['budget_per_iteration']); \
		print(''); \
		print('Convergence:'); \
		print('  Metric:', config['training']['convergence']['metric']); \
		print('  Threshold:', config['training']['convergence']['threshold']); \
		print('  Min iterations:', config['training']['convergence']['min_iterations']); \
		print(''); \
		print('Data:'); \
		print('  Lists:', config['lists']['n_lists']); \
		print('  Items per list:', config['lists']['items_per_list']); \
		print('  Quantile bins:', config['lists']['quantile_bins'])"
	@echo ""

# ============================================================================
# Cleaning Targets
# ============================================================================

.PHONY: clean
clean: clean-items clean-cache ## Clean generated items and cache
	@echo "$(GREEN)✓ Cleaned generated files$(NC)"

.PHONY: clean-items
clean-items: ## Clean generated items (cross-product and 2AFC pairs)
	@echo "$(BLUE)Cleaning generated items...$(NC)"
	@rm -f $(ITEMS_DIR)/cross_product_items.jsonl
	@rm -f $(ITEMS_DIR)/2afc_pairs.jsonl
	@echo "$(GREEN)✓ Items cleaned$(NC)"

.PHONY: clean-cache
clean-cache: ## Clean model cache
	@echo "$(BLUE)Cleaning model cache...$(NC)"
	@rm -rf $(CACHE_DIR)
	@echo "$(GREEN)✓ Cache cleaned$(NC)"

.PHONY: clean-results
clean-results: ## Clean pipeline results
	@echo "$(BLUE)Cleaning results...$(NC)"
	@rm -rf $(RESULTS_DIR)
	@echo "$(GREEN)✓ Results cleaned$(NC)"

.PHONY: clean-simulation
clean-simulation: ## Clean simulation outputs
	@echo "$(BLUE)Cleaning simulation outputs...$(NC)"
	@rm -rf simulation_output/
	@echo "$(GREEN)✓ Simulation outputs cleaned$(NC)"

.PHONY: clean-all
clean-all: clean-items clean-cache clean-results clean-lexicons clean-templates ## Clean everything (including lexicons and templates)
	@echo "$(RED)⚠ Warning: All generated data has been removed$(NC)"
	@echo "$(GREEN)✓ Complete clean finished$(NC)"

.PHONY: clean-lexicons
clean-lexicons: ## Clean generated lexicons
	@echo "$(BLUE)Cleaning lexicons...$(NC)"
	@rm -f $(LEXICONS_DIR)/*.jsonl
	@echo "$(GREEN)✓ Lexicons cleaned$(NC)"

.PHONY: clean-templates
clean-templates: ## Clean extracted templates
	@echo "$(BLUE)Cleaning templates...$(NC)"
	@rm -f $(TEMPLATES_DIR)/verbnet_frames.jsonl
	@rm -f $(TEMPLATES_DIR)/generic_frames.jsonl
	@echo "$(GREEN)✓ Templates cleaned$(NC)"

.PHONY: clean-filled-templates
clean-filled-templates: ## Clean filled templates
	@echo "$(BLUE)Cleaning filled templates...$(NC)"
	@rm -f filled_templates/*.jsonl
	@echo "$(GREEN)✓ Filled templates cleaned$(NC)"

.PHONY: clean-lists
clean-lists: ## Clean experiment lists
	@echo "$(BLUE)Cleaning experiment lists...$(NC)"
	@rm -f lists/*.jsonl
	@echo "$(GREEN)✓ Experiment lists cleaned$(NC)"

.PHONY: clean-deployment
clean-deployment: ## Clean deployment files
	@echo "$(BLUE)Cleaning deployment files...$(NC)"
	@rm -rf deployment/jatos deployment/local
	@echo "$(GREEN)✓ Deployment files cleaned$(NC)"

# ============================================================================
# Development Targets
# ============================================================================

.PHONY: dev-setup
dev-setup: ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	@echo "Checking Python version..."
	@$(PYTHON) --version
	@echo "Installing dependencies..."
	@cd ../../.. && $(PYTHON) -m pip install -e ".[dev,api,training]"
	@echo "$(GREEN)✓ Development environment ready$(NC)"

.PHONY: dev-test-small
dev-test-small: ## Quick development test (small data)
	@echo "$(BLUE)Running quick development test...$(NC)"
	@$(MAKE) clean-items
	@$(PYTHON) generate_cross_product.py --limit 20
	@$(PYTHON) create_2afc_pairs.py --limit 20
	@$(PYTHON) run_pipeline.py --dry-run --initial-size 10 --unlabeled-size 5
	@echo "$(GREEN)✓ Development test passed$(NC)"

.PHONY: dev-test-medium
dev-test-medium: ## Medium development test
	@echo "$(BLUE)Running medium development test...$(NC)"
	@$(MAKE) clean-items
	@$(PYTHON) generate_cross_product.py --limit 200
	@$(PYTHON) create_2afc_pairs.py --limit 100
	@$(PYTHON) run_pipeline.py --dry-run --initial-size 50 --unlabeled-size 50
	@echo "$(GREEN)✓ Medium development test passed$(NC)"

# ============================================================================
# Production Targets
# ============================================================================

.PHONY: prod-data
prod-data: clean-items lexicons templates cross-product-full 2afc-pairs-full ## Generate production data (full cross-product)
	@echo "$(GREEN)✓ Production data generated$(NC)"
	@$(MAKE) show-stats

.PHONY: prod-pipeline
prod-pipeline: ## Run production pipeline
	@echo "$(RED)⚠ Warning: This will run the full production pipeline$(NC)"
	@echo "$(YELLOW)This may take several hours and requires human data collection.$(NC)"
	@read -p "Continue? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		$(PYTHON) run_pipeline.py --config config.yaml; \
	else \
		echo "$(YELLOW)Production pipeline cancelled$(NC)"; \
	fi

# ============================================================================
# Documentation Targets
# ============================================================================

.PHONY: docs
docs: ## Generate documentation
	@echo "$(BLUE)Documentation:$(NC)"
	@echo "  - README.md: Main documentation"
	@echo "  - PROGRESS.md: Development progress log"
	@echo "  - config.yaml: Pipeline configuration"
	@echo "  - Makefile: This file (run 'make help')"
	@echo ""
	@echo "To view configuration: make show-config"
	@echo "To view data stats: make show-stats"

# ============================================================================
# Simulation Targets
# ============================================================================

.PHONY: simulate simulate-quick simulate-medium simulate-full simulate-noisy simulate-clean
.PHONY: show-simulation-results analyze-convergence clean-simulation

simulate: simulate-quick ## Run simulation (alias for simulate-quick)

simulate-quick: ## Quick simulation (50 items, 3 iterations, ~2 min)
	@echo "$(BLUE)Running quick simulation...$(NC)"
	$(PYTHON) simulate_pipeline.py \
		--initial-size 30 \
		--budget 10 \
		--max-iterations 3 \
		--temperature 1.0 \
		--seed 42 \
		--max-items 50 \
		--output-dir simulation_output/quick
	@echo "$(GREEN)✓ Quick simulation completed$(NC)"

simulate-medium: ## Medium simulation (200 items, 10 iterations, ~10 min)
	@echo "$(BLUE)Running medium simulation...$(NC)"
	$(PYTHON) simulate_pipeline.py \
		--initial-size 50 \
		--budget 20 \
		--max-iterations 10 \
		--temperature 1.0 \
		--seed 42 \
		--max-items 200 \
		--output-dir simulation_output/medium
	@echo "$(GREEN)✓ Medium simulation completed$(NC)"

simulate-full: ## Full simulation (1000 items, 20 iterations, ~30 min)
	@echo "$(BLUE)Running full simulation...$(NC)"
	@echo "$(YELLOW)⚠ This may take 10-30 minutes$(NC)"
	$(PYTHON) simulate_pipeline.py \
		--initial-size 100 \
		--budget 50 \
		--max-iterations 20 \
		--temperature 1.0 \
		--seed 42 \
		--max-items 1000 \
		--output-dir simulation_output/full
	@echo "$(GREEN)✓ Full simulation completed$(NC)"

simulate-noisy: ## Simulation with high temperature (noisy judgments)
	@echo "$(BLUE)Running noisy simulation (temperature=2.0)...$(NC)"
	$(PYTHON) simulate_pipeline.py \
		--initial-size 50 \
		--budget 20 \
		--max-iterations 15 \
		--temperature 2.0 \
		--seed 42 \
		--max-items 200 \
		--output-dir simulation_output/noisy
	@echo "$(GREEN)✓ Noisy simulation completed$(NC)"

simulate-clean: ## Simulation with low temperature (clean judgments)
	@echo "$(BLUE)Running clean simulation (temperature=0.5)...$(NC)"
	$(PYTHON) simulate_pipeline.py \
		--initial-size 50 \
		--budget 20 \
		--max-iterations 15 \
		--temperature 0.5 \
		--seed 42 \
		--max-items 200 \
		--output-dir simulation_output/clean
	@echo "$(GREEN)✓ Clean simulation completed$(NC)"

show-simulation-results: ## Show latest simulation results
	@echo "$(BLUE)============================================================================$(NC)"
	@echo "$(BLUE)Latest Simulation Results$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@if [ -f simulation_output/simulation_results.json ]; then \
		$(PYTHON) -m json.tool simulation_output/simulation_results.json; \
	else \
		echo "$(RED)No simulation results found. Run 'make simulate' first.$(NC)"; \
	fi

analyze-convergence: ## Analyze convergence from simulation results
	@echo "$(BLUE)============================================================================$(NC)"
	@echo "$(BLUE)Convergence Analysis$(NC)"
	@echo "$(BLUE)============================================================================$(NC)"
	@if [ -f simulation_output/simulation_results.json ]; then \
		$(PYTHON) -c "import json; \
		data = json.load(open('simulation_output/simulation_results.json')); \
		print('Iterations:', len(data['iterations'])); \
		print('Final accuracy: {:.3f}'.format(data['iterations'][-1]['test_accuracy'])); \
		print('Human agreement: {:.3f}'.format(data['human_agreement'])); \
		print('Gap: {:.3f}'.format(abs(data['iterations'][-1]['test_accuracy'] - data['human_agreement']))); \
		print('Converged:', data['converged'])"; \
	else \
		echo "$(RED)No simulation results found. Run 'make simulate' first.$(NC)"; \
	fi

# ============================================================================
# Utility Targets
# ============================================================================

.PHONY: dirs
dirs: ## Create necessary directories
	@mkdir -p $(ITEMS_DIR)
	@mkdir -p $(TEMPLATES_DIR)
	@mkdir -p $(LEXICONS_DIR)
	@mkdir -p $(RESULTS_DIR)
	@mkdir -p $(CACHE_DIR)
	@echo "$(GREEN)✓ Directories created$(NC)"

# ============================================================================
# Special Targets
# ============================================================================

.DEFAULT_GOAL := help

# Prevent make from deleting intermediate files
.SECONDARY:

# Declare phony targets (targets that don't create files)
.PHONY: all data help clean test
