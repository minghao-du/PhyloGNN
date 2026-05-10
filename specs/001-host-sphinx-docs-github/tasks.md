---
description: "Task list for GitHub Pages documentation deployment"
---

# Tasks: Host Sphinx Docs on GitHub Pages

**Input**: Design documents from `/specs/001-host-sphinx-docs-github/`
**Prerequisites**: plan.md (required), spec.md (required for user stories), research.md, contracts/

**Tests**: Validation for CI workflows is done via manual verification or PR checks.
Tests may be omitted only for documentation-only or pure maintenance work, and the omission must be justified in the plan. (Justified: CI/CD configuration change).

**Organization**: Tasks are grouped by user story to enable independent implementation and testing of each story.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: Can run in parallel (different files, no dependencies)
- **[Story]**: Which user story this task belongs to (e.g., US1, US2, US3)
- Include exact file paths in descriptions

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: Project initialization and basic structure

- [X] T001 Create `.github/workflows/` directory structure for GitHub Actions

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: Core infrastructure that MUST be complete before ANY user story can be implemented

- [X] T002 Add `sphinx.ext.githubpages` to `extensions` list in `docs/source/conf.py` to automatically generate the `.nojekyll` file

## Phase 3: User Story 1 & 2 - Automatic Deployment & Public Access (Priority: P1) 🎯 MVP

**Goal**: Automatically deploy documentation to GitHub pages on push to main branch so it is publicly accessible.

**Independent Test**: Verify that pushing a change to `main` triggers the GitHub Action, which successfully builds and deploys to GitHub Pages without errors, and the docs are accessible.

### Tests for User Story 1 & 2 ⚠️

> **NOTE**: Automated testing for GitHub Actions is limited. Validation relies on PR checks and integration tests in the GitHub environment.

- [X] T003 [US1] Document steps in README.md to test build locally with `make html SPHINXOPTS="-W"` in `docs/`

### Implementation for User Story 1 & 2

- [X] T004 [US1] Create `.github/workflows/docs-deploy.yml` with `push` trigger on `main` branch, and configure `concurrency` group to cancel in-progress builds
- [X] T005 [US1] Add job in `.github/workflows/docs-deploy.yml` to checkout code, setup Python 3.12, and run `pip install .[docs,all]`
- [X] T006 [US1] Add step in `.github/workflows/docs-deploy.yml` to build Sphinx docs using `make html` with `-W` flag (warnings as errors), specifying `working-directory: docs`
- [X] T007 [US1] Add step in `.github/workflows/docs-deploy.yml` to run link validation for repository-owned documentation links, specifying `working-directory: docs`; if using `make linkcheck`, configure external-link ignores in `docs/source/conf.py` so SC-003 is not gated on third-party network availability
- [X] T008 [US1] Add step in `.github/workflows/docs-deploy.yml` to upload `docs/_build/html` using `actions/upload-pages-artifact` only after the Sphinx build and link validation steps succeed
- [X] T009 [US2] Add deployment job in `.github/workflows/docs-deploy.yml` using `actions/deploy-pages`, configure permissions (pages: write, id-token: write), and make deployment depend on successful artifact upload via `needs` and success-only job conditions
- [X] T010 [US1] Document in `README.md` that failed docs builds notify maintainers through the failed GitHub Actions workflow run/status check, and verify that failed build or link validation jobs stop before artifact upload/deployment so the previous GitHub Pages version remains online

**Checkpoint**: At this point, the main deployment workflow should be fully functional when merged to main.

## Phase 4: User Story 3 - Manual Deployment Trigger (Priority: P2)

**Goal**: Allow manual triggering of the documentation build and deploy process.

**Independent Test**: Manually run the deployment trigger from the GitHub Actions tab.

### Implementation for User Story 3

- [X] T011 [US3] Add `workflow_dispatch` to the `on` triggers in `.github/workflows/docs-deploy.yml`

## Phase 5: PR Validation (FR-007)

**Goal**: Trigger a documentation build (without deployment) on pull requests targeting the main branch to validate the build process.

**Independent Test**: Open a PR targeting `main` and verify the `docs-pr.yml` action runs and succeeds (or fails on Sphinx warnings).

### Implementation for PR Validation

- [X] T012 [P] Create `.github/workflows/docs-pr.yml` with `pull_request` trigger for `main` branch, and configure `concurrency` group to cancel in-progress builds
- [X] T013 Configure `.github/workflows/docs-pr.yml` build job (setup Python 3.12, `pip install .[docs,all]`, `make html SPHINXOPTS="-W"` and repository-owned link validation with `working-directory: docs`) without any deployment steps

## Phase 6: Polish & Cross-Cutting Concerns

**Purpose**: Improvements that affect multiple user stories

- [X] T014 Document in `README.md` how to validate that GitHub Pages is enabled in the repository settings and uses GitHub Actions as the source
- [X] T015 Verify that the entire GitHub Actions build and deployment pipeline completes in under 5 minutes (SC-002); if it exceeds the target, record the measured duration in `README.md` and either optimize assets/dependency caching or document a follow-up remediation
- [X] T016 Compare the deployed GitHub Pages homepage, navigation, and static styling against the local `docs/_build/html` build and record the visual parity check in `README.md` (SC-004)

## Dependencies & Execution Order

### Phase Dependencies
- **Setup (Phase 1)**: No dependencies - can start immediately
- **Foundational (Phase 2)**: Depends on Setup completion - BLOCKS all user stories
- **User Stories (Phase 3+)**: All depend on Foundational phase completion
- **PR Validation (Phase 5)**: Can be done in parallel with Phase 3

### User Story Dependencies
- **User Story 1 & 2 (P1)**: Can start after Foundational (Phase 2)
- **User Story 3 (P2)**: Depends on User Story 1 & 2 being implemented (modifies the same file)

## Implementation Strategy

### MVP First (User Story 1 & 2 Only)
1. Complete Phase 1: Setup
2. Complete Phase 2: Foundational
3. Complete Phase 3: User Story 1 & 2
4. Complete Phase 5: PR Validation
5. **STOP and VALIDATE**: Test PR build and main deployment.

### Incremental Delivery
1. Add `workflow_dispatch` (User Story 3) for manual triggers.
