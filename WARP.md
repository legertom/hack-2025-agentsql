# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project overview

Repository: `hack-2025-agentsql`

Current description from `README.md`:
- Agent that generates SQL to transform raw input to target schemas.

As of 2025-12-10, the repository contains only this README and no implementation code yet.

## Development commands

As of 2025-12-10, there are no language-specific build, lint, or test commands defined in the repo (no `package.json`, `pyproject.toml`, `Makefile`, `Cargo.toml`, or similar configuration files are present).

When such tooling is added, prefer to:
- Read configuration files (for example, `package.json`, `pyproject.toml`, `Makefile`, `Cargo.toml`, or language-specific tool configs) to discover the canonical build/lint/test commands.
- Update this section with concrete commands, including how to run the full test suite and a single test.

Do not invent commands that are not explicitly defined in the repository.

## Code architecture and structure

There is currently no source code in the repository, so there is no concrete architecture to document yet.

Once implementation is added, update this section to briefly describe:
- The main modules or services involved in generating SQL from raw input to target schemas.
- Where schema definitions live and how they are represented.
- How the agent interfaces with the SQL generation logic (e.g., any orchestration, prompt templates, or pipelines).
- Any entrypoints used in local development (CLI commands, HTTP endpoints, notebooks, etc.).

Focus this section on high-level structure that spans multiple files or directories, not exhaustive file listings.
