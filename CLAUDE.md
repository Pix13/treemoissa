# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**treemoissa** is a Python-based car trackdays and shows triage tool. This tool runs on WSL2 with the Cuda toolkit installed, and is running on a GeForce 4060ti 8Gb
The aim of this project is to create a console tool that will take as input a flat directory of hundreds of pictures, and out put a directory tree with all brands, models, and colors of cars discovered.
If a picture have multiple cars on it, the picture must be duplicated in each brand/model/color subdirectory.
Claude will identify the best AI models for this, and the tool will download them automatically and set itself up on the computer.

## Architecture

- **LLM mode (default):** lightweight install (`pip install treemoissa`), uses a local llama.cpp server with Qwen3.5-9B for car identification. Supports multiple servers via `--llm-host ip:port,ip:port`.
- **ML mode (optional):** requires `pip install treemoissa[ml]` for PyTorch + YOLO + ViT pipeline, activated with `--model`.
- `runserver` command auto-downloads llama.cpp and the GGUF model.

## Notes

- The `.gitignore` is configured for Python projects and includes patterns for common tools: pytest, mypy, ruff, venv, uv, poetry, pdm, pixi, and marimo.
- the tool must be able to run over a NFS subdirectory so no hardlinks for duplicated photos.

## Conventions
- Python 3.10+, PEP8 strict
- Commits au format Conventional Commits:x

