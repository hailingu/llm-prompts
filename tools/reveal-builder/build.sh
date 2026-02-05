#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper to call node build.js with forwarded args
ROOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT_DIR"

node tools/reveal-builder/build.js "$@"
