#!/bin/bash
set -e
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --no-install-package flash-attn
uv sync