#!/bin/bash
set -e
git config --global user.name "Kuzey Aldemir"
git config --global user.email "kuzeyaldemir@gmail.com"
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env
uv sync --no-install-package flash-attn
uv sync