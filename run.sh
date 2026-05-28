#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
    set -a; source .env; set +a
fi

if [ -d "venv/bin" ]; then
    source venv/bin/activate
fi

python main.py "$@"
