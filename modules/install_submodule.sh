#!/bin/bash
set -e

cd "$(dirname "$0")"

REPOS=(
  "https://github.com/facebookresearch/co-tracker.git"
  "https://github.com/aliang8/robot_learning.git|dev"
  "https://github.com/allenai/molmo.git"
  "https://github.com/IDEA-Research/Grounded-SAM-2.git"
)

for ENTRY in "${REPOS[@]}"; do
  IFS='|' read -r REPO BRANCH <<< "$ENTRY"
  NAME=$(basename "$REPO" .git)

  if [ ! -d "$NAME" ]; then
    echo "📥 Cloning $NAME..."
    git clone "$REPO"
  fi

  cd "$NAME"

  if [ -n "$BRANCH" ]; then
    echo "🔀 Switching $NAME to branch $BRANCH..."
    git fetch origin "$BRANCH"
    git checkout "$BRANCH"
  fi

  if [ "$NAME" == "co-tracker" ]; then
    echo "📦 Installing custom dependencies for co-tracker..."
    pip install -e .
    pip install matplotlib flow_vis tqdm tensorboard
    pip install 'imageio[ffmpeg]'
  elif [[ -f "setup.py" || -f "pyproject.toml" ]]; then
    echo "📦 Installing $NAME with pip..."
    pip install -e .[all] || pip install -e .
  else
    echo "⚠️ Skipping pip install for $NAME (no setup.py or pyproject.toml)"
  fi

  cd ..
done

echo "✅ All submodules cloned and installed (with special handling for co-tracker)."