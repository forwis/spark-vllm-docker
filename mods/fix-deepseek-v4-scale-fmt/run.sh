#!/bin/bash
set -e

cd /usr/local/lib/python3.12/dist-packages
echo "Applying PR #41791 (Fix DeepSeek V4 scale_fmt default)"
if curl -fsL https://patch-diff.githubusercontent.com/raw/vllm-project/vllm/pull/41791.diff | git apply --exclude="tests/*"; then
  echo "- PR #41791 applied successfully"
else
  echo "- PR #41791 can't be applied, skipping"
fi
