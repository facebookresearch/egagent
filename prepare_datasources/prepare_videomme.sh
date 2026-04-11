# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Creative Commons Attribution-NonCommercial 4.0 International (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://creativecommons.org/licenses/by-nc/4.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env bash
# Unzip all Video-MME archives in parallel using native unzip.
# Uses VIDEO_MME_ROOT from paths.py (repo root). Override by passing a directory as first argument.

set -e

start=$(date +%s)

EGAGENT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
if [[ -n "$1" ]]; then
  ROOT="$1"
else
  ROOT="$(cd "$EGAGENT_ROOT" && python -c "from paths import VIDEO_MME_ROOT; print(VIDEO_MME_ROOT)")"
fi
cd "$ROOT"

# Run one unzip per .zip file, up to 32 in parallel (tune as needed).
printf 'Unzipping all .zip files in %s with max 32 parallel jobs\n' "$ROOT"
find . -maxdepth 1 -name '*.zip' -print0 | xargs -0 -P 32 -I {} unzip -o -q {}

end=$(date +%s)
echo "Unzipping done. Time elapsed: $((end - start)) seconds"

# Delete the zip files now that they've been extracted.
find . -maxdepth 1 -name '*.zip' -print0 | xargs -0 rm -f
