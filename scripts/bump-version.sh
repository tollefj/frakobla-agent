#!/usr/bin/env bash
# Tags HEAD with the next version. `make versioning` bumps minor, `make versioning major` bumps major.
set -euo pipefail

part="${1:-minor}"

if [ -n "$(git status --porcelain)" ]; then
    echo "uncommitted changes, commit them first so the tag points at the code you want to release" >&2
    exit 1
fi

if tagged=$(git describe --tags --exact-match --match 'v[0-9]*' HEAD 2>/dev/null); then
    echo "HEAD is already tagged $tagged, commit something new before bumping" >&2
    exit 1
fi

latest=$(git tag --list 'v[0-9]*' --sort=-v:refname | head -n 1)
latest="${latest:-v0.0.0}"
IFS=. read -r major minor _ <<<"${latest#v}"

case "$part" in
    major) next="v$((major + 1)).0.0" ;;
    minor) next="v${major}.$((minor + 1)).0" ;;
    *) echo "usage: bump-version.sh [minor|major]" >&2; exit 1 ;;
esac

git tag -a "$next" -m "$next"
echo "tagged $next (was $latest)"
echo "publish it with: git push origin $next"
