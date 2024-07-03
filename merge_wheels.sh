#!/bin/bash
# merges the lib and bin wheels into a single wheel
# maturin currently doesn't support building multiple wheels at once
# see https://github.com/PyO3/maturin/issues/368
set -e

TMPDIR="build_tmp"

# Grab Info
file_name_lib=$(basename "$(ls "$1"/*.whl)")
file_name_bin=$(basename "$(ls "$2"/*.whl)")
echo $file_name_bin
dist_info=$(unzip -qql "$1/*.whl" | grep "\.dist-info/METADATA" | awk '{print $4}' | cut -d/ -f1)
name_version=$(basename -s '.dist-info' $dist_info)

# Merge wheel
mkdir -p "$TMPDIR"
unzip -qo "$1/$file_name_lib" -d "$TMPDIR"
unzip -qo "$2/$file_name_bin" -d "$TMPDIR"

# Merge record
unzip -qjo "$1/$file_name_lib" "*.dist-info/RECORD" -d "$1"
unzip -qjo "$2/$file_name_bin" "*.dist-info/RECORD" -d "$2"
cat "$1/RECORD" "$2/RECORD" | sort | uniq > "$TMPDIR/$name_version.dist-info/RECORD"

# Create the wheel

mkdir -p "$3"
cd "$TMPDIR"
zip -qr "../$3/$file_name_lib" *
cd ..
rm -rf "$TMPDIR"