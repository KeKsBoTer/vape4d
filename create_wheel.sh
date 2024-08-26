#!/bin/bash

set -e

TMPDIR="build_tmp"
mkdir -p "$TMPDIR"

# Build the wheel
# Note that for my specific use case, "python" feature is needed. You might want to change it.
maturin build -F python -F colormaps --release --bindings pyo3 -o "dist_lib" $@
maturin build -F python -F colormaps --release --bindings bin  -o "dist_bin" $@

# # Grab Info
# file_name_lib=$(basename $(/bin/ls "$TMPDIR/lib"/*.whl))
# file_name_bin=$(basename $(/bin/ls "$TMPDIR/bin"/*.whl))
# dist_info=$(unzip -qql "$TMPDIR/lib/*.whl" | grep "\.dist-info/METADATA" | awk '{print $4}' | cut -d/ -f1)
# name_version=$(basename -s '.dist-info' $dist_info)

# # Merge wheel
# mkdir -p "$TMPDIR/merged"
# unzip -qo "$TMPDIR/lib/$file_name_lib" -d "$TMPDIR/merged"
# unzip -qo "$TMPDIR/bin/$file_name_bin" -d "$TMPDIR/merged"

# # Merge record
# unzip -qjo "$TMPDIR/lib/$file_name_lib" "*.dist-info/RECORD" -d "$TMPDIR/lib"
# unzip -qjo "$TMPDIR/bin/$file_name_bin" "*.dist-info/RECORD" -d "$TMPDIR/bin"
# cat "$TMPDIR/lib/RECORD" "$TMPDIR/bin/RECORD" | sort | uniq > "$TMPDIR/merged/$name_version.dist-info/RECORD"

# # Create the wheel

# cd "$TMPDIR/merged"
# zip -qr "../../$file_name_lib" *
# cd ../..
# rm -rf "$TMPDIR"