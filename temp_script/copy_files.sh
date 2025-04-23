#!/bin/bash
for file in /scr/shared/datasets/LIBERO/libero_10_highres/*.hdf5; do
  basename=$(basename "$file")
  target=$(readlink "$file")
  target_basename=$(basename "$target")
  source_file="/scr/kimkj/.cache/huggingface/hub/datasets--jesbu1--libero_openvla_roboverse_hdf5/blobs/$target_basename"
  if [ -f "$source_file" ]; then
    cp "$source_file" "/scr/kimkj/molmo-handtracker/libero_10_highres_fixed/$basename"
    echo "Copied $basename"
  else
    echo "Source file for $basename not found"
  fi
done
