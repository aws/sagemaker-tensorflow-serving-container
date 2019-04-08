#!/bin/bash
#
# Script to build all versions of TFS and EIA containers.

tfs_versions=( "1.11.1" "1.12.0" "1.13.0" )
tfs_arches=( "cpu" "gpu" )
eia_versions=( "1.11" "1.12" )

for version in "${tfs_versions[@]}"
do
    for arch in "${tfs_arches[@]}"
    do
        echo "Building TFS container for version $version arch $arch"
        ./scripts/build.sh --version $version --arch $arch
    done
done

for eia_version in "${eia_versions[@]}"
do
    echo "Building EIA container for version $eia_version"
    ./scripts/build.sh --version $eia_version --arch eia
done
