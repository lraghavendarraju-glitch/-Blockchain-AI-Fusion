#!/usr/bin/env bash
set -euo pipefail
# Real Fabric test-network requires Docker and the official binary distributions.
# This is a two-organization development network, not the paper's 4-org topology.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SAMPLES="${FABRIC_SAMPLES:-$ROOT/fabric/network/fabric-samples}"
command -v docker >/dev/null || { echo 'Docker is required' >&2; exit 1; }
if [[ ! -f "$SAMPLES/test-network/network.sh" ]]; then
    echo 'Clone fabric-samples at the chosen Fabric 2.5 release into:' >&2
    echo " $SAMPLES" >&2
    echo 'See fabric/README.md for commands and binary installation.' >&2
    exit 1
fi
cd "$SAMPLES/test-network"
./network.sh up createChannel -c mychannel -ca
./network.sh deployCC -ccn wfgov -ccp "$ROOT/fabric/chaincode/wfgov" -ccl go -c mychannel -cccg "$ROOT/fabric/network/collections-config.json"
printf 'wfgov deployed on two-org development test network.\n'
