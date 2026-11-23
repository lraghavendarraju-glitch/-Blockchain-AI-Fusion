# Hyperledger Fabric deployment: Go chaincode and Gateway service

## Scope and verification status

This project contains **actual** Fabric contract and Gateway client source code. The provided integration harness uses the **official two-organization Fabric test network**; it is not the 4-organization/8-peer/3-Raft-orderer configuration reported in the manuscript. The Python smoke tests do not require Fabric. The Go modules and Docker network cannot be built or executed in restricted/offline environments without Fabric binaries, Docker and Go module dependencies. Validate these elements on the target host before publishing a benchmark.

## 1. Prerequisites

- Docker Engine/Desktop with Docker Compose.
- Go 1.23+ for the pinned Go modules. The Fabric Gateway release here is compatible with Go 1.23.
- Official Fabric 2.5 binary tools, Docker images and matching `fabric-samples` test-network.
- Recommended reference: https://hyperledger-fabric.readthedocs.io/en/release-2.5/test_network.html
- Official Gateway: https://hyperledger-fabric.readthedocs.io/en/release-2.5/gateway.html

## 2. Download the official sample network and Go dependencies

```bash
git clone --branch release-2.5 https://github.com/hyperledger/fabric-samples.git fabric/network/fabric-samples
# Install the Fabric 2.5 binaries and Docker images following the official documentation.
# With internet available:
(cd fabric/chaincode/wfgov && go mod tidy && go build ./...)
(cd fabric/gateway && go mod tidy && go build ./...)
```

If the `release-2.5` branch is not present in your cloned repository, use an official Fabric 2.5 release tag. Do not mix Fabric 3.x test-network configuration with the manuscript's 2.5.x setup.

## 3. Start the development Fabric network and deploy wfgov

```bash
export FABRIC_SAMPLES="$PWD/fabric/network/fabric-samples"
./fabric/network/start-test-network.sh
```

The helper runs `network.sh up createChannel -c mychannel -ca` and `network.sh deployCC -ccn wfgov -ccp <repo>/fabric/chaincode/wfgov -ccl go -c mychannel -cccg <repo>/fabric/network/collections-config.json`. The collection `parametersPrivateCollection` permits Org1/Org2 and is configured with a limited `blockToLive`.

## 4. Configure and launch the Go Gateway REST bridge

Configure a **certificate-bearing administrator** for the local demonstration. On a real multi-user network, enroll clients with an X.509 `role` attribute corresponding to the allowed workflow step role; the chaincode checks the attribute. Admin certificates with `OU=admin` are accepted as a development convenience, not as a substitute for user-level RBAC evaluation.

The following is an example for the official Org1 sample network (adapt paths to your Fabric sample version):

```bash
export FABRIC_SAMPLES="$PWD/fabric/network/fabric-samples"
export ORG1="$FABRIC_SAMPLES/test-network/organizations/peerOrganizations/org1.example.com"
export FABRIC_MSP_ID=Org1MSP
export FABRIC_PEER_ENDPOINT=localhost:7051
export FABRIC_PEER_HOST_ALIAS=peer0.org1.example.com
export FABRIC_TLS_CERT="$ORG1/peers/peer0.org1.example.com/tls/ca.crt"
export FABRIC_CLIENT_CERT="$ORG1/users/Admin@org1.example.com/msp/signcerts/cert.pem"
export FABRIC_CLIENT_KEY="$(find "$ORG1/users/Admin@org1.example.com/msp/keystore" -type f | head -1)"
export FABRIC_CHANNEL=mychannel
export FABRIC_CHAINCODE=wfgov
export GATEWAY_LISTEN=127.0.0.1:8080
export BLOCKAI_GATEWAY_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
(cd fabric/gateway && go run .)
```

In a second terminal, **export the same `BLOCKAI_GATEWAY_TOKEN`** as in the Gateway terminal. Never use an open `0.0.0.0` listener without strong TLS/authentication protections.

```bash
python -m blockai generate --n 10 --output datasets/synthetic/small.jsonl.gz
python -m blockai ledger --input datasets/synthetic/small.jsonl.gz \
  --backend fabric --gateway http://127.0.0.1:8080 --limit 1 --store-private \
  --output results/fabric_demo
```

Successful transactions are signed and submitted to a Fabric peer through the official Gateway API, and a commit status is awaited. Chaincode validation failures return errors and are recorded by the Python client as **off-chain rejected attempts**. They are not falsely represented as accepted on-chain records.

**Private parameters**: `--store-private` sends the canonical salted private payload via the Fabric transient map as part of the same step transaction. Chaincode recomputes SHA-256, compares it with the public commitment, and writes the private bytes to `parametersPrivateCollection`. Public blocks never include the raw parameters.

## 5. Adaptive endorsement / four-organization network

The 4-organization manuscript topology requires additional infrastructure beyond the bundled demo:

- Eight peers (two per organization), three separate Raft orderers, an application channel, CA/MSP identities, organization TLS settings and a source-of-truth organizational role policy.
- Four application MSPs. The chaincode's low-trust endorsement policy expects exact MSP identifiers `DataEngineeringMSP`, `ModelDevelopmentMSP`, and `AuditMSP` in its `UpdateTrust` implementation. Configure them consistently at installation, or deliberately revise both network channel and chaincode policy.
- Normal operation should define a **three-of-four** chaincode endorsement policy. After `UpdateTrust(workflowID, trust<0.55)`, the workflow state key is assigned a stricter state-based policy **requiring the AuditMSP plus the two named organizations**. Since `SubmitStep` writes the workflow state key, future steps must satisfy that policy.
- The two-organization demonstration lacks `AuditMSP`. Therefore DO NOT invoke low-trust endorsement changes there without reconfiguring the expected MSPs; use `governance-demo` in Python to exercise the control loop locally.
- Cross-organization identities, per-actor certificate revocation, peer conflict measurements and WAN netem experiments must be tested and reported only after external provisioning.

## 6. Contract operations and data policy

| Chaincode method | Purpose |
|---|---|
| `RegisterWorkflow(id, template_id)` | Register one workflow |
| `SubmitStep(step_record_json)` | Validate certified role, position, parents, hashes, private commitments and commit step |
| `GetWorkflow(id)` | Retrieve committed DAG nodes and workflow state |
| `StorePrivateParameters(id, step_id)` | Alternative private collection upload via transient map |
| `UpdateTrust(id, score)` | Change trust score and (on compatible four-org network) endorsement policy |
| `RecordAudit(id, audit_hash)` | Auditor-gated audit-report hash |

Public state contains pseudonyms, IDs, hashes, timestamps and commitments. It does not contain artifact bytes or plaintext parameters. The identity vault in the Python research simulator is independent of the X.509 certs used by real Fabric. The Go Gateway service has **one enrolled client identity**: to evaluate user-specific access denial, deploy per-actor credentials and route events through identities rather than assuming a single administrator reproduces the manuscript's access-control threat model.

## 7. Benchmarking requirements

Record peer-side block commit times, client end-to-end commit latency, validation failures, accepted/invalid transaction counts, throughput over steady intervals with warm-up, offered load, block parameters, orderer count, peer count, and network RTT. A 2-org, one-orderer sample is **not** an 8-peer, 3-Raft-orderer result; no such measurements are in the ZIP. Use equivalent-load baselines and include error bars/ten run statistics.
