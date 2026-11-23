package main

import (
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"

	"github.com/hyperledger/fabric-chaincode-go/pkg/statebased"
	"github.com/hyperledger/fabric-contract-api-go/contractapi"
)

// Workflow step claims are signed by Fabric's X.509 gateway identity.
// On-chain hashes are commitments; the chaincode cannot verify absent off-chain bytes.
type StepRecord struct {
	WorkflowID       string   `json:"workflow_id"`
	StepID           string   `json:"step_id"`
	StepType         string   `json:"step_type"`
	ParentIDs        []string `json:"parent_ids"`
	InputHash        string   `json:"input_hash"`
	OutputHash       string   `json:"output_hash"`
	ArtifactHash     string   `json:"artifact_hash"`
	ParamsCommitment string   `json:"params_commitment"`
	MetricCommitment string   `json:"metric_commitment"`
	ActorPseudonym   string   `json:"actor_pseudonym"`
	Timestamp        int64    `json:"timestamp"`
	ExecutionIndex   int      `json:"execution_index"`
	ArtifactVersion  int      `json:"artifact_version"`
	ModelHash        string   `json:"model_hash"`
	CertifiedRole    string   `json:"certified_role"`
	MSPID            string   `json:"msp_id"`
}

type WorkflowState struct {
	ID              string  `json:"workflow_id"`
	TemplateID      int     `json:"template_id"`
	StepCount       int     `json:"step_count"`
	Trust           float64 `json:"trust"`
	RequiresAuditor bool    `json:"requires_auditor"`
	CreatedByMSP    string  `json:"created_by_msp"`
}

type WorkflowView struct {
	Workflow WorkflowState `json:"workflow"`
	Steps    []StepRecord  `json:"steps"`
}

type SmartContract struct{ contractapi.Contract }

func validHash(s string) bool {
	if len(s) != 64 {
		return false
	}
	_, err := hex.DecodeString(s)
	return err == nil
}

func allowedRole(role, stepType string) bool {
	policies := map[string][]string{
		"ingest":           {"data_engineer", "administrator"},
		"validate":         {"data_engineer", "reviewer", "administrator"},
		"preprocess":       {"data_engineer", "data_scientist", "administrator"},
		"feature_engineer": {"data_scientist", "ml_engineer", "administrator"},
		"train":            {"data_scientist", "ml_engineer", "administrator"},
		"tune":             {"data_scientist", "ml_engineer", "administrator"},
		"evaluate":         {"reviewer", "data_scientist", "administrator"},
		"register":         {"ml_engineer", "administrator"},
		"deploy":           {"mlops_engineer", "administrator"},
		"monitor":          {"mlops_engineer", "auditor", "administrator"},
		"retrain":          {"ml_engineer", "administrator"},
		"audit":            {"auditor", "administrator"},
	}
	for _, candidate := range policies[stepType] {
		if candidate == role {
			return true
		}
	}
	return false
}

func certifiedRole(ctx contractapi.TransactionContextInterface) (string, error) {
	role, found, err := ctx.GetClientIdentity().GetAttributeValue("role")
	if err != nil {
		return "", err
	}
	if found && role != "" {
		return role, nil
	}
	// In Fabric's sample network, a *certificate with OU=admin* may be used
	// for development. A client-supplied role is NEVER accepted as evidence.
	cert, err := ctx.GetClientIdentity().GetX509Certificate()
	if err != nil {
		return "", err
	}
	for _, ou := range cert.Subject.OrganizationalUnit {
		if strings.EqualFold(ou, "admin") {
			return "administrator", nil
		}
	}
	return "", errors.New("X.509 certificate is missing a certified 'role' attribute (or admin OU)")
}

func (s *SmartContract) RegisterWorkflow(ctx contractapi.TransactionContextInterface, workflowID string, templateText string) error {
	if workflowID == "" || strings.ContainsRune(workflowID, '\x00') {
		return errors.New("invalid workflow id")
	}
	templateID, err := strconv.Atoi(templateText)
	if err != nil {
		return err
	}
	key := "wf:" + workflowID
	prior, err := ctx.GetStub().GetState(key)
	if err != nil {
		return err
	}
	if prior != nil {
		return fmt.Errorf("workflow %s already exists", workflowID)
	}
	msp, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return err
	}
	state := WorkflowState{ID: workflowID, TemplateID: templateID, Trust: 1., CreatedByMSP: msp}
	encoded, err := json.Marshal(state)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, encoded)
}

func (s *SmartContract) SubmitStep(ctx contractapi.TransactionContextInterface, recordJSON string) (string, error) {
	var rec StepRecord
	if err := json.Unmarshal([]byte(recordJSON), &rec); err != nil {
		return "", err
	}
	if rec.WorkflowID == "" || rec.StepID == "" || rec.StepType == "" {
		return "", errors.New("missing required ids/type")
	}
	if strings.ContainsRune(rec.WorkflowID, '\x00') || strings.ContainsRune(rec.StepID, '\x00') {
		return "", errors.New("invalid key")
	}
	for _, h := range []string{rec.InputHash, rec.OutputHash, rec.ArtifactHash, rec.ParamsCommitment, rec.MetricCommitment} {
		if !validHash(h) {
			return "", errors.New("invalid SHA-256 commitment")
		}
	}
	if rec.OutputHash != rec.ArtifactHash {
		return "", errors.New("artifact/output commitment mismatch")
	}
	role, err := certifiedRole(ctx)
	if err != nil {
		return "", err
	}
	if !allowedRole(role, rec.StepType) {
		return "", fmt.Errorf("access denied: role %s step %s", role, rec.StepType)
	}
	msp, err := ctx.GetClientIdentity().GetMSPID()
	if err != nil {
		return "", err
	}
	key := "wf:" + rec.WorkflowID
	rawState, err := ctx.GetStub().GetState(key)
	if err != nil {
		return "", err
	}
	if rawState == nil {
		return "", errors.New("workflow not registered")
	}
	var state WorkflowState
	if err := json.Unmarshal(rawState, &state); err != nil {
		return "", err
	}
	if rec.ExecutionIndex != state.StepCount {
		return "", fmt.Errorf("sequence index %d expected %d", rec.ExecutionIndex, state.StepCount)
	}
	// Fabric state-based endorsement on the workflow key enforces extra auditor
	// participation when the trust update enabled a strict policy.
	prior, err := ctx.GetStub().CreateCompositeKey("step", []string{rec.WorkflowID, rec.StepID})
	if err != nil {
		return "", err
	}
	exists, err := ctx.GetStub().GetState(prior)
	if err != nil {
		return "", err
	}
	if exists != nil {
		return "", errors.New("duplicate step id")
	}
	expected := make([][2]string, 0, len(rec.ParentIDs))
	for _, id := range rec.ParentIDs {
		parentKey, err := ctx.GetStub().CreateCompositeKey("step", []string{rec.WorkflowID, id})
		if err != nil {
			return "", err
		}
		parentBytes, err := ctx.GetStub().GetState(parentKey)
		if err != nil {
			return "", err
		}
		if parentBytes == nil {
			return "", fmt.Errorf("missing predecessor %s", id)
		}
		var parent StepRecord
		if err := json.Unmarshal(parentBytes, &parent); err != nil {
			return "", err
		}
		expected = append(expected, [2]string{id, parent.OutputHash})
	}
	sort.Slice(expected, func(i, j int) bool { return expected[i][0] < expected[j][0] })
	digestJSON, err := json.Marshal(expected)
	if err != nil {
		return "", err
	}
	digest := sha256.Sum256(digestJSON)
	if rec.InputHash != hex.EncodeToString(digest[:]) {
		return "", errors.New("parent hash commitment mismatch")
	}
	// Optional transient private payload is validated and written atomically
	// with the workflow step. No parameter values enter public block arguments.
	transient, err := ctx.GetStub().GetTransient()
	if err != nil {
		return "", err
	}
	if privateBytes, ok := transient["params"]; ok {
		if !json.Valid(privateBytes) || len(privateBytes) > 262144 {
			return "", errors.New("invalid transient private parameters")
		}
		digest := sha256.Sum256(privateBytes)
		if hex.EncodeToString(digest[:]) != rec.ParamsCommitment {
			return "", errors.New("private commitment mismatch")
		}
		if err := ctx.GetStub().PutPrivateData("parametersPrivateCollection", prior, privateBytes); err != nil {
			return "", err
		}
	}
	rec.CertifiedRole = role
	rec.MSPID = msp
	encoded, err := json.Marshal(rec)
	if err != nil {
		return "", err
	}
	if err := ctx.GetStub().PutState(prior, encoded); err != nil {
		return "", err
	}
	state.StepCount++
	encodedState, err := json.Marshal(state)
	if err != nil {
		return "", err
	}
	if err := ctx.GetStub().PutState(key, encodedState); err != nil {
		return "", err
	}
	return ctx.GetStub().GetTxID(), nil
}

func (s *SmartContract) GetWorkflow(ctx contractapi.TransactionContextInterface, workflowID string) (*WorkflowView, error) {
	raw, err := ctx.GetStub().GetState("wf:" + workflowID)
	if err != nil {
		return nil, err
	}
	if raw == nil {
		return nil, fmt.Errorf("workflow not found: %s", workflowID)
	}
	var state WorkflowState
	if err := json.Unmarshal(raw, &state); err != nil {
		return nil, err
	}
	iterator, err := ctx.GetStub().GetStateByPartialCompositeKey("step", []string{workflowID})
	if err != nil {
		return nil, err
	}
	defer iterator.Close()
	steps := []StepRecord{}
	for iterator.HasNext() {
		entry, err := iterator.Next()
		if err != nil {
			return nil, err
		}
		var sr StepRecord
		if err := json.Unmarshal(entry.Value, &sr); err != nil {
			return nil, err
		}
		steps = append(steps, sr)
	}
	sort.Slice(steps, func(i, j int) bool { return steps[i].ExecutionIndex < steps[j].ExecutionIndex })
	return &WorkflowView{Workflow: state, Steps: steps}, nil
}

// Strict endorsement requires Auditor + two additional organizations. The
// four-org deployment must register exactly these MSP names. SBE is installed
// on workflow state; each SubmitStep reads/writes this state key.
func (s *SmartContract) UpdateTrust(ctx contractapi.TransactionContextInterface, workflowID string, trust float64) error {
	if trust < 0 || trust > 1 {
		return errors.New("trust outside [0,1]")
	}
	role, err := certifiedRole(ctx)
	if err != nil {
		return err
	}
	if role != "auditor" && role != "administrator" {
		return errors.New("trust update requires auditor or administrator")
	}
	key := "wf:" + workflowID
	raw, err := ctx.GetStub().GetState(key)
	if err != nil {
		return err
	}
	if raw == nil {
		return errors.New("workflow not found")
	}
	var st WorkflowState
	if err := json.Unmarshal(raw, &st); err != nil {
		return err
	}
	st.Trust = trust
	if trust < .55 && !st.RequiresAuditor {
		ep, err := statebased.NewStateEP(nil)
		if err != nil {
			return err
		}
		if err := ep.AddOrgs(statebased.RoleTypePeer, "DataEngineeringMSP", "ModelDevelopmentMSP", "AuditMSP"); err != nil {
			return err
		}
		bytes, err := ep.Policy()
		if err != nil {
			return err
		}
		if err := ctx.GetStub().SetStateValidationParameter(key, bytes); err != nil {
			return err
		}
		st.RequiresAuditor = true
	} else if trust >= .55 && st.RequiresAuditor {
		if err := ctx.GetStub().SetStateValidationParameter(key, nil); err != nil {
			return err
		}
		st.RequiresAuditor = false
	}
	blob, err := json.Marshal(st)
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, blob)
}

// StorePrivateParameters places salted parameter values in Fabric private
// collections. The private bytes MUST be passed as transient data, and their
// SHA-256 digest MUST match the commitment submitted with the step.
func (s *SmartContract) StorePrivateParameters(ctx contractapi.TransactionContextInterface, workflowID string, stepID string) error {
	role, err := certifiedRole(ctx)
	if err != nil {
		return err
	}
	if role == "external_collaborator" {
		return errors.New("private parameter access denied")
	}
	transient, err := ctx.GetStub().GetTransient()
	if err != nil {
		return err
	}
	privateBytes, ok := transient["params"]
	if !ok || !json.Valid(privateBytes) {
		return errors.New("missing/invalid transient params")
	}
	if len(privateBytes) > 262144 {
		return errors.New("private parameters too large")
	}
	stepKey, err := ctx.GetStub().CreateCompositeKey("step", []string{workflowID, stepID})
	if err != nil {
		return err
	}
	committed, err := ctx.GetStub().GetState(stepKey)
	if err != nil {
		return err
	}
	if committed == nil {
		return errors.New("step not committed")
	}
	var rec StepRecord
	if err := json.Unmarshal(committed, &rec); err != nil {
		return err
	}
	digest := sha256.Sum256(privateBytes)
	if hex.EncodeToString(digest[:]) != rec.ParamsCommitment {
		return errors.New("private parameters do not match on-chain commitment")
	}
	// Collection policy, in addition to the X.509 role, restricts the peers
	// allowed to distribute private content.
	return ctx.GetStub().PutPrivateData("parametersPrivateCollection", stepKey, privateBytes)
}

func (s *SmartContract) RecordAudit(ctx contractapi.TransactionContextInterface, workflowID string, reportHash string) error {
	role, err := certifiedRole(ctx)
	if err != nil {
		return err
	}
	if role != "auditor" && role != "administrator" {
		return errors.New("audit requires auditor role")
	}
	if !validHash(reportHash) {
		return errors.New("invalid audit report hash")
	}
	key, err := ctx.GetStub().CreateCompositeKey("audit", []string{workflowID, ctx.GetStub().GetTxID()})
	if err != nil {
		return err
	}
	return ctx.GetStub().PutState(key, []byte(reportHash))
}

func main() {
	cc, err := contractapi.NewChaincode(new(SmartContract))
	if err != nil {
		panic(err)
	}
	if err := cc.Start(); err != nil {
		panic(err)
	}
}
