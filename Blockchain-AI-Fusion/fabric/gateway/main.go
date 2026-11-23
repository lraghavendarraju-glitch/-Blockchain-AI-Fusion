package main

import (
	"crypto/tls"
	"crypto/x509"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"log"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/hyperledger/fabric-gateway/pkg/client"
	"github.com/hyperledger/fabric-gateway/pkg/identity"
	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
)

type Server struct {
	contract *client.Contract
	secret   string
}

func required(name string) string {
	v := os.Getenv(name)
	if v == "" {
		log.Fatalf("missing required environment variable %s", name)
	}
	return v
}

func main() {
	msp := required("FABRIC_MSP_ID")
	peer := required("FABRIC_PEER_ENDPOINT")
	serverName := required("FABRIC_PEER_HOST_ALIAS")
	tlsBytes, err := os.ReadFile(required("FABRIC_TLS_CERT"))
	if err != nil {
		log.Fatal(err)
	}
	roots := x509.NewCertPool()
	if !roots.AppendCertsFromPEM(tlsBytes) {
		log.Fatal("invalid TLS certificate")
	}
	connection, err := grpc.NewClient(peer, grpc.WithTransportCredentials(credentials.NewTLS(&tls.Config{
		RootCAs: roots, ServerName: serverName, MinVersion: tls.VersionTLS12,
	})))
	if err != nil {
		log.Fatal(err)
	}
	defer connection.Close()
	certPEM, err := os.ReadFile(required("FABRIC_CLIENT_CERT"))
	if err != nil {
		log.Fatal(err)
	}
	certificate, err := identity.CertificateFromPEM(certPEM)
	if err != nil {
		log.Fatal(err)
	}
	id, err := identity.NewX509Identity(msp, certificate)
	if err != nil {
		log.Fatal(err)
	}
	keyPEM, err := os.ReadFile(required("FABRIC_CLIENT_KEY"))
	if err != nil {
		log.Fatal(err)
	}
	privateKey, err := identity.PrivateKeyFromPEM(keyPEM)
	if err != nil {
		log.Fatal(err)
	}
	sign, err := identity.NewPrivateKeySign(privateKey)
	if err != nil {
		log.Fatal(err)
	}
	gateway, err := client.Connect(id, client.WithSign(sign), client.WithClientConnection(connection),
		client.WithEvaluateTimeout(15*time.Second), client.WithEndorseTimeout(30*time.Second),
		client.WithSubmitTimeout(30*time.Second), client.WithCommitStatusTimeout(90*time.Second))
	if err != nil {
		log.Fatal(err)
	}
	defer gateway.Close()
	network := gateway.GetNetwork(envDefault("FABRIC_CHANNEL", "mychannel"))
	s := Server{contract: network.GetContract(envDefault("FABRIC_CHAINCODE", "wfgov")),
		secret: required("BLOCKAI_GATEWAY_TOKEN")}
	mux := http.NewServeMux()
	mux.HandleFunc("/health", s.auth(s.health))
	mux.HandleFunc("/workflows", s.auth(s.register))
	mux.HandleFunc("/steps", s.auth(s.step))
	mux.HandleFunc("/workflows/", s.auth(s.get))
	mux.HandleFunc("/audit", s.auth(s.audit))
	mux.HandleFunc("/trust", s.auth(s.trust))
	mux.HandleFunc("/private/parameters", s.auth(s.privateParameters))
	address := envDefault("GATEWAY_LISTEN", "127.0.0.1:8080")
	log.Printf("Fabric Gateway REST bridge listening on %s", address)
	server := &http.Server{Addr: address, Handler: mux, ReadHeaderTimeout: 5 * time.Second}
	log.Fatal(server.ListenAndServe())
}

func envDefault(name, def string) string {
	if v := os.Getenv(name); v != "" {
		return v
	}
	return def
}
func (s *Server) auth(next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("Authorization") != "Bearer "+s.secret {
			http.Error(w, "unauthorized", http.StatusUnauthorized)
			return
		}
		next(w, r)
	}
}
func respond(w http.ResponseWriter, obj any) {
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(obj)
}
func failure(w http.ResponseWriter, err error) {
	http.Error(w, fmt.Sprintf(`{"error":%q}`, err.Error()), http.StatusBadRequest)
}
func readJSON(r *http.Request, v any) error {
	if r.ContentLength > 1024*1024 {
		return errors.New("request too large")
	}
	r.Body = http.MaxBytesReader(nil, r.Body, 1024*1024)
	return json.NewDecoder(r.Body).Decode(v)
}
func (s *Server) health(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "method", 405)
		return
	}
	respond(w, map[string]string{"status": "ready", "backend": "hyperledger_fabric_gateway"})
}
func (s *Server) register(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method", 405)
		return
	}
	var req struct {
		WorkflowID string `json:"workflow_id"`
		TemplateID int    `json:"template_id"`
	}
	if err := readJSON(r, &req); err != nil {
		failure(w, err)
		return
	}
	result, err := s.contract.SubmitTransaction("RegisterWorkflow", req.WorkflowID, strconv.Itoa(req.TemplateID))
	if err != nil {
		failure(w, err)
		return
	}
	respond(w, map[string]string{"result": string(result), "status": "committed"})
}
func (s *Server) step(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method", 405)
		return
	}
	var record map[string]any
	if err := readJSON(r, &record); err != nil {
		failure(w, err)
		return
	}
	privateText, hasPrivate := record["private_payload_b64"].(string)
	delete(record, "private_payload_b64")
	data, err := json.Marshal(record)
	if err != nil {
		failure(w, err)
		return
	}
	var result []byte
	if hasPrivate {
		privateData, decodeErr := base64.StdEncoding.DecodeString(privateText)
		if decodeErr != nil {
			failure(w, decodeErr)
			return
		}
		if len(privateData) > 262144 || !json.Valid(privateData) {
			failure(w, errors.New("invalid private payload"))
			return
		}
		result, err = s.contract.Submit("SubmitStep", client.WithArguments(string(data)),
			client.WithTransient(map[string][]byte{"params": privateData}))
	} else {
		result, err = s.contract.SubmitTransaction("SubmitStep", string(data))
	}
	if err != nil {
		failure(w, err)
		return
	}
	respond(w, map[string]string{"tx_id": string(result), "status": "committed"})
}
func (s *Server) get(w http.ResponseWriter, r *http.Request) {
	if r.Method != "GET" {
		http.Error(w, "method", 405)
		return
	}
	id := strings.TrimPrefix(r.URL.Path, "/workflows/")
	if id == "" {
		http.Error(w, "missing workflow id", 400)
		return
	}
	result, err := s.contract.EvaluateTransaction("GetWorkflow", id)
	if err != nil {
		failure(w, err)
		return
	}
	w.Header().Set("Content-Type", "application/json")
	_, _ = w.Write(result)
}
func (s *Server) audit(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method", 405)
		return
	}
	var req struct {
		WorkflowID string `json:"workflow_id"`
		ReportHash string `json:"report_hash"`
	}
	if err := readJSON(r, &req); err != nil {
		failure(w, err)
		return
	}
	result, err := s.contract.SubmitTransaction("RecordAudit", req.WorkflowID, req.ReportHash)
	if err != nil {
		failure(w, err)
		return
	}
	respond(w, map[string]string{"result": string(result), "status": "committed"})
}
func (s *Server) trust(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method", 405)
		return
	}
	var req struct {
		WorkflowID string  `json:"workflow_id"`
		Trust      float64 `json:"trust"`
	}
	if err := readJSON(r, &req); err != nil {
		failure(w, err)
		return
	}
	result, err := s.contract.SubmitTransaction("UpdateTrust", req.WorkflowID, strconv.FormatFloat(req.Trust, 'f', 6, 64))
	if err != nil {
		failure(w, err)
		return
	}
	respond(w, map[string]string{"result": string(result), "status": "committed"})
}

func (s *Server) privateParameters(w http.ResponseWriter, r *http.Request) {
	if r.Method != "POST" {
		http.Error(w, "method", 405)
		return
	}
	var req struct {
		WorkflowID string `json:"workflow_id"`
		StepID     string `json:"step_id"`
		PayloadB64 string `json:"payload_b64"`
	}
	if err := readJSON(r, &req); err != nil {
		failure(w, err)
		return
	}
	payload, err := base64.StdEncoding.DecodeString(req.PayloadB64)
	if err != nil {
		failure(w, err)
		return
	}
	if len(payload) > 262144 || !json.Valid(payload) {
		failure(w, errors.New("invalid private JSON"))
		return
	}
	result, err := s.contract.Submit("StorePrivateParameters", client.WithArguments(req.WorkflowID, req.StepID),
		client.WithTransient(map[string][]byte{"params": payload}))
	if err != nil {
		failure(w, err)
		return
	}
	respond(w, map[string]string{"result": string(result), "status": "committed"})
}
