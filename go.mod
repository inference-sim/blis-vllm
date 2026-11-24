module github.com/inference-sim/blis-vllm

go 1.27.2

// TEMPORARY: user-approved remote replace exception until
// https://github.com/inference-sim/blis-substrate/pull/417 is merged and pinned upstream.
replace github.com/inference-sim/blis-substrate => github.com/VincyZhang/blis-substrate v0.0.0-20261124030424-064e101ab511

require (
	github.com/inference-sim/blis-substrate v0.15.1-0.20261118162118-b6748ca9d757
	github.com/sirupsen/logrus v1.9.4
	github.com/stretchr/testify v1.12.1
	gopkg.in/yaml.v3 v3.0.1
	pgregory.net/rapid v1.2.0
)

require (
	go.yaml.in/yaml/v3 v3.0.5 // indirect
	golang.org/x/sys v0.13.0 // indirect
)
