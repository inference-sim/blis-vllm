// Original to blis-vllm.

package engine_test

import (
	"testing"

	"github.com/inference-sim/blis-vllm/compat/engine"
)

// admissionModel prices admission as a fixed delay plus a per-token part, and says which part
// holds a render thread, as the kernel does (fixed 40 ms, per-token 0.35 us).
type admissionModel struct {
	fixedUs   int64
	perTokenN int64 // nanoseconds per prompt token
}

func (m admissionModel) tokenUs(r *engine.Request) int64 {
	return m.perTokenN * int64(len(r.InputTokens)) / 1000
}
func (m admissionModel) StepTime([]*engine.Request) int64     { return 1000 }
func (m admissionModel) QueueingTime(r *engine.Request) int64 { return m.fixedUs + m.tokenUs(r) }
func (m admissionModel) RenderTime(r *engine.Request) int64   { return m.tokenUs(r) }
func (m admissionModel) OutputTokenProcessingTime() int64     { return 0 }
func (m admissionModel) PostDecodeFixedOverhead() int64       { return 0 }

func admissionEngine(t *testing.T, m engine.LatencyModel, workers int) *engine.EngineCore {
	t.Helper()
	cfg := engine.SimConfig{
		KVCacheConfig: engine.NewKVCacheConfig(64, 16, 0, 0, 0, 0),
		BatchConfig:   engine.NewBatchConfig(16, 2048, 0),
		Frontend:      engine.FrontendConfig{RendererNumWorkers: workers},
	}
	e, err := engine.NewEngineCore(cfg, engine.MustNewKVStoreFromConfig(cfg.KVCacheConfig), m)
	if err != nil {
		t.Fatal(err)
	}
	return e
}

func promptOf(n int) *engine.Request {
	return &engine.Request{ID: "r", InputTokens: make([]engine.TokenID, n)}
}

// lastQueued admits n simultaneous requests at time 0 and returns when the last one is queued.
func lastQueued(e *engine.EngineCore, n, tokens int) int64 {
	var at int64
	for i := 0; i < n; i++ {
		at = e.QueuedAt(promptOf(tokens), 0)
	}
	return at
}

// The 40 ms per-request term is a delay, not thread time: a hundred simultaneous 1k-token
// requests on one render thread are queued after 40 ms plus a hundred tokenizations
// (0.35 ms each), not after a hundred times 40 ms.
func TestFixedAdmissionDelayDoesNotSerialiseOnTheRenderThread(t *testing.T) {
	m := admissionModel{fixedUs: 40_000, perTokenN: 350}
	e := admissionEngine(t, m, 1)
	got := lastQueued(e, 100, 1000)
	perReq := m.RenderTime(promptOf(1000)) // 350 us
	if want := 40_000 + 100*perReq; got != want {
		t.Fatalf("last request queued at %d us, want %d us (delay once, tokenizer per request)", got, want)
	}
	// Throughput: admitting 1000 requests takes the tokenizer's time, so the endpoint can start
	// far more than the 25 requests a second a 40 ms service time would allow.
	e = admissionEngine(t, m, 1)
	span := lastQueued(e, 1000, 1000)
	if rate := 1000 * 1e6 / float64(span); rate < 2000 {
		t.Fatalf("one render thread starts %.0f requests a second, want >= 2000", rate)
	}
}

// One request alone sees the whole admission overhead whether or not there is a render stage.
func TestSingleRequestAdmissionIsTheSameWithAndWithoutARenderStage(t *testing.T) {
	m := admissionModel{fixedUs: 40_000, perTokenN: 350}
	want := m.QueueingTime(promptOf(1000))
	for _, workers := range []int{0, 1, 4} {
		e := admissionEngine(t, m, workers)
		if got := e.QueuedAt(promptOf(1000), 7_000); got != 7_000+want {
			t.Errorf("workers=%d: queued at %d, want %d", workers, got, 7_000+want)
		}
	}
}

// The cap stays for the part that is thread work: with a tokenizer that costs 10 ms a request,
// one thread starts 100 requests a second and four start 400, whatever the fixed delay.
func TestRenderThreadStillCapsTheTokenizerWork(t *testing.T) {
	m := admissionModel{fixedUs: 40_000, perTokenN: 10_000} // 10 ms for a 1000-token prompt
	for _, c := range []struct {
		workers int
		want    int64
	}{{1, 40_000 + 50*10_000}, {4, 40_000 + 13*10_000}} {
		e := admissionEngine(t, m, c.workers)
		if got := lastQueued(e, 50, 1000); got != c.want {
			t.Errorf("workers=%d: last of 50 queued at %d us, want %d", c.workers, got, c.want)
		}
	}
}

// The remaining cap scales with the front end's width as vLLM's does: each render thread
// (renderer_num_workers) and each API server (api_server_count, its own pool) adds one. The
// span to render 64 simultaneous 10 ms tokenizations halves with two threads and with two
// servers.
func TestRenderCapScalesWithRendererNumWorkersAndAPIServerCount(t *testing.T) {
	m := admissionModel{fixedUs: 50_000, perTokenN: 10_000}
	span := func(f engine.FrontendConfig) int64 {
		cfg := engine.SimConfig{
			KVCacheConfig: engine.NewKVCacheConfig(64, 16, 0, 0, 0, 0),
			BatchConfig:   engine.NewBatchConfig(16, 2048, 0),
			Frontend:      f,
		}
		e, err := engine.NewEngineCore(cfg, engine.MustNewKVStoreFromConfig(cfg.KVCacheConfig), m)
		if err != nil {
			t.Fatal(err)
		}
		return lastQueued(e, 64, 1000) - m.fixedUs
	}
	one := span(engine.FrontendConfig{RendererNumWorkers: 1})
	for _, f := range []engine.FrontendConfig{
		{RendererNumWorkers: 2},
		{RendererNumWorkers: 1, APIServerCount: 2},
	} {
		if got := span(f); got*2 != one {
			t.Errorf("%+v: 64 renders take %d us, want half of one thread's %d", f, got, one)
		}
	}
}
