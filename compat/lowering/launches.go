// Original to blis-vllm.

package lowering

import (
	"fmt"
	"strconv"

	"github.com/inference-sim/blis-substrate/contracts/engine/plan"
	"github.com/inference-sim/blis-substrate/contracts/spec/model"
	"github.com/inference-sim/blis-substrate/contracts/units"

	corelower "github.com/inference-sim/blis-vllm/core/lower"
	"github.com/inference-sim/blis-vllm/internal/upstream"
)

// museLaunches states, for a Muse Spark target on vLLM v0.33.0, how the engine's fused kernels
// change each layer's launches in a step of a given size (core/lower MuseSparkLaunchRanges),
// as the plan's Launches (inference-sim/blis-vllm#161). It states none for another model or
// release: the catalog graph's own count stands.
//
// The facts it cannot read from the deployment are named in the returned note rather than
// defaulted silently: the chip's NVLink multicast, which FlashInfer asks the device (a Blackwell
// part with NVLink is taken to have it), and whether the tensor-parallel group spans hosts (it
// does when it is wider than the packaging's GPUs per node).
func museLaunches(g *model.Graph, c *lowerContext, lay layout, pp int, expert bool, cacheRecord, release string,
	note func(name, value, reason string)) ([]plan.LaunchChange, error) {
	if release != upstream.Version || !corelower.IsMuseSpark(g) {
		return nil, nil
	}
	l := corelower.MuseSparkLayout{
		CapabilityMajor: c.family,
		TP:              lay.TP, PP: pp, DP: lay.DP, PCP: lay.PCP, DCP: lay.DCP,
		ExpertParallel: expert,
		KVCacheDType:   cacheRecord,
		TPSpansNodes:   lay.GPUsPerNode > 0 && lay.TP > lay.GPUsPerNode,
		Multicast:      c.family == 10 && c.chip.IntraNodeBwGBps > 0,
	}
	if c.family != 9 && c.family != 10 {
		note("launches", "graph count",
			"Muse Spark's fused kernels are resolved for compute capability 9 and 10; on capability "+
				strconv.Itoa(c.family)+" the graph's launch count stands (inference-sim/blis-vllm#191)")
		return nil, nil
	}
	ranges, err := corelower.MuseSparkLaunchRanges(g, l)
	if err != nil {
		return nil, fmt.Errorf("resolving Muse Spark's fused kernels: %w", err)
	}
	out := make([]plan.LaunchChange, 0, len(ranges))
	for _, r := range ranges {
		out = append(out, plan.LaunchChange{FromTokens: units.Tokens(r.From), ToTokens: units.Tokens(r.To), PerLayer: r.PerLayer})
	}
	if c.family == 10 && l.TP > 1 {
		note("muse_spark_multicast", strconv.FormatBool(l.Multicast),
			"FlashInfer's is_multicast_supported is a device query the catalog does not record; a "+
				"capability-10 part with NVLink is taken to support NVLink multicast, which the two "+
				"MNNVL fusions need (vllm/models/muse_spark/nvidia/ops/mnnvl_fusions.py:32-58 at "+
				upstream.Commit[:8]+")")
	}
	return out, nil
}
