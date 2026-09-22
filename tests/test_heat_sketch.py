from __future__ import annotations

import math
import time

from bodocache.planner.heat_sketch import CountMinSketch, HeatSketch, SpaceSaving


def test_cms_add_query_exact_with_wide_table():
    cms = CountMinSketch(width=4096, depth=4, seed=1337)
    cms.add("a", 5)
    cms.add("b", 3)
    cms.add("a", 2)
    assert cms.query("a") == 7
    assert cms.query("b") == 3
    assert cms.query("unseen") == 0


def test_cms_query_is_upper_bound_under_collisions():
    # Tiny table forces collisions; CMS must never underestimate.
    cms = CountMinSketch(width=4, depth=2, seed=1)
    truth = {"x": 10, "y": 7, "z": 1}
    for k, c in truth.items():
        cms.add(k, c)
    for k, c in truth.items():
        assert cms.query(k) >= c


def test_cms_deterministic_across_instances():
    # Same seed + same inserts must give identical estimates (no salted hash()).
    def build():
        cms = CountMinSketch(width=256, depth=4, seed=99)
        for i in range(200):
            cms.add(f"key-{i % 37}", i + 1)
        return [cms.query(f"key-{i}") for i in range(37)]

    assert build() == build()


def test_space_saving_topk_bounded_and_exact_for_small_streams():
    ss = SpaceSaving(k=5)
    for i in range(5):
        ss.add(f"k{i}", 10 - i)
    top = dict((k, cnt) for k, cnt, _ in ss.topk())
    assert len(top) <= 5
    assert top["k0"] == 10
    assert top["k4"] == 6


def test_space_saving_replaces_min_when_full():
    ss = SpaceSaving(k=2)
    ss.add("a", 5)
    ss.add("b", 3)
    ss.add("c", 1)  # evicts b (min), inherits its count as error
    keys = {k for k, _, _ in ss.topk()}
    assert keys == {"a", "c"}


def test_heat_sketch_estimate_uses_min_of_cms_and_spacesaving():
    hs = HeatSketch(width=4096, depth=4, k=64)
    hs.add("hot", 12)
    assert hs.estimate("hot") == 12
    assert hs.estimate("cold") == 0


def test_heat_sketch_deterministic_across_instances():
    def build():
        hs = HeatSketch(width=512, depth=4, k=64, seed=7)
        for i in range(300):
            hs.add(f"page-{i % 50}", (i % 5) + 1)
        return {f"page-{i}": hs.estimate(f"page-{i}") for i in range(50)}

    assert build() == build()


def test_heat_sketch_decay_scales_spacesaving_counts():
    hs = HeatSketch(width=4096, depth=4, k=64, decay_lambda=0.01)
    hs.add("p", 100)
    # Pretend 100 seconds elapsed since the last decay.
    hs._last_decay_ts = time.time() - 100.0
    hs.decay()
    expected = int(100 * math.exp(-0.01 * 100.0))
    assert hs.ss.counters["p"][0] == expected
    # estimate() intersects CMS (undecayed) with SpaceSaving (decayed)
    assert hs.estimate("p") == expected


def test_heat_sketch_export_heat():
    hs = HeatSketch(width=4096, depth=4, k=64)
    hs.add("a", 4)
    hs.add("b", 9)
    exported = hs.export_heat()
    assert exported["a"] == 4
    assert exported["b"] == 9
