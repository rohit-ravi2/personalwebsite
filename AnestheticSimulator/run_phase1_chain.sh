#!/usr/bin/env bash
# WF2 Phase-1 heavy-run chain (close-out). Deflation-risk-first; full-core ensembles serial.
# P17 (streaming) || P4 (small sim), then P20, then P13-SOL28. Continues on per-block failure.
set -u
cd /mnt/ssd4tb/Desktop/website/personalwebsite/AnestheticSimulator
PY=/home/rohit/miniconda3/envs/ml/bin/python
export PYTHONPATH=src
NOTIFY=/home/rohit/bin/notify
log(){ echo "[$(date +%H:%M:%S)] $*"; }

$NOTIFY "AnesthSim Phase-1 chain START: P17||P4 -> P20 -> SOL28 (~7hr)" "AnesthSim chain" 2>/dev/null || true

# P17 model-valley (quick prerequisite) then P17 heavy (streaming) in background
log "P17 --model-valley"
$PY src/state_validation/p17_readout_validity.py --model-valley > /tmp/p17.log 2>&1 || log "P17 model-valley FAILED"
log "P17 --heavy (background, streaming)"
( $PY src/state_validation/p17_readout_validity.py --heavy >> /tmp/p17.log 2>&1; \
  $NOTIFY "AnesthSim P17 readout-validity DONE" "AnesthSim chain" 2>/dev/null || true ) &
P17PID=$!

# P4 concurrent with P17 streaming (small 70-sim job)
log "P4 falsifier worm"
$PY src/state_validation/p4_gate4_entailment.py falsifier worm > /tmp/p4.log 2>&1 || log "P4 FAILED"
$NOTIFY "AnesthSim P4 SNARE-falsifier DONE" "AnesthSim chain" 2>/dev/null || true

wait $P17PID 2>/dev/null || true

# P20 full-core ensemble (serial)
log "P20 battery"
$PY src/state_validation/p20_two_block_reachability.py battery > /tmp/p20.log 2>&1 || log "P20 FAILED"
$NOTIFY "AnesthSim P20 reachability DONE" "AnesthSim chain" 2>/dev/null || true

# P13-SOL28 full-core ensemble (serial, last)
log "P13-SOL28 heavy"
$PY src/state_validation/p13_sol28_nca_interval.py --heavy --nca-pa 75.0 > /tmp/sol28.log 2>&1 || log "SOL28 FAILED"
$NOTIFY "AnesthSim SOL28 nca-interval DONE" "AnesthSim chain" 2>/dev/null || true

$NOTIFY "AnesthSim Phase-1 chain COMPLETE (P17/P4/P20/SOL28). Collect verdicts." "AnesthSim chain" urgent 2>/dev/null || true
log "CHAIN COMPLETE"
