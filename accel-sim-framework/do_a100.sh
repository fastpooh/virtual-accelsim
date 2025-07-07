#!/bin/bash 
rm -rf sim_run*
./gpu-simulator/bin/release/accel-sim.out   -trace /casl/accel-sim/accel-sim-framework/a100_traces/kernelslist.g   -config ./gpu-simulator/gpgpu-sim/configs/tested-cfgs/SM86_A100/gpgpusim.config   -config ./gpu-simulator/configs/tested-cfgs/SM86_A100/trace.config   -gpgpu_ptx_sim_mode 0   -trace_enabled 1   -gpgpu_ptx_instruction_classification 0 | tee .log.txt
cat .log.txt | grep "gpu_sim_cycle" 
