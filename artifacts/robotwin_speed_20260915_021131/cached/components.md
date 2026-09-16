FasterWAM RoboTwin native components; text mode: cached. All times in ms.

| Component | 10 steps | 1 steps |
| --- | ---: | ---: |
| vae_encode | 5.356 | 5.338 |
| text_encode | 0.000 | 0.000 |
| video_prepare | 0.571 | 0.561 |
| video_dit_prefill | 25.808 | 25.886 |
| action_dit_denoise | 169.625 | 16.991 |
| action_scheduler | 0.215 | 0.022 |
| overlap_correction | 0.000 | 0.000 |
| other | 6.325 | 5.961 |
| total | 207.901 | 54.758 |
| total_uninstrumented | 207.127 | 54.531 |

Components are CUDA intervals in the instrumented native request. other is wall time minus interval union, including unmarked work and synchronization. Use total_uninstrumented to compare latency; all totals include VAE, input transfers, noise and CPU output. Video prefill includes future-frame KV fusion. Action denoise includes pre_dit, all action blocks and post_dit. Dataset decode, mosaic/state normalization, model load and validation are excluded.

| Steps | Mean | p50 | p90 | Min | Max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 207.127 | 206.802 | 208.584 | 205.608 | 213.648 |
| 1 | 54.531 | 54.418 | 54.768 | 54.057 | 58.528 |
