FasterWAM RoboTwin native components; text mode: per_request. All times in ms.

| Component | 10 steps | 1 steps |
| --- | ---: | ---: |
| vae_encode | 5.321 | 5.281 |
| text_encode | 16.590 | 16.519 |
| video_prepare | 0.610 | 0.598 |
| video_dit_prefill | 26.029 | 26.074 |
| action_dit_denoise | 170.305 | 17.123 |
| action_scheduler | 0.215 | 0.022 |
| overlap_correction | 0.000 | 0.000 |
| other | 6.527 | 6.034 |
| total | 225.598 | 71.650 |
| total_uninstrumented | 226.029 | 71.801 |

Components are CUDA intervals in the instrumented native request. other is wall time minus interval union, including unmarked work and synchronization. Use total_uninstrumented to compare latency; all totals include VAE, input transfers, noise and CPU output. Video prefill includes future-frame KV fusion. Action denoise includes pre_dit, all action blocks and post_dit. Dataset decode, mosaic/state normalization, model load and validation are excluded.

| Steps | Mean | p50 | p90 | Min | Max |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 10 | 226.029 | 225.883 | 227.551 | 222.989 | 231.107 |
| 1 | 71.801 | 71.431 | 72.627 | 70.484 | 81.057 |
