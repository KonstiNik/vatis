# mem_profile — allenai/Olmo-3-7B-Think-SFT

- device: NVIDIA H100 (99.9 GB)
- n_params: 7,298,011,136; weights 14.6 GB; flat_grad fp32 29.2 GB
- micro_batch_size=1, n_hutchinson=8

| B | S | delta_loss peak | grad dtype | grad tuple | hutch peak | pseqcv peak |
|---|---|---|---|---|---|---|
| 2 | 64 | 74.7 GB | torch.bfloat16 | 14.6 GB | 43.9 GB | 59.2 GB |
| 2 | 256 | 74.7 GB | torch.bfloat16 | 14.6 GB | 44.0 GB | 61.4 GB |
| 2 | 512 | 74.8 GB | torch.bfloat16 | 14.6 GB | 44.2 GB | 64.4 GB |
| 10 | 64 | 74.7 GB | torch.bfloat16 | 14.6 GB | 43.9 GB | 59.2 GB |
| 10 | 256 | 74.7 GB | torch.bfloat16 | 14.6 GB | 44.0 GB | 61.4 GB |
| 10 | 512 | 74.8 GB | torch.bfloat16 | 14.6 GB | 44.2 GB | 64.4 GB |
