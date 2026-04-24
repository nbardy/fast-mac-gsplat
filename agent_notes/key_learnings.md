# Key Learnings

- v7.2/v7.3 hardware raster training is dominated by CPU aux/state movement, not
  just the front-K capture algorithm. Routing gradient calls through the v5
  compute training path moved the 4K/64K uniform F+B probe from 438.782 ms
  (`v72_tiled_k2`) to 76.464 ms (`v73_hybrid_k2`), within 3.8% of direct v5.
- Skipping front-K capture for hardware eval helps but does not make hardware
  raster the 4K forward winner yet: 4K/64K forward improved from 183.679 ms
  (`v72_tiled_k2`) to 133.129 ms (`v73_hybrid_k2`), but v5 stayed at 11.781 ms.
  The remaining eval blocker is the Metal texture/output readback into torch MPS.
