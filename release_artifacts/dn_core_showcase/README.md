# Event-guided DN refinement: privacy-safe core excerpt

This archive is a compact implementation-oriented illustration of the method.
It includes a generic data loader, event voxelization, neural modules, loss
assembly, and an optimization step, while omitting identifying metadata.

## Included ideas

1. **Signed temporal event representation**: retain temporal bins, separate
   polarities, accumulated event mass, and optional temporal decay.
2. **Event differential-normal prediction**: predict pixelwise
   `(dN/dx, dN/dy)` instead of an absolute normal map.
3. **Detail-balanced supervision**: separately normalize strong derivative
   pixels so flat regions cannot make the zero solution optimal.
4. **Event-conditioned HDR token residual**: use selected event features to
   complement an LDR geometry token through an additive residual.
5. **Pixel geometry refiner**: predict a bounded high-frequency log-depth
   residual around a stable coarse/HDR-like depth.
6. **Depth-normal coupling**: require the normal derivative induced by final
   depth to agree with the ground-truth differential geometry.
7. **Cross-view DN consistency**: use detached coarse depth, intrinsics, and
   pose to map adjacent-view patches; compare calibrated derivative magnitude
   only in reliable overlapping patches.



## Files

- `data_pipeline.py`: sequence discovery, adjacent-view sampling, synchronized
  RGB/depth/event loading, and linear polarity voxelization;
- `network_modules.py`: event encoder, relevance field, HDR adapter,
  differential-normal head, and dense depth refiner;
- `core_showcase.py`: geometry operators and core losses;
- `training_step.py`: explicit loss assembly and optimization step;
- `esim_event_generator.py`: anonymized event generation source.
