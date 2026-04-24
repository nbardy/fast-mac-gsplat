# v7.1 bottleneck response

Yes — your diagnosis is right.

The earlier v7.1 code fixed the backward recurrence but accidentally built front-K with another `pixels × splats` pass:

- `capture_front_k` in `gsplat_v71.metal` scanned `for i in 0..G-1` for every pixel
- the docs talked about saved-state backward and tile-aware fallback, but they did **not** explicitly catch that the implemented capture path had reintroduced the same scaling wall on the forward side

So the honest answer is:

- **did the earlier docs fully discover this?** No. They were directionally right, but they missed the concrete capture-pass complexity regression.
- **did the earlier code fix it?** No.
- **does this new handoff fix that specific bottleneck?** Yes, by replacing full-scene pixel scans with tile-bin scans and by reusing the same bins for overflow replay.

The important new idea here is simple:

- hardware render still draws the final image
- front-K capture is built from **sparse tile bins**
- overflow replay also uses **sparse tile bins**
- saved aux state stays on CPU instead of bouncing CPU → MPS → CPU across forward/backward

That is the specific repair your timing breakdown was pointing at.
