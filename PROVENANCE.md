# Provenance: public code vs private research archive

This repository is a clean export of the private research repository at commit `1369363`, published with a fresh history so that unpublished third-party material in earlier private commits never becomes public.

- Commit ids quoted in reports and protocols (for example the code version of the one-time evaluations) refer to the **private archive**, which is preserved unchanged. They do not exist in this public history.
- The frozen models are verified by **content hashes**, which do not depend on commit ids. `src/v2/forward_eval.py` checks them:

  - `models/frozen_phase5.json`: `16fe577b12eb96d3`
  - `models/frozen_v2.json`: `42d6a5a803741412`
  - `models/v2_config.json`: `e785ce642bce11f3`

- Removed from this export: a Valuebetennis audit report and its script (licence unresolved) and assistant working notes. Some passages were shortened where they quoted that material.
