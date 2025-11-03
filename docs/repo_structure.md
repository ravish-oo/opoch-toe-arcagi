arc-bit-algebra/
├─ README.md
├─ pyproject.toml
├─ docs/
│  ├─ anchors              # math + addendum (frozen) + pure CS computing spec      
│  ├─ BYTES_AND_HASHING.md           # BE row-major format; section receipts schema
│  └─ PARAM_REGISTRY.md              # frozen orders, priorities, constants
├─ fixtures/
│  ├─ arc/                           # real ARC tasks for dev runs
│  └─ tiny/                          # minimal kernels: pose/period/shift checks
├─ receipts/                         # deterministic outputs (no timestamps)
│  └─ runs/<run_id>/sections.jsonl   # sealed section receipts
│                     output.json    # final grid + summary hashes
├─ src/arcbit/
│  ├─ __init__.py
│  ├─ core/                          # foundation (spec-level)
│  │  ├─ __init__.py
│  │  ├─ registry.py                 # param_registry(), frozen orders, spec_version
│  │  ├─ bytesio.py                  # serialize_grid_be_row_major(), serialize_planes_...
│  │  ├─ hashing.py                  # blake3_hash(); no RNG/time
│  │  └─ receipts.py                 # Receipts class; double-run checker
│  ├─ kernel/                        # Bit-Kernel v2 (5 ops) + frames
│  │  ├─ __init__.py
│  │  ├─ planes.py                   # PACK/UNPACK, color ordering, row-masks
│  │  ├─ ops.py                      # SHIFT, POSE(8 D4), AND/OR/ANDN
│  │  ├─ period.py                   # minimal 1D period; 2D residues
│  │  └─ frames.py                   # D4 lex-min pose; top-left anchor
│  ├─ normalization/                 # common canvas and exact reduction
│  │  ├─ __init__.py
│  │  ├─ lcm_canvas.py               # LCM lift for training outputs
│  │  └─ downscale_strict.py         # constant-block reduction only
│  ├─ evidence/                      # pure measurements (no painting)
│  │  ├─ __init__.py
│  │  ├─ components.py               # 4-CC over planes; bbox/area/perim/outline hash
│  │  └─ features.py                 # T9 frozen feature vector (counts, periods, etc.)
│  ├─ emitters/                      # (A,S) producers; kernel-only ops
│  │  ├─ __init__.py
│  │  ├─ output_transport.py         # transport Y' to LCM canvas (hard admits)
│  │  ├─ unanimity.py                # singleton votes on shared canvas
│  │  ├─ witness_learn.py            # rigid pieces + σ (exact equality)
│  │  ├─ witness_emit.py             # conjugate + forward admits from X*
│  │  ├─ lattice.py                  # exact periodic residues admits
│  │  ├─ icl_conv.py                 # stencil admits (optional)
│  │  ├─ icl_kron.py                 # block replication admits (optional)
│  │  ├─ morph.py                    # reachability admits (N=4; optional)
│  │  ├─ logic_guard.py              # local predicates; truth rows (optional)
│  │  └─ param_icl.py                # input-param kernels (optional)
│  ├─ constraints/                   # negative info
│  │  ├─ __init__.py
│  │  ├─ forbids.py                  # learn M(c,d) (directed/symmetric)
│  │  └─ ac3.py                      # frozen 4-nbr queue; arc consistency
│  ├─ propagation/                   # one monotone loop
│  │  ├─ __init__.py
│  │  └─ lfp.py                      # admit ∧ then AC-3 to fixed point
│  ├─ selection/                     # who speaks and pick rule
│  │  ├─ __init__.py
│  │  ├─ engine_winner.py            # choose global EngineWinner (train-scope)
│  │  └─ selector.py                 # witness → engine_winner → unanimity → bottom
│  ├─ sizing/                        # T9 size (pure mapping)
│  │  ├─ __init__.py
│  │  └─ aggmap_size.py              # hypothesis search; predict (R_out,C_out)
│  └─ runner/
│     ├─ __init__.py
│     └─ solve.py                    # end-to-end: build C; frames; LCM; emitters; LFP; select; downscale; receipts
├─ scripts/
│  ├─ solve_task.py                  # python -m scripts.solve_task fixtures/arc/00576224.json
│  ├─ sweep.py                       # batch real ARC tests → receipts/runs/<run_id>
│  └─ compare_runs.py                # verify double-run section hashes identical
└─ tools/
   └─ blake3/                        # vendored deterministic BLAKE3
