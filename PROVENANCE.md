# Source provenance

The initial solver snapshot was extracted without switching or modifying the
active Numi Lab checkout.

- Source repository: `https://github.com/Numi2/numi-lab.git`
- Source branch: `numisolver`
- Source revision: `0e2bd8b24f13fa2c839eb21be9f521762545af73`
- Extraction date: `2026-08-15`

The copied source files and their SHA-256 digests at extraction were:

```text
17ee01457c73da786158619f8cf49977e0b58092b15af749b979e49b32c98073  src/metal/MetalWorldContact.metal
00a2ff82e0926f6e7789a30c6730225761b438bef06aab273982b65e4f1c07a0  include/metalrobo/constraint_ir_shared.h
540403b52fbcaefa472ea5f8dab291bb27ad9eadb702315e9d0355dcf75a1e76  include/metalrobo/engine_types.h
55777caa42851b167780255c76e02c40e2285ee43cc6dda77063d8639f6728d8  include/metalrobo/gpu_types.h
7aeecb8caff40d18be1e635622ed41a036d601d35cc4ea507d68521ed59bc21c  include/metalrobo/rod_gpu_shared.h
51040bc1ca54d2d8deec332d44c73a3ac19bbf157a362854afabd00f0f3f2673  include/metalrobo/unified_quality_shared.h
```

The articulated-response milestone later copied one additional owning
numerical source from the same branch and revision, again without switching
the active Numi Lab checkout:

```text
14f88429562abf3c53481283f1c403f8c0293dc7533779a0a701d420a2704983  src/metal/ArticulatedOperator.metal
```

The streamed inverse-ABA milestone copied the owning inverse-mass source and
its pointer-free schedule ABI from the active `coupled` checkout at revision
`d0be935070df92811a7d0a650df271a2181b04a9`. Those source paths were clean;
their last owning revisions were `b26fe576550bbf55c4a9840c448240631956d7ee`
and `50e047ac605bb7dbefab9ae8b6a904b7316cc2ca`, respectively.

```text
4833d1701012e516954191e86843d5c09a6e0a8be74605f06304361254169090  src/metal/ArticulatedInverseMass.metal
017a29c5a992df98541274975f7604a24e260d33b141aff30a51c0695e52b00d  include/metalrobo/parallel_aba_shared.h
```

The first independent build used Apple Metal toolchain `32023.883` and
produced a metallib with SHA-256
`d36b038304d6494ed73a34c01abe21513a850c217721853a4105a6317983c070`.
Metallib bytes are toolchain-dependent and are not treated as a stable source
fingerprint.

The hashes above identify the initial import, not the current working sources.
All independent changes after extraction are recorded in this repository's Git
history.
