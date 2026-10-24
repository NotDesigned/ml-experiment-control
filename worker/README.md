# ML-Expd worker

Independent, standard-library-only programs executed inside training, data-copy,
result-recovery and image-build jobs. The package has no runtime dependencies
and imports neither the ML-Expd server nor its scheduler library.

The server uses `ml_exp_worker.source_path(name)` to copy the reviewed programs
into sealed image contexts or remote scripts. Container paths, capabilities and
the bytes used for worker identities remain unchanged by this package split.
Workers run with exact-job credentials provided by the server; this package is
not a second scheduler or an independently configured daemon.

Training code writes metrics with the standalone `ml_exp_metrics.py` generated
by the client. Workers read and transmit those records; metric validation and
W&B publication belong to the server.
