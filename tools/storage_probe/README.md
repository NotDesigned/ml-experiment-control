# Storage probe

Establishes what the shared filesystem really does, before relying on it. See the
questions at the top of `probe.py`. Run it once per backend storage, with zero GPUs and
two jobs at the same time, so that they land on different nodes:

```sh
ml-exp submit tools/storage_probe --executor pai-native-cpu --gpus 0 --time 30m --run probe-a -- \
  python3 probe.py --role a --session first --note "two words"
ml-exp submit tools/storage_probe --executor pai-native-cpu --gpus 0 --time 30m --run probe-b -- \
  python3 probe.py --role b --session first --note "two words"
```

**Needs PR #132 on the server.** The default directory is `DATA_DIR`, which sits inside the
dataset store. Before #132, creating it on a backend with no dataset blocks later deliveries.

Set the digest in `Dockerfile` first. Each job waits up to 30 minutes for the other, then
writes `storage-probe-a.json` or `-b.json` to its results. Read `findings` in both.

| Finding | Good | If not |
| --- | --- | --- |
| `nodes` | absent | The jobs shared a host, so nothing below about clients counts. Run again |
| `flock` | excludes across clients | Dataset recovery, the Attempt lock and other locks do not protect across nodes. Do not cut over until this is understood |
| `inode` | same | The dataset marker and the metrics cursor bind the inode, and would refuse |
| `device` | either | Expected to differ on some clients; the code no longer compares it |
| `mtime_ns`, `ctime_ns` | same | A stat check for registered directories must leave out what differs |
| `root` | either | As root, read-only modes stop nothing, and hashing is what catches changes |
| `stat_per_second` | recorded | Sets the file limit for registered directories |
| `race_*` | exactly one wins | That way of creating cannot replace a lock across nodes |

`command_chain` and `argv` show how the provider started the job. If a shell received one
joined string, `--note` reaches the job split or quoted differently, and a long
configuration on the command line needs to come from object storage instead.

It leaves `ml-exp-storage-probe` in the shared directory. Remove it by hand afterwards.
