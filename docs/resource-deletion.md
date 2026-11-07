# Administrative maintenance and deletion

## Native commands

The operator enables this with `administration: true` in the deployment file. Each
step is explicit:

```sh
ml-exp admin plan --submission SUBMISSION_ID --request-id 32_HEX_REQUEST_ID --kind outputs
ml-exp admin show MAINTENANCE_ID
ml-exp admin execute MAINTENANCE_ID --confirmation 'CONFIRMATION_RETURNED_BY_THE_API'
ml-exp admin retire --project P --run R --confirmation 'DELETE P/R'
```

`admin plan` and `admin show` return the confirmation for `admin execute`. `admin retire`
has no earlier call, so its confirmation is the fixed text `DELETE P/R`.

`admin plan` prepares removal of one terminal Attempt's `outputs`, persistent `state`
or one `checkpoint` (with `--checkpoint`). The checkpoint is selected by its manifest
and not hashed first, so a corrupt generation can be removed. A link, FIFO, socket or
device inside the target is removed itself and never followed, so what a link points
at stays. A target that is itself a link is refused. Datasets and source trees are out
of scope. Real provider cleanup remains pending acceptance; see the
[acceptance status](acceptance-status.md).

On PAI each plan runs as its own CPU job, and running `admin execute`
again observes that job without creating another. Execute one plan at a time on an
Attempt: a job that finds another holding the Attempt's lock fails, and a new request
ID then makes a new plan.

## Delete bytes first, then retire

`admin retire` retires a terminal Run registration and keeps small audit records. It
removes no bytes and answers `status: RETIRED` with `payloads_removed: false`. Once a
Run is retired `admin plan` is refused, so remove what you want gone with `admin plan`
and `admin execute` first. Retire does not check that the bytes are gone, and it does
not check whether another Run's `resume_from` names one of its checkpoints; `admin plan`
does check that.

The native client has no `run delete` command.
