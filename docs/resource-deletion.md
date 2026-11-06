# Administrative maintenance and deletion

## Native commands

The operator enables this with `administration: true` in the deployment file. Each
step is explicit:

```sh
ml-exp admin plan --submission SUBMISSION_ID --request-id 32_HEX_REQUEST_ID --kind outputs
ml-exp admin show MAINTENANCE_ID
ml-exp admin execute MAINTENANCE_ID --confirmation 'CONFIRMATION_RETURNED_BY_THE_API'
ml-exp admin retire --project P --run R --confirmation 'CONFIRMATION_RETURNED_BY_THE_API'
```

`admin plan` prepares removal of one terminal Attempt's `outputs`, persistent `state`
or one `checkpoint` (with `--checkpoint`). The checkpoint is selected by its manifest
and not hashed first, so a corrupt generation can be removed. A link, FIFO, socket or
device inside the target is removed itself and never followed, so what a link points
at stays. A target that is itself a link is refused. Datasets and source trees are out
of scope. `admin retire` retires a terminal Run registration and keeps small audit
records. Real provider cleanup remains pending acceptance; see the
[acceptance status](acceptance-status.md).

The native client has no `run delete` command.
