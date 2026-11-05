"""Offline test double for SDK transport; never contacts W&B."""
import json
import os


def read():
    with open(os.environ["TEST_CLOUD"]) as stream:
        return json.load(stream)


def write(value):
    with open(os.environ["TEST_CLOUD"], "w") as stream:
        json.dump(value, stream)


class Settings:
    def __init__(self, **kwargs):
        pass


class Summary:
    def __init__(self, identity):
        self.identity = identity

    def __getitem__(self, key):
        return read()["runs"][self.identity]["summary"][key]

    def __setitem__(self, key, value):
        data = read()
        data["runs"][self.identity]["summary"][key] = value
        write(data)

    def __delitem__(self, key):
        data = read()
        del data["runs"][self.identity]["summary"][key]
        write(data)

    def get(self, key, default=None):
        return read()["runs"][self.identity]["summary"].get(key, default)

    def keys(self):
        return read()["runs"][self.identity]["summary"].keys()


class Handle:
    def __init__(self, identity):
        self.id = identity
        self.summary = Summary(identity)

    @property
    def entity(self):
        return read()["runs"][self.id]["entity"]

    @property
    def project(self):
        return read()["runs"][self.id]["project"]

    @property
    def config(self):
        return read()["runs"][self.id]["config"]

    @property
    def state(self):
        return read()["runs"][self.id]["state"]

    @property
    def lastHistoryStep(self):
        return max((row["_step"] for row in read()["runs"][self.id]["history"]), default=-1)

    @property
    def step(self):
        run = read()["runs"][self.id]
        return max((row["_step"] for row in run['history'] + run['pending']), default=-1) + 1

    def scan_history(self, min_step, keys=None, max_step=None, **kwargs):
        data = read()
        run = data["runs"][self.id]
        if not data["hidden"]:
            run["history"].extend(run["pending"])
            run["pending"] = []
            write(data)
        rows = [({key: row[key] for key in keys} if keys else row.copy())
                for row in run["history"] if min_step <= row["_step"]
                and (max_step is None or row["_step"] < max_step)]
        if data.get("repeat_pages"):
            rows += [row.copy() for row in rows]
        if data.get("conflicting_page") and rows:
            rows += [{**rows[0], "curves/loss": -9}]
        if data.get("duplicate_sequence") and rows:
            rows += [{**rows[0], "_step": self.lastHistoryStep + 1}]
        return rows

    def define_metric(self, name, **options):
        data = read()
        data["runs"][self.id]["definitions"][name] = options
        write(data)

    def log(self, record, step, commit):
        assert commit is True
        data = read()
        data["runs"][self.id]["pending"].append({**record, "_step": step})
        data["logs"] += 1
        write(data)

    def finish(self, exit_code):
        data = read()
        run = data["runs"][self.id]
        run["history"].extend(run["pending"])
        run["pending"] = []
        data["runs"][self.id]["state"] = "finished" if exit_code == 0 else "failed"
        data["finishes"] += 1
        write(data)


class Api:
    default_entity = "fixture"

    def __init__(self, **kwargs):
        pass

    def run(self, path):
        entity, project, identity = path.split("/")
        if identity not in read()["runs"]:
            raise CommError('Could not find run; sensitive diagnostic must not be transported')
        return Handle(identity)


class CommError(Exception):
    pass


def init(**kwargs):
    data = read()
    identity = kwargs["id"]
    if kwargs["resume"] == "must":
        assert identity in data["runs"]
    if kwargs["resume"] == "never":
        assert identity not in data["runs"]
    data["inits"] += 1
    def visible(value):
        if type(value) is float and value == 3.4976104655686546:
            return 3.497610465568654
        if type(value) is float and value.is_integer():
            return int(value)
        if isinstance(value, list):
            return [visible(item) for item in value]
        if not isinstance(value, dict):
            return value
        return {key: item for key, value in value.items()
                if (item := visible(value)) != {}}
    data["runs"].setdefault(identity, {"entity": kwargs["entity"],
        "project": kwargs["project"], "config": visible(kwargs["config"]),
        "history": [], "pending": [], "definitions": {}, "summary": {}})
    data["runs"][identity]["state"] = "running"
    write(data)
    return Handle(identity)
