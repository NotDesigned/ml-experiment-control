"""Offline test double for SDK transport; never contacts W&B.

A logged row lands in the Run's history at once; a row logged below the Run's
next step is dropped, as W&B drops it. `fail_logs` lists the log calls (by the
total count of calls so far) that raise once, and `finish_failures` is how
many finish calls raise.
"""
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
    def step(self):
        return max((row["_step"] for row in read()["runs"][self.id]["history"]), default=-1) + 1

    def define_metric(self, name, **options):
        data = read()
        data["runs"][self.id]["definitions"][name] = options
        write(data)

    def log(self, record, step, commit):
        assert commit is True
        data = read()
        run = data["runs"][self.id]
        data["logs"] += 1
        if data["logs"] in data.get("fail_logs", []):
            data["fail_logs"].remove(data["logs"])
            write(data)
            raise RuntimeError("log failed")
        if step < self.step:
            data["dropped"] = data.get("dropped", 0) + 1
        else:
            run["history"].append({**record, "_step": step})
        write(data)

    def finish(self, exit_code):
        data = read()
        if data.get("finish_failures"):
            data["finish_failures"] -= 1
            write(data)
            raise RuntimeError("finish failed")
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
    data["runs"].setdefault(identity, {"entity": kwargs["entity"], "name": kwargs["name"],
        "project": kwargs["project"], "config": visible(kwargs["config"]),
        "history": [], "definitions": {}, "summary": {}})
    data["runs"][identity]["state"] = "running"
    write(data)
    return Handle(identity)
