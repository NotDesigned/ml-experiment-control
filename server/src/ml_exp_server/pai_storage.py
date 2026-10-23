"""Approved PAI personal storage and role policy shared by DSW callers."""
import re


class PaiStorage:
    def __init__(self, config, api):
        self.config, self.api = config, api

    def datasets(self):
        """Approved account workspace and public RO mounts; never mount a root."""
        datasets = []
        if any(key in self.config for key in ("cpfs_uri", "user_vpc")):
            uri, user, vpc = (self.config.get(key) for key in ("cpfs_uri", "workspace_user", "user_vpc"))
            host = r"cpfs-[a-z0-9-]+\." + re.escape(self.api.config["region"]) + r"\.cpfs\.aliyuncs\.com/"
            if (not isinstance(uri, str) or not re.fullmatch("bmcpfs://" + host, uri)
                    or not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", user)
                    or "-vpc-" in uri and vpc is None
                    or "user_vpc" in self.config and (
                        not isinstance(vpc, dict) or set(vpc) != {"VpcId", "VSwitchId", "SecurityGroupId"}
                        or any(not isinstance(vpc[key], str) or not re.fullmatch(prefix + r"[a-z0-9]+", vpc[key])
                               for key, prefix in (("VpcId", "vpc-"), ("VSwitchId", "vsw-"), ("SecurityGroupId", "sg-"))))):
                raise ValueError("CPFS workspace requires an explicit same-region endpoint, approved user and matching user_vpc")
            datasets = [{"Uri": uri + path, "MountPath": mount, "MountAccess": access}
                        for path, mount, access in (("users/" + user + "/", "/mnt/ml-expd-personal", "RW"),
                                                    ("public/models/", "/mnt/models", "RO"),
                                                    ("public/datasets/", "/mnt/datasets", "RO"))]
        uri = self.config.get("oss_uri")
        mount = self.config.get("oss_mount_path", "/mnt/ml-expd-oss")
        if "oss_uri" not in self.config and "oss_mount_path" not in self.config:
            return datasets
        if not isinstance(uri, str) or not isinstance(mount, str):
            raise ValueError("DSW OSS mount requires a URI and a canonical mount path")
        host = r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]\.oss-" + re.escape(self.api.config["region"]) + r"(?:-internal)?\.aliyuncs\.com"
        match = re.fullmatch(r"oss://" + host + r"(/[A-Za-z0-9._/-]+/)", uri)
        if (not match or any(p in {"", ".", ".."} for p in match[1].split("/")[1:-1])
                or not re.fullmatch(r"/(?:mnt|data)/[A-Za-z0-9][A-Za-z0-9._/-]*", mount)
                or any(p in {"", ".", ".."} for p in mount.split("/")[1:])
                or datasets and (mount == "/mnt/workspace" or mount.startswith("/mnt/workspace/"))
                or any(mount == d["MountPath"] or mount.startswith(d["MountPath"] + "/")
                       or d["MountPath"].startswith(mount + "/") for d in datasets)):
            raise ValueError("DSW OSS mount must bind an explicit same-region prefix to /mnt or /data")
        role = self.role()
        if role is None:
            return datasets + [{"Uri": uri, "MountPath": mount, "MountAccess": "RW"}]
        user = self.config["workspace_user"]
        if match[1] != "/users/" + user + "/" or mount != "/mnt/oss":
            raise ValueError("personal OSS mount must use the approved user's prefix at /mnt/oss")
        root = uri[:-len(match[1])] + "/"
        return datasets + [{"Uri": root + path, "MountPath": target, "MountAccess": access, "RoleArn": role[0]}
                           for path, target, access in (("users/" + user + "/", "/mnt/oss", "RW"),
                                                       ("public/models/", "/mnt/oss-models", "RO"),
                                                       ("public/datasets/", "/mnt/oss-datasets", "RO"))]

    def role(self):
        arn = self.config.get("data_role_arn")
        if "data_role_arn" not in self.config:
            return None
        user = self.config.get("workspace_user")
        if not isinstance(user, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}", user):
            raise ValueError("personal data role requires the approved workspace user")
        match = re.fullmatch(r"acs:ram::([0-9]{1,32}):role/pai-data-" + re.escape(user), arn) if isinstance(arn, str) else None
        if not match:
            raise ValueError("data role must be the approved user's pai-data role")
        return arn, match[1]
