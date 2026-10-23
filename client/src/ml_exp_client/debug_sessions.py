"""Debug machines use the same HTTP credentials, never cloud access keys."""
from .api import ClientError, segment


def debug_session(args, client, health):
    if "debug-sessions.v1" not in health.get("capabilities", []):
        raise ClientError("server lacks debug-sessions.v1")
    action = args.debug_action
    if action == "profiles":
        return client.call("/api/debug-session-profiles")
    if action == "list":
        return client.call("/api/debug-sessions")
    if action == "create":
        try:
            if args.ssh_key.stat().st_size > 16384:
                raise ValueError()
            key = args.ssh_key.read_text(encoding="ascii").strip()
            if (not key.startswith(("ssh-ed25519 ", "ssh-rsa ")) or "PRIVATE KEY" in key
                    or "\n" in key or "\r" in key):
                raise ValueError()
        except (OSError, ValueError, UnicodeError):
            raise ClientError("--ssh-key must be a single OpenSSH public key file (.pub)") from None
        return client.call("/api/debug-sessions", data={"session_id": args.id, "profile": args.profile,
                           "ssh_public_key": key, "duration_seconds": args.duration_seconds})
    path = "/api/debug-sessions/" + segment(args.id)
    if action == "status":
        return client.call(path)
    if action == "delete":
        return client.call(path, method="DELETE")
    return client.call(path + "/" + action, data={})
