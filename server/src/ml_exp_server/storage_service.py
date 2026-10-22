"""One local storage policy, built from the existing durable stores.

This facade owns no extra journal or retention state. The artifact receipt still
authorizes ACK release, UploadStore still owns staging TTL, and StorageCapacity
still arbitrates concurrent disk promises. NAS and datapool are outside its scope.
"""
from __future__ import annotations

from functools import cached_property

from .data.multipart_upload import UploadStore
from .results.artifact_lifecycle import ArtifactLifecycle


class StorageService:
    def __init__(self, objects, *, index=None):
        self.objects, self.index = objects, index

    @property
    def capacity(self):
        return self.objects.capacity

    @cached_property
    def uploads(self):
        return UploadStore(self.objects.root.parent, self.objects.config, capacity=self.capacity)

    @cached_property
    def lifecycle(self):
        return ArtifactLifecycle(self.objects)

    def maintain(self):
        # Expired upload bytes free local capacity first. ACK release retains
        # all receipt/identity guards and retries cache cleanup independently.
        self.uploads.prune()
        released = self.lifecycle.run_due()
        from .local_retention import LocalRetention
        LocalRetention(self.objects, self.index).maintain()
        return released
