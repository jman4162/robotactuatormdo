"""Robot task/trajectory descriptions feeding joint duty cycles."""

from __future__ import annotations


def joint_duty_from_task(task):
    raise NotImplementedError("planned: robot task -> joint duty mapping")
