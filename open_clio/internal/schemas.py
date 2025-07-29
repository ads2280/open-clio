import uuid
from typing_extensions import TypedDict, NotRequired


class Summary(TypedDict):
    example_id: str | uuid.UUID
    summary: str
    partition: NotRequired[str]


class ClusterInfo(TypedDict):
    name: str
    description: str
    size: int
    summaries: list[str]
    example_ids: list[str | uuid.UUID]
    category: str
    id: uuid.UUID
