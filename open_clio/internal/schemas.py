import uuid
from typing_extensions import TypedDict, NotRequired


class ExampleSummary(TypedDict):
    example_id: str | uuid.UUID
    summary: str
    category: NotRequired[str]


class ClusterInfo(TypedDict):
    name: str
    description: str
    size: int
    summaries: list[str]
    example_ids: list[str | uuid.UUID]
    category: str
    id: uuid.UUID
