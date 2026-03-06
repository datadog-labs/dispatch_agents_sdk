from agentservice.v1 import message_pb2 as _message_pb2
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class InvokeRequest(_message.Message):
    __slots__ = ("function_name", "payload", "uid", "trace_id", "ts", "topic", "message_type")
    FUNCTION_NAME_FIELD_NUMBER: _ClassVar[int]
    PAYLOAD_FIELD_NUMBER: _ClassVar[int]
    UID_FIELD_NUMBER: _ClassVar[int]
    TRACE_ID_FIELD_NUMBER: _ClassVar[int]
    TS_FIELD_NUMBER: _ClassVar[int]
    TOPIC_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_TYPE_FIELD_NUMBER: _ClassVar[int]
    function_name: str
    payload: _message_pb2.Payload
    uid: str
    trace_id: str
    ts: str
    topic: str
    message_type: str
    def __init__(self, function_name: _Optional[str] = ..., payload: _Optional[_Union[_message_pb2.Payload, _Mapping]] = ..., uid: _Optional[str] = ..., trace_id: _Optional[str] = ..., ts: _Optional[str] = ..., topic: _Optional[str] = ..., message_type: _Optional[str] = ...) -> None: ...

class InvokeResponse(_message.Message):
    __slots__ = ("result", "is_error")
    RESULT_FIELD_NUMBER: _ClassVar[int]
    IS_ERROR_FIELD_NUMBER: _ClassVar[int]
    result: _message_pb2.Payload
    is_error: bool
    def __init__(self, result: _Optional[_Union[_message_pb2.Payload, _Mapping]] = ..., is_error: _Optional[bool] = ...) -> None: ...

class HealthCheckRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class HealthCheckResponse(_message.Message):
    __slots__ = ("status",)
    class ServingStatus(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
        __slots__ = ()
        SERVING_STATUS_UNSPECIFIED: _ClassVar[HealthCheckResponse.ServingStatus]
        SERVING_STATUS_SERVING: _ClassVar[HealthCheckResponse.ServingStatus]
        SERVING_STATUS_NOT_SERVING: _ClassVar[HealthCheckResponse.ServingStatus]
        SERVING_STATUS_SERVICE_UNKNOWN: _ClassVar[HealthCheckResponse.ServingStatus]
    SERVING_STATUS_UNSPECIFIED: HealthCheckResponse.ServingStatus
    SERVING_STATUS_SERVING: HealthCheckResponse.ServingStatus
    SERVING_STATUS_NOT_SERVING: HealthCheckResponse.ServingStatus
    SERVING_STATUS_SERVICE_UNKNOWN: HealthCheckResponse.ServingStatus
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: HealthCheckResponse.ServingStatus
    def __init__(self, status: _Optional[_Union[HealthCheckResponse.ServingStatus, str]] = ...) -> None: ...
