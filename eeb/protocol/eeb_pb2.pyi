from google.protobuf.internal import containers as _containers
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Iterable as _Iterable, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class Empty(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class Session(_message.Message):
    __slots__ = ("session_id",)
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    def __init__(self, session_id: _Optional[str] = ...) -> None: ...

class Image(_message.Message):
    __slots__ = ("camera", "height", "width", "channels", "dtype", "data")
    CAMERA_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    DTYPE_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    camera: str
    height: int
    width: int
    channels: int
    dtype: str
    data: bytes
    def __init__(self, camera: _Optional[str] = ..., height: _Optional[int] = ..., width: _Optional[int] = ..., channels: _Optional[int] = ..., dtype: _Optional[str] = ..., data: _Optional[bytes] = ...) -> None: ...

class Caps(_message.Message):
    __slots__ = ("track", "horizon", "dt_ms", "modalities", "precision", "model_id", "server_impl", "action_dim", "proprio_dim", "cameras", "image_height", "image_width", "execute_horizon")
    TRACK_FIELD_NUMBER: _ClassVar[int]
    HORIZON_FIELD_NUMBER: _ClassVar[int]
    DT_MS_FIELD_NUMBER: _ClassVar[int]
    MODALITIES_FIELD_NUMBER: _ClassVar[int]
    PRECISION_FIELD_NUMBER: _ClassVar[int]
    MODEL_ID_FIELD_NUMBER: _ClassVar[int]
    SERVER_IMPL_FIELD_NUMBER: _ClassVar[int]
    ACTION_DIM_FIELD_NUMBER: _ClassVar[int]
    PROPRIO_DIM_FIELD_NUMBER: _ClassVar[int]
    CAMERAS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_HEIGHT_FIELD_NUMBER: _ClassVar[int]
    IMAGE_WIDTH_FIELD_NUMBER: _ClassVar[int]
    EXECUTE_HORIZON_FIELD_NUMBER: _ClassVar[int]
    track: str
    horizon: int
    dt_ms: float
    modalities: _containers.RepeatedScalarFieldContainer[str]
    precision: str
    model_id: str
    server_impl: str
    action_dim: int
    proprio_dim: int
    cameras: _containers.RepeatedScalarFieldContainer[str]
    image_height: int
    image_width: int
    execute_horizon: int
    def __init__(self, track: _Optional[str] = ..., horizon: _Optional[int] = ..., dt_ms: _Optional[float] = ..., modalities: _Optional[_Iterable[str]] = ..., precision: _Optional[str] = ..., model_id: _Optional[str] = ..., server_impl: _Optional[str] = ..., action_dim: _Optional[int] = ..., proprio_dim: _Optional[int] = ..., cameras: _Optional[_Iterable[str]] = ..., image_height: _Optional[int] = ..., image_width: _Optional[int] = ..., execute_horizon: _Optional[int] = ...) -> None: ...

class PredictRequest(_message.Message):
    __slots__ = ("session_id", "step", "images", "proprio_f32", "instruction", "t_send")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    STEP_FIELD_NUMBER: _ClassVar[int]
    IMAGES_FIELD_NUMBER: _ClassVar[int]
    PROPRIO_F32_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    T_SEND_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    step: int
    images: _containers.RepeatedCompositeFieldContainer[Image]
    proprio_f32: bytes
    instruction: str
    t_send: float
    def __init__(self, session_id: _Optional[str] = ..., step: _Optional[int] = ..., images: _Optional[_Iterable[_Union[Image, _Mapping]]] = ..., proprio_f32: _Optional[bytes] = ..., instruction: _Optional[str] = ..., t_send: _Optional[float] = ...) -> None: ...

class PredictResponse(_message.Message):
    __slots__ = ("actions_f32", "horizon", "dim", "t_recv", "t_done")
    ACTIONS_F32_FIELD_NUMBER: _ClassVar[int]
    HORIZON_FIELD_NUMBER: _ClassVar[int]
    DIM_FIELD_NUMBER: _ClassVar[int]
    T_RECV_FIELD_NUMBER: _ClassVar[int]
    T_DONE_FIELD_NUMBER: _ClassVar[int]
    actions_f32: bytes
    horizon: int
    dim: int
    t_recv: float
    t_done: float
    def __init__(self, actions_f32: _Optional[bytes] = ..., horizon: _Optional[int] = ..., dim: _Optional[int] = ..., t_recv: _Optional[float] = ..., t_done: _Optional[float] = ...) -> None: ...

class RolloutRequest(_message.Message):
    __slots__ = ("session_id", "context_frames", "proprio_f32", "instruction", "actions_f32", "horizon", "dim", "num_frames", "t_send")
    SESSION_ID_FIELD_NUMBER: _ClassVar[int]
    CONTEXT_FRAMES_FIELD_NUMBER: _ClassVar[int]
    PROPRIO_F32_FIELD_NUMBER: _ClassVar[int]
    INSTRUCTION_FIELD_NUMBER: _ClassVar[int]
    ACTIONS_F32_FIELD_NUMBER: _ClassVar[int]
    HORIZON_FIELD_NUMBER: _ClassVar[int]
    DIM_FIELD_NUMBER: _ClassVar[int]
    NUM_FRAMES_FIELD_NUMBER: _ClassVar[int]
    T_SEND_FIELD_NUMBER: _ClassVar[int]
    session_id: str
    context_frames: _containers.RepeatedCompositeFieldContainer[Image]
    proprio_f32: bytes
    instruction: str
    actions_f32: bytes
    horizon: int
    dim: int
    num_frames: int
    t_send: float
    def __init__(self, session_id: _Optional[str] = ..., context_frames: _Optional[_Iterable[_Union[Image, _Mapping]]] = ..., proprio_f32: _Optional[bytes] = ..., instruction: _Optional[str] = ..., actions_f32: _Optional[bytes] = ..., horizon: _Optional[int] = ..., dim: _Optional[int] = ..., num_frames: _Optional[int] = ..., t_send: _Optional[float] = ...) -> None: ...

class Frame(_message.Message):
    __slots__ = ("index", "image", "t_done")
    INDEX_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    T_DONE_FIELD_NUMBER: _ClassVar[int]
    index: int
    image: Image
    t_done: float
    def __init__(self, index: _Optional[int] = ..., image: _Optional[_Union[Image, _Mapping]] = ..., t_done: _Optional[float] = ...) -> None: ...
