"""v1 API; the public contract matches src.app without importing its backend."""

from contextlib import asynccontextmanager
from time import perf_counter_ns
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, StrictStr, field_validator

from src.onnx_backend import OnnxBackend


class ClassifyRequest(BaseModel):
    text: StrictStr = Field(min_length=1, max_length=2000)

    @field_validator('text')
    @classmethod
    def require_nonblank(cls, text):
        if not text.strip():
            raise ValueError('text must contain a non-whitespace character')
        return text


class Timings(BaseModel):
    tokenize: float = Field(ge=0, allow_inf_nan=False)
    model: float = Field(ge=0, allow_inf_nan=False)
    postprocess: float = Field(ge=0, allow_inf_nan=False)
    server_total: float = Field(ge=0, allow_inf_nan=False)


class ClassifyResponse(BaseModel):
    label: Literal['POSITIVE', 'NEGATIVE']
    score: float = Field(ge=0, le=1, allow_inf_nan=False)
    truncated: bool
    timings_ms: Timings


def create_app(backend_factory=None):
    factory = backend_factory if backend_factory is not None else OnnxBackend.from_environment

    @asynccontextmanager
    async def lifespan(application):
        application.state.backend = None
        backend = factory()
        if backend is None:
            raise RuntimeError('Backend initialization returned no backend')
        application.state.backend = backend
        try:
            yield
        finally:
            application.state.backend = None

    application = FastAPI(lifespan=lifespan)
    application.state.backend = None

    def ready_backend():
        backend = application.state.backend
        if backend is None:
            raise HTTPException(status_code=503, detail='Model backend is not ready')
        return backend

    @application.get('/health')
    def health():
        ready_backend()
        return {'status': 'ok', 'version': 'v1'}

    @application.post('/classify', response_model=ClassifyResponse)
    def classify(request: ClassifyRequest):
        start = perf_counter_ns()
        result = ready_backend().classify(request.text)
        result['timings_ms']['server_total'] = (perf_counter_ns() - start) / 1_000_000
        return result

    return application


app = create_app()
