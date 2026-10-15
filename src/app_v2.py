"""V2 uses the v1 public models and inference with a checked signed-int8 model."""

from contextlib import asynccontextmanager
from time import perf_counter_ns

from fastapi import FastAPI, HTTPException

from src.app_v1 import ClassifyRequest, ClassifyResponse, Timings
from src.onnx_backend_v2 import OnnxBackendV2


def create_app(backend_factory=None):
    factory = backend_factory if backend_factory is not None else OnnxBackendV2.from_environment

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
        if application.state.backend is None:
            raise HTTPException(status_code=503, detail='Model backend is not ready')
        return application.state.backend

    @application.get('/health')
    def health():
        ready_backend()
        return {'status': 'ok', 'version': 'v2'}

    @application.post('/classify', response_model=ClassifyResponse)
    def classify(request: ClassifyRequest):
        start = perf_counter_ns()
        result = ready_backend().classify(request.text)
        result['timings_ms']['server_total'] = (perf_counter_ns() - start) / 1_000_000
        return result

    return application


app = create_app()
