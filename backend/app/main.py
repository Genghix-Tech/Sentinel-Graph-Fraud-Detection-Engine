import asyncio
import logging
import random
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from app.api import router, stream_state
from app.config import CORS_ORIGINS
from app.db import SessionLocal, init_db
from app.services.engine import engine
from app.services.simulator import run_scenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("sentinelgraph")


def _tick(scenario: str) -> None:
    with SessionLocal() as s:
        run_scenario(engine, s, scenario, n_legit=random.randint(1, 3))


async def _stream_loop() -> None:
    """When enabled, trickles live traffic through the engine with an occasional attack."""
    tick = 0
    while True:
        await asyncio.sleep(3)
        if not (stream_state["enabled"] and engine.ready):
            continue
        tick += 1
        scenario = random.choice(["ring", "ato", "card_testing"]) if tick % 20 == 0 else "legit"
        try:
            await run_in_threadpool(_tick, scenario)
        except Exception:
            log.exception("stream tick failed")


@asynccontextmanager
async def lifespan(_: FastAPI):
    init_db()
    try:
        await run_in_threadpool(engine.load)
    except Exception:
        log.exception("engine failed to load; seed the database with `python -m app.cli seed`")
    task = asyncio.create_task(_stream_loop())
    yield
    task.cancel()


app = FastAPI(
    title="SentinelGraph",
    description="Graph-based fraud & anomaly investigation engine",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=CORS_ORIGINS, allow_methods=["*"], allow_headers=["*"])
app.include_router(router)
