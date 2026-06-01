from fastapi import FastAPI

from secrag import __version__

app = FastAPI(title="secrag", version=__version__)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "version": __version__}
