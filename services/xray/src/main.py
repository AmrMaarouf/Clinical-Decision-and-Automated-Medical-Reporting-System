import os

from fastapi import FastAPI

SERVICE = "xray"
app = FastAPI(title=f"mica-{SERVICE}")


@app.get("/health")
def health():
    return {
        "service": SERVICE,
        "status": "ok",
        "fake": os.getenv("XRAY_FAKE", "1") == "1",
    }


# TODO(owner): replace the stub with the real endpoint, e.g. POST /run.
# Inputs and outputs must validate against mica_schemas (contract test).
