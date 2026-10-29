"""The ASGI entrypoint (`uvicorn app.main:app`).

Importing this reads the environment. Tests import `create_app` from `app.factory`.
"""

from app.factory import create_app

app = create_app()
