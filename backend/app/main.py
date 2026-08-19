"""The ASGI entrypoint - `uvicorn app.main:app`, `fastapi run app/main.py`.

Nothing but the eager instantiation lives here, and that is the point:
importing this module reads the environment and fails without one, which is
correct for a server and wrong for everything else. Code that wants the
factory - tests above all - imports `create_app` from `app.factory`, whose
import is side-effect free.
"""

from app.factory import create_app

app = create_app()
