import azure.functions as func
import json
from datetime import datetime, timezone


def main(req: func.HttpRequest) -> func.HttpResponse:
    """Health check endpoint for Azure Functions."""
    payload = {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "functions",
        "version": "1.0.0",
    }
    return func.HttpResponse(
        json.dumps(payload), status_code=200, mimetype="application/json"
    )


