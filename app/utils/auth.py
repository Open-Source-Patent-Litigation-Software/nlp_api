from fastapi import HTTPException, Security, Depends
from fastapi.security.api_key import APIKeyHeader
from starlette.status import HTTP_403_FORBIDDEN

# Assuming `api_keys` dictionary is your storage for keys
api_keys = {
    "user": "4912615a2c3d2fe73dabb0ea95c342a6f35954ca47514a51c611c21ee27d72c2",
}

api_key_header = APIKeyHeader(name="key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key in api_keys.values():
        return api_key
    else:
        raise HTTPException(
            status_code=HTTP_403_FORBIDDEN, detail="Could not validate API key"
        )
