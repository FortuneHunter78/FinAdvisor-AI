
from fastapi import APIRouter
import datetime

from database import cursor, conn

import secrets
import hashlib

router = APIRouter()


@router.get("/get_key")
async def generate_key():
    raw_key = secrets.token_hex(32)
    hashed_key = hashlib.sha256(raw_key.encode()).hexdigest()


    timestamp = datetime.datetime.utcnow().isoformat()


    cursor.execute("INSERT INTO auth_keys(auth_key, created_at, updated_at) VALUES (?, ?, ?)",
                   (hashed_key, timestamp, timestamp))
    conn.commit()


    return {"auth_key": raw_key}


@router.post("/auth")
async def get_key(auth_key: str):
    hashed_key = hashlib.sha256(auth_key.encode()).hexdigest()
    cursor.execute("SELECT auth_key FROM auth_keys WHERE auth_key = ?", (hashed_key,))

    result = cursor.fetchone()


    if result:
        return {"status": '200', 'approved': True}
    else:
        return {"status": '400', 'approved': False}

@router.delete("/delete/")
async def delete_key(auth_key: str):
    hashed_key = hashlib.sha256(auth_key.encode()).hexdigest()
    cursor.execute("DELETE FROM auth_keys WHERE auth_key = ?", (hashed_key,))
    conn.commit()


    return {"status": '200', "message": "Key deleted successfully"}

