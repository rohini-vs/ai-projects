import psycopg
from pgvector.psycopg import register_vector

DSN = "postgresql://rag:rag@localhost:5432/rag"


def connect():
    # the vector type must exist before register_vector can find it
    with psycopg.connect(DSN, autocommit=True) as c:
        c.execute("CREATE EXTENSION IF NOT EXISTS vector")
    conn = psycopg.connect(DSN)
    register_vector(conn)
    return conn


def init_schema(conn):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id        SERIAL PRIMARY KEY,
            strategy  TEXT NOT NULL,        -- fixed / recursive / structure
            chunk_no  INT  NOT NULL,
            page      INT,
            page_end  INT,
            heading   TEXT,
            content   TEXT NOT NULL,
            embedding vector(768) NOT NULL
        )""")
    conn.commit()


def insert_chunks(conn, strategy: str, chunks: list[dict], vectors):
    conn.execute("DELETE FROM chunks WHERE strategy = %s", (strategy,))  # re-runnable
    with conn.cursor() as cur:
        cur.executemany(
            "INSERT INTO chunks (strategy, chunk_no, page, page_end, heading, content, embedding) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)",
            [(strategy, c["id"], c["page"], c.get("page_end", c["page"]),
              c.get("heading"), c["text"], v)
             for c, v in zip(chunks, vectors)])
    conn.commit()


def search(conn, query_vec, strategy: str, k: int = 5) -> list[dict]:
    rows = conn.execute(
        """SELECT chunk_no, page, heading, content, 1 - (embedding <=> %s) AS score
           FROM chunks WHERE strategy = %s
           ORDER BY embedding <=> %s LIMIT %s""",
        (query_vec, strategy, query_vec, k)).fetchall()
    return [{"chunk_no": r[0], "page": r[1], "heading": r[2], "text": r[3], "score": float(r[4])}
            for r in rows]