You are debugging a query optimizer bug in SQLite.

The bug: The EXISTS-to-JOIN optimization incorrectly handles UNION queries inside EXISTS clauses, causing queries to return empty results instead of matching rows.

The code is at /scratch

Reproducer:
```sql
CREATE TABLE parent(id TEXT);
CREATE TABLE child_a(id TEXT);
CREATE TABLE child_b(id TEXT);
INSERT INTO parent VALUES('p1');
INSERT INTO child_a VALUES('p1');

SELECT count(*), parent.id FROM parent
WHERE EXISTS (
    SELECT 1 FROM child_a WHERE child_a.id = parent.id
    UNION
    SELECT 1 FROM child_b WHERE child_b.id = parent.id
);
```
Expected: `1|p1`. Actual (buggy): empty result.

Find and fix the bug in the query optimizer. The issue is in how the EXISTS-to-JOIN optimization was introduced in this version. Build SQLite and verify the fix:
```bash
./configure && make
./sqlite3 ':memory:' < test.sql
```
