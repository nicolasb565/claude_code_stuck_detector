You are debugging a bug in Boost.Beast's WebSocket permessage-deflate implementation.

The bug: When reading WebSocket messages with permessage-deflate compression enabled and a small read buffer (even 1 byte), `async_read` / `read_some` fails with "invalid stored block length". The bug only manifests with small buffers — large buffers work fine.

The code is at /scratch

The issue is in the WebSocket read path. After decompressing a message frame, Beast appends 4 RFC 7692 tail bytes (0x00 0x00 0xFF 0xFF) to finalize the deflate stream. If zlib's output buffer is too small, it only partially consumes these bytes. On the next iteration, Beast re-appends all 4 bytes from the beginning, corrupting the zlib state.

Key files:
- include/boost/beast/websocket/impl/read.hpp (the read coroutine)
- include/boost/beast/websocket/detail/impl_base.hpp (WebSocket state)

The fix needs to track how many of the 4 tail bytes zlib has already consumed and only feed the remaining ones on the next iteration.

The full Boost tree is at /scratch. Build and run tests:
```bash
cd /scratch
./b2 libs/beast/test/beast/websocket --build-dir=build -j$(nproc)
```
