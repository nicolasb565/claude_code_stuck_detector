You are debugging a bug in Express.js v5.2.0.

The bug: Express 5.2.0 changed the extended query parser to return a null-prototype object. This broke applications relying on `req.query` having standard Object.prototype methods like `hasOwnProperty`. 

The code is at /scratch

Find the problematic change in the query parser middleware and fix it so that `req.query` returns a normal object with Object.prototype methods available.

Verify by checking that this works:
```javascript
const express = require('.');
const app = express();
app.get('/', (req, res) => {
  // This should not throw
  const has = req.query.hasOwnProperty('foo');
  res.send('ok');
});
```
