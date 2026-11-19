// Worker entry point. Only the default export lives here: the Workers runtime treats every
// named export of the entry module as an entrypoint, so helpers stay in app.js.
import app from "./app.js";

export default app;
