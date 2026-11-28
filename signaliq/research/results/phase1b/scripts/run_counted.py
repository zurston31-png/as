"""Run universe_validation_batch.py unchanged, logging Alpaca client lines and the store's counters."""
import atexit, logging, runpy, sys
log_path, *batch_args = sys.argv[1:]
handler = logging.FileHandler(log_path)
handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
alpaca = logging.getLogger("app.data.alpaca_client")
alpaca.setLevel(logging.INFO); alpaca.addHandler(handler); alpaca.propagate = False
from app.data import response_store
def report():
    for (path, mode), store in response_store._shared.items():
        line = f"STORE COUNTERS mode={mode} served_from_store={store.served_from_store} fetched_live={store.fetched_live} path={path}"
        print(line, file=sys.stderr); handler.stream.write(line + "\n"); handler.flush()
atexit.register(report)
sys.argv = ["scripts/universe_validation_batch.py", *batch_args]
runpy.run_path("scripts/universe_validation_batch.py", run_name="__main__")
