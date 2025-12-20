# logger.py
import os
import time

def clear_logs():
    """Clears the trace log file at the start of a new run."""
    try:
        # Import config here to avoid circular dependency
        import config as c
        with open(c.TRACE_LOG_PATH, "w") as f:
            f.write(f"--- Simulation Trace Log Started: {time.strftime('%H:%M:%S')} ---\n")
            f.write(f"{'Time':<12} {'Worker':<6} {'Event':<12} {'Status':<15} : {'Message'}\n")
            f.write("-" * 100 + "\n")
    except Exception as e:
        print(f"[LOGGER ERROR] Could not clear logs: {e}")

def write_trace(lock, sim_time, worker_id, event_type, status, message):
    """
    Thread-safe writer for detailed packet events.
    
    Args:
        lock: The multiprocessing lock object (manager.Lock())
        sim_time: Current simulation time (float)
        worker_id: ID of the process (str)
        event_type: Category (e.g., "PKT_GEN", "DISC", "FLOW")
        status: Outcome (e.g., "SUCCESS", "FAIL", "DROP")
        message: Detailed context info
    """
    if lock is None: return

    # Import config locally to check if logging is enabled
    import config as c
    if not c.ENABLE_TRACE_LOGGING: return

    log_entry = f"[{sim_time:09.3f}s] [W-{worker_id}] [{event_type:^12}] [{status:^15}] : {message}\n"
    
    try:
        with lock:
            with open(c.TRACE_LOG_PATH, "a") as f:
                f.write(log_entry)
    except Exception as e:
        print(f"[LOGGER ERROR] Failed to write trace: {e}")

def write_separator(lock):
    """Adds a visual separator line to the log."""
    if lock is None: return
    import config as c
    if not c.ENABLE_TRACE_LOGGING: return
    
    try:
        with lock:
            with open(c.TRACE_LOG_PATH, "a") as f:
                f.write("-" * 100 + "\n")
    except: pass