1. Centralize logger configuration via `dictConfig`  
   • Move from `logging.basicConfig` to a single, version‐controlled dict or YAML config.  
   • Benefits:  
     – You can declaratively add handlers (e.g. console, rotating file, external systems) without touching code.  
     – Per‐module log levels, formatters, and propagation rules live in one place.  
   • Complexity: low–medium  
     – Add one new file (`logging_config.py` or `logging.yaml`) and replace `setup_logging` with a call to `logging.config.dictConfig(...)`.  
   • Alternative: use a third‑party structured logger (e.g. `structlog`), at the cost of a steeper learning curve.

2. Use module‑level loggers, not root, everywhere  
   • In each module, do:  
     ```python
     import logging
     logger = logging.getLogger(__name__)
     ```  
   • Call `logger.debug()/info()/warning()/error()/exception()` instead of the wrapper functions in logging.py.  
   • Benefits: clearer “who said what” in stack traces, easier to override per‐module log levels.  
   • Complexity: low  
   • (You can keep your `log_info` wrappers if you really want, but I’d migrate gradually.)

3. Capture stack traces with `logger.exception()`  
   • In your `except:` blocks, prefer:  
     ```python
     except SomeError:
         logger.exception("Failed to do X")
         raise
     ```  
   • Benefits: you get full traceback in your logs.  
   • Complexity: low  

4. Define and use domain‑specific exception classes  
   • Rather than catching `Exception` everywhere, create a small hierarchy under, say, `transmotify.errors`:  
     ```python
     class TransmotifyError(Exception): ...
     class PipelineStageError(TransmotifyError): ...
     ```  
   • Wrap and rethrow lower‐level errors with context:  
     ```python
     except IOError as e:
         raise PipelineStageError("Could not read input file") from e
     ```  
   • Benefits: callers can catch only the exceptions they know how to handle; stack traces aren’t lost.  
   • Complexity: medium  

5. Add a top‑level “boundary” that logs uncaught exceptions and exits cleanly  
   • In your main script or pipeline runner:  
     ```python
     def main():
         try:
             run_pipeline(...)
         except TransmotifyError as err:
             logger.error("Pipeline failed: %s", err, exc_info=True)
             sys.exit(1)

     if __name__ == "__main__":
         main()
     ```  
   • Benefits: single point of exit, standardized logging of failures.  
   • Complexity: low  

6. (Optional) Rotate or archive your log files  
   • Swap `FileHandler` for `RotatingFileHandler` or `TimedRotatingFileHandler` in your config.  
   • Benefits: prevents unbounded log growth.  
   • Complexity: low  

––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

Summary of trade‑offs  
- Moving to `dictConfig` plus structured per‑module loggers is low‑risk and high‑reward—do this first.  
- Defining custom exceptions and wrapping errors gives you precise control flow, but costs a bit more code.  
- Using rotating handlers is a drop‑in improvement to your existing setup.  

With these incremental steps, you’ll get richer, more maintainable logs and a robust error model without wholesale changes to your pipeline logic.