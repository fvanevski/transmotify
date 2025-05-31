# main.py
# Revised entry point using the refactored Orchestrator and UI components.

import sys
import traceback
# import torch

# Import refactored components
try:
    from config.config import Config
    from core.logging import setup_logging, log_error, log_info, log_warning
    from core.orchestrator import Orchestrator
    from ui.webapp import UI  # Import the UI class from the new location
except ImportError as e:
    # Use basic print for errors before logging is set up
    print(f"FATAL ERROR: Failed to import core components: {e}")
    print(
        "Ensure you are running from the project root directory and all dependencies are installed."
    )
    sys.exit(1)


def main():
    # Initial environment setup (like torch settings) before logging
    try:
        if torch.cuda.is_available():
            # Check if TF32 is supported (Ampere GPUs onwards)
            if torch.cuda.get_device_capability(0)[0] >= 8:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                print("INFO: TF32 optimizations enabled for CUDA.")
            else:
                print("INFO: TF32 optimizations not supported on this GPU.")
        else:
            print("INFO: CUDA not available, running on CPU.")
    except Exception as e:
        print(f"WARN: Could not set Torch backend settings: {e}")

    # --- Configuration and Logging Setup ---
    try:
        # 1. Initialize Configuration
        config_manager = Config()  # Uses config.json by default

        # 2. Setup Logging (using the loaded config)
        # setup_logging expects the config *dictionary*
        setup_logging(config_manager.config)

        # Optional: Log config source for debugging
        log_info(
            f"Configuration initialized using: {config_manager.config_file.resolve()}"
        )

        # 3. Perform Pre-computation Checks (like HF Token) after logging is ready
        hf_token = config_manager.get("hf_token")
        if not hf_token:
            # This warning is now more effective as logging is configured
            log_warning(
                "Hugging Face token ('hf_token') is missing or null in config/environment. "
                "Diarization may fail if using models requiring authentication (e.g., Pyannote)."
            )
        else:
            # Sensitive info, avoid logging the token itself
            log_info("Hugging Face token is configured.")

    except Exception as e:
        # Catch errors during config/logging setup
        initialization_error = (
            f"FATAL ERROR during initialization: {e}\n{traceback.format_exc()}"
        )
        # Try logging, but also print as logging might have failed
        try:
            log_error(initialization_error)
        except:
            pass
        print(initialization_error)
        sys.exit(1)

    # --- Application Setup and Launch ---
    try:
        # 4. Initialize the Orchestrator (passing the Config instance)
        orchestrator = Orchestrator(config=config_manager)

        # 5. Initialize the UI (passing the Orchestrator instance)
        webapp = UI(orchestrator=orchestrator)

        # 6. Launch the UI
        log_info("Launching Gradio UI...")
        # Check if running in an interactive environment (like IPython/Jupyter)
        # Gradio might not launch correctly or block in such environments.
        if hasattr(sys, "ps1") and sys.ps1:
            log_warning(
                "Running in an interactive shell (e.g., IPython/Jupyter). "
                "Gradio UI might not launch as expected or may block. "
                "Run as a standard Python script for best results."
            )
            # Optionally, you might prevent launch here or change launch parameters
            # webapp.launch(prevent_blocking=True) # Example, check Gradio docs

        # Launch for standard execution (bind to all interfaces)
        # Add other Gradio launch options as needed (e.g., share=True for public link)
        webapp.launch(
            server_name="0.0.0.0"
        )  # Binds to 0.0.0.0 to be accessible on network

        log_info("Gradio UI closed.")

    except ImportError as ie:
        # Catch potential import errors within Orchestrator or UI if missed earlier
        critical_error = f"FATAL ERROR: Missing dependency required by Orchestrator or UI: {ie}\n{traceback.format_exc()}"
        try:
            log_error(critical_error)
        except:
            pass
        print(critical_error)
        sys.exit(1)
    except Exception as e:
        runtime_error = (
            f"FATAL ERROR during application runtime: {e}\n{traceback.format_exc()}"
        )
        try:
            log_error(runtime_error)
        except:
            pass
        print(runtime_error)
        sys.exit(1)


if __name__ == "__main__":
    main()
