# Lesson 1.6: The "Local Security" Fallacy

While Shellworks emphasizes local execution for privacy and performance, a critical architectural lesson is that **local does not automatically mean safe**. When running high-performance models like Nemotron-3 on hardware like Jetson Thor, we introduce specific "execution-level" risks that the orchestrator must eventually manage.

## 1. The Supply Chain Risk (Dynamic Injection)
In our `nemotron.sh` startup script, we use `wget` to pull a reasoning parser directly from a remote repository at runtime.
* **The Vulnerability**: We are trusting that the remote file has not been compromised.
* **The Reality**: If a malicious actor replaces that parser file, they are injecting arbitrary Python code directly into your model server's runtime. Because vLLM often requires elevated privileges to manage GPU memory, a compromised parser could theoretically access the broader system.

## 2. The Persistence of "Remote Code"
The use of the `--trust-remote-code` flag is currently a necessity for many cutting-edge models to handle their specific architectures (like Mixture-of-Experts).
* **The Vulnerability**: This flag allows the model weights themselves to include and execute Python code.
* **The Reality**: The orchestrator treats the model as a "black box" that only returns text. However, if the model files were tampered with, the "box" is no longer black—it's a live execution environment running untrusted code on your local hardware.

## 3. Resource Exhaustion as a Denial of Service
We observed that the vLLM server, by default, might try to reserve up to 80% (or ~100GB) of the Jetson Thor's memory.
* **The Vulnerability**: A probabilistic model can sometimes enter a "loop" or generate an infinite sequence of tokens.
* **The Reality**: Without strict enforcement of `--gpu-memory-utilization` and `--max-model-len`, a single runaway inference task can starve the rest of the system. On a robotics platform, this could crash critical safety or navigation processes running alongside the AI.

## Summary: The Orchestrator's True Job
The orchestrator's role is not just to make the "math" work; it is to act as a **Deterministic Sandbox**. Even in a Proof of Concept, recognizing where we have traded security for "bleeding-edge" features is vital for moving from a demo to a robust system.
