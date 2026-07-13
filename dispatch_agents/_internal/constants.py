# Packages seeded into every new agent's dispatch.yaml at init/create time.
# The registries install exactly what's listed — these are not injected implicitly.
DEFAULT_SYSTEM_PACKAGES: list[str] = ["git", "curl", "wget", "gcc"]
