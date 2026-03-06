"""Resource parsing utilities for Kubernetes-style resource specifications."""


def _parse_cpu(value: str | int | float) -> int:
    """Parse CPU value in Kubernetes format to internal units.

    Accepts:
    - Millicores as string: "250m", "500m", "1000m"
    - Cores as string or number: "0.25", "0.5", "1", "2", 0.25, 1

    Returns:
        CPU in internal units (1024 = 1 core)

    Examples:
        "250m" -> 256
        "500m" -> 512
        "1" -> 1024
        "2" -> 2048
    """
    if isinstance(value, int | float):
        # Numeric value treated as cores
        return int(value * 1024)

    value_str = str(value).strip().lower()

    if value_str.endswith("m"):
        # Millicores: "250m" -> 250 millicores -> 256 internal units
        millicores = int(value_str[:-1])
        return int(millicores * 1024 / 1000)
    else:
        # Cores as string: "0.25" -> 0.25 cores -> 256 internal units
        cores = float(value_str)
        return int(cores * 1024)


def _parse_memory(value: str | int) -> int:
    """Parse memory value in Kubernetes format to MB.

    Accepts:
    - Mebibytes: "512Mi", "1024Mi"
    - Gibibytes: "1Gi", "2Gi", "4Gi"
    - Plain MB as int: 512, 1024

    Returns:
        Memory in MB

    Examples:
        "512Mi" -> 512
        "1Gi" -> 1024
        "2Gi" -> 2048
    """
    if isinstance(value, int):
        return value

    value_str = str(value).strip()

    if value_str.endswith("Gi"):
        gibibytes = float(value_str[:-2])
        return int(gibibytes * 1024)
    elif value_str.endswith("Mi"):
        return int(float(value_str[:-2]))
    elif value_str.endswith("G"):
        gigabytes = float(value_str[:-1])
        return int(gigabytes * 1000)
    elif value_str.endswith("M"):
        return int(float(value_str[:-1]))
    else:
        return int(float(value_str))
