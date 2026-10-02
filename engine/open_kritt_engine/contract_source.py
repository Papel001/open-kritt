import json
import re
import shutil
import urllib.error
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any

SOURCIFY_BASE_URL = "https://sourcify.dev/server"
EVM_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
CONTRACT_REVISION = "ONCHAIN"


class ContractSourceError(RuntimeError):
    pass


def _fetch_contract(chain_id: int, address: str) -> dict[str, Any]:
    url = f"{SOURCIFY_BASE_URL}/v2/contract/{chain_id}/{address}?fields=all"
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/json",
            "User-Agent": "open-kritt-contract-source/1",
        },
    )

    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            raise ContractSourceError(
                f"Contract {address} on chain {chain_id} is not verified on Sourcify."
            ) from exc
        raise ContractSourceError(
            f"Sourcify returned HTTP {exc.code} for {address} on chain {chain_id}."
        ) from exc
    except urllib.error.URLError as exc:
        raise ContractSourceError(f"Could not reach Sourcify: {exc.reason}") from exc
    except json.JSONDecodeError as exc:
        raise ContractSourceError("Sourcify returned invalid JSON.") from exc

    if not isinstance(payload, dict):
        raise ContractSourceError("Sourcify returned an unexpected response.")

    return payload


def _safe_destination(root: Path, source_name: str) -> Path:
    normalized = str(source_name or "").replace("\\", "/").lstrip("/")
    path = PurePosixPath(normalized)

    if not normalized or ".." in path.parts:
        raise ContractSourceError(f"Unsafe source path returned by explorer: {source_name!r}")

    destination = root.joinpath(*path.parts)
    resolved_root = root.resolve()
    resolved_destination = destination.resolve()

    try:
        resolved_destination.relative_to(resolved_root)
    except ValueError as exc:
        raise ContractSourceError(
            f"Source path escapes contract workspace: {source_name!r}"
        ) from exc

    return destination


def _write_sources(root: Path, sources: Any) -> int:
    if not isinstance(sources, dict):
        return 0

    written = 0

    for source_name, source in sources.items():
        if not isinstance(source_name, str) or not isinstance(source, dict):
            continue

        content = source.get("content")
        if not isinstance(content, str):
            continue

        destination = _safe_destination(root, source_name)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(content, encoding="utf-8")
        written += 1

    return written


def _implementation_addresses(payload: dict[str, Any]) -> list[str]:
    resolution = payload.get("proxyResolution")
    if not isinstance(resolution, dict) or resolution.get("isProxy") is not True:
        return []

    result: list[str] = []

    for implementation in resolution.get("implementations") or []:
        if not isinstance(implementation, dict):
            continue

        address = implementation.get("address")
        if isinstance(address, str) and EVM_ADDRESS_RE.fullmatch(address):
            if address.lower() not in {item.lower() for item in result}:
                result.append(address)

    return result


def materialize_contract_source(
    address: str,
    chain_id: int,
    destination: str | Path,
) -> tuple[str, str]:
    address = str(address or "").strip()

    if not EVM_ADDRESS_RE.fullmatch(address):
        raise ContractSourceError(f"Invalid EVM contract address: {address!r}")

    try:
        chain_id = int(chain_id)
    except (TypeError, ValueError) as exc:
        raise ContractSourceError("Chain ID must be a positive integer.") from exc

    if chain_id <= 0:
        raise ContractSourceError("Chain ID must be a positive integer.")

    destination = Path(destination)

    if destination.exists():
        shutil.rmtree(destination)

    source_root = destination / "source"
    source_root.mkdir(parents=True, exist_ok=True)

    primary = _fetch_contract(chain_id, address)
    primary_source_count = _write_sources(source_root, primary.get("sources"))

    if primary_source_count == 0:
        std_json = primary.get("stdJsonInput")
        if isinstance(std_json, dict):
            primary_source_count = _write_sources(source_root, std_json.get("sources"))

    if primary_source_count == 0:
        raise ContractSourceError(
            f"Sourcify returned no source files for {address} on chain {chain_id}."
        )

    implementations = []

    for implementation_address in _implementation_addresses(primary):
        try:
            implementation = _fetch_contract(chain_id, implementation_address)
        except ContractSourceError as exc:
            implementations.append(
                {
                    "address": implementation_address,
                    "verified": False,
                    "error": str(exc),
                }
            )
            continue

        implementation_root = (
            source_root / "implementations" / implementation_address.lower()
        )

        source_count = _write_sources(
            implementation_root,
            implementation.get("sources"),
        )

        if source_count == 0:
            std_json = implementation.get("stdJsonInput")
            if isinstance(std_json, dict):
                source_count = _write_sources(
                    implementation_root,
                    std_json.get("sources"),
                )

        implementations.append(
            {
                "address": implementation_address,
                "verified": source_count > 0,
                "sourceFiles": source_count,
                "compilation": implementation.get("compilation"),
            }
        )

    target_metadata = {
        "targetType": "contract",
        "provider": "sourcify-v2",
        "chainId": chain_id,
        "address": address,
        "match": primary.get("match"),
        "verifiedAt": primary.get("verifiedAt"),
        "compilation": primary.get("compilation"),
        "proxyResolution": primary.get("proxyResolution"),
        "implementations": implementations,
        "sourceFiles": primary_source_count,
    }

    (source_root / ".kritt-target.json").write_text(
        json.dumps(target_metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return str(source_root), CONTRACT_REVISION
