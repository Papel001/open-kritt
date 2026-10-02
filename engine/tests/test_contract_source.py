import json
from pathlib import Path

import pytest

from open_kritt_engine import contract_source

ADDRESS = "0x1234567890abcdef1234567890abcdef12345678"
IMPLEMENTATION = "0xabcdefabcdefabcdefabcdefabcdefabcdefabcd"


def test_materialize_contract_source_writes_verified_sources_and_metadata(tmp_path, monkeypatch):
    responses = {
        ADDRESS.lower(): {
            "match": "exact_match",
            "verifiedAt": "2026-01-01T00:00:00Z",
            "sources": {
                "contracts/Test.sol": {
                    "content": "pragma solidity ^0.8.0; contract Test {}"
                }
            },
            "compilation": {
                "language": "Solidity",
                "compilerVersion": "0.8.30",
                "name": "Test",
            },
            "proxyResolution": {
                "isProxy": True,
                "proxyType": "EIP1967Proxy",
                "implementations": [{"address": IMPLEMENTATION}],
            },
        },
        IMPLEMENTATION.lower(): {
            "match": "exact_match",
            "sources": {
                "contracts/Implementation.sol": {
                    "content": "pragma solidity ^0.8.0; contract Implementation {}"
                }
            },
            "compilation": {
                "language": "Solidity",
                "compilerVersion": "0.8.30",
                "name": "Implementation",
            },
        },
    }

    def fake_fetch(chain_id, address):
        assert chain_id == 1
        return responses[address.lower()]

    monkeypatch.setattr(contract_source, "_fetch_contract", fake_fetch)

    repo_dir, revision = contract_source.materialize_contract_source(
        ADDRESS,
        1,
        tmp_path / "cache",
    )

    root = Path(repo_dir)

    assert revision == "ONCHAIN"
    assert (root / "contracts/Test.sol").is_file()
    assert (
        root
        / "implementations"
        / IMPLEMENTATION.lower()
        / "contracts/Implementation.sol"
    ).is_file()

    metadata = json.loads((root / ".kritt-target.json").read_text())

    assert metadata["targetType"] == "contract"
    assert metadata["chainId"] == 1
    assert metadata["address"] == ADDRESS
    assert metadata["proxyResolution"]["isProxy"] is True
    assert metadata["implementations"][0]["address"] == IMPLEMENTATION


def test_materialize_contract_source_rejects_invalid_address(tmp_path):
    with pytest.raises(contract_source.ContractSourceError):
        contract_source.materialize_contract_source(
            "0x1234",
            1,
            tmp_path / "cache",
        )


def test_safe_destination_blocks_parent_traversal(tmp_path):
    with pytest.raises(contract_source.ContractSourceError):
        contract_source._safe_destination(tmp_path, "../../escape.sol")
