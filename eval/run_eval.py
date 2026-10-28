"""Small repeatable retrieval/answer smoke evaluation against the running API."""

import json
import os
import sys
from pathlib import Path

import httpx


def main() -> int:
    base_url = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")
    cases = json.loads((Path(__file__).parent / "cases.json").read_text(encoding="utf-8"))
    failures = 0
    with httpx.Client(base_url=base_url, timeout=60) as client:
        for case in cases:
            response = client.post("/query", json={"question": case["question"]})
            response.raise_for_status()
            result = response.json()
            answer = result["answer"].casefold()
            terms_ok = all(term.casefold() in answer for term in case["expected_terms"])
            source_ok = not case["expected_source"] or any(
                case["expected_source"] in source["filename"] for source in result["sources"]
            )
            abstention_ok = result["abstained"] is case["should_abstain"]
            passed = terms_ok and source_ok and abstention_ok
            print(f"{'PASS' if passed else 'FAIL'} {case['id']}: {result['answer']}")
            failures += not passed
    print(f"\n{len(cases) - failures}/{len(cases)} evaluation cases passed")
    return int(failures > 0)


if __name__ == "__main__":
    sys.exit(main())
