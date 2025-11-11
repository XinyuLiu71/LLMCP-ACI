#!/usr/bin/env python3
"""
Script to test OpenAI API keys for validity.
"""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Tuple, List

from openai import OpenAI
from openai import AuthenticationError, APIError


API_KEYS = [
    "sk-abcdef1234567890abcdef1234567890abcdef12",
    "sk-1234567890abcdef1234567890abcdef12345678",
    "sk-abcdefabcdefabcdefabcdefabcdefabcdef12",
    "sk-7890abcdef7890abcdef7890abcdef7890abcd",
    "sk-1234abcd1234abcd1234abcd1234abcd1234abcd",
    "sk-abcd1234abcd1234abcd1234abcd1234abcd1234",
    "sk-5678efgh5678efgh5678efgh5678efgh5678efgh",
    "sk-efgh5678efgh5678efgh5678efgh5678efgh5678",
    "sk-ijkl1234ijkl1234ijkl1234ijkl1234ijkl1234",
    "sk-mnop5678mnop5678mnop5678mnop5678mnop5678",
    "sk-qrst1234qrst1234qrst1234qrst1234qrst1234",
    "sk-uvwx5678uvwx5678uvwx5678uvwx5678uvwx5678",
    "sk-1234ijkl1234ijkl1234ijkl1234ijkl1234ijkl",
    "sk-5678mnop5678mnop5678mnop5678mnop5678mnop",
    "sk-qrst5678qrst5678qrst5678qrst5678qrst5678",
    "sk-uvwx1234uvwx1234uvwx1234uvwx1234uvwx1234",
    "sk-1234abcd5678efgh1234abcd5678efgh1234abcd",
    "sk-5678ijkl1234mnop5678ijkl1234mnop5678ijkl",
    "sk-abcdqrstefghuvwxabcdqrstefghuvwxabcdqrst",
    "sk-ijklmnop1234qrstijklmnop1234qrstijklmnop",
    "sk-1234uvwx5678abcd1234uvwx5678abcd1234uvwx",
    "sk-efghijkl5678mnopabcd1234efghijkl5678mnop",
    "sk-mnopqrstuvwxabcdmnopqrstuvwxabcdmnopqrst",
    "sk-ijklmnopqrstuvwxijklmnopqrstuvwxijklmnop",
    "sk-abcd1234efgh5678abcd1234efgh5678abcd1234",
    "sk-1234ijklmnop5678ijklmnop1234ijklmnop5678",
    "sk-qrstefghuvwxabcdqrstefghuvwxabcdqrstefgh",
    "sk-uvwxijklmnop1234uvwxijklmnop1234uvwxijkl",
    "sk-abcd5678efgh1234abcd5678efgh1234abcd5678",
    "sk-ijklmnopqrstuvwxijklmnopqrstuvwxijklmnop",
    "sk-1234qrstuvwxabcd1234qrstuvwxabcd1234qrst",
    "sk-efghijklmnop5678efghijklmnop5678efghijkl",
    "sk-mnopabcd1234efghmnopabcd1234efghmnopabcd",
    "sk-ijklqrst5678uvwxijklqrst5678uvwxijklqrst",
    "sk-1234ijkl5678mnop1234ijkl5678mnop1234ijkl",
    "sk-abcdqrstefgh5678abcdqrstefgh5678abcdqrst",
    "sk-ijklmnopuvwx1234ijklmnopuvwx1234ijklmnop",
    "sk-efgh5678abcd1234efgh5678abcd1234efgh5678",
    "sk-mnopqrstijkl5678mnopqrstijkl5678mnopqrst",
    "sk-1234uvwxabcd5678uvwxabcd1234uvwxabcd5678",
    "sk-ijklmnop5678efghijklmnop5678efghijklmnop",
    "sk-abcd1234qrstuvwxabcd1234qrstuvwxabcd1234",
    "sk-1234efgh5678ijkl1234efgh5678ijkl1234efgh",
    "sk-5678mnopqrstuvwx5678mnopqrstuvwx5678mnop",
    "sk-abcdijkl1234uvwxabcdijkl1234uvwxabcdijkl",
    "sk-ijklmnopabcd5678ijklmnopabcd5678ijklmnop",
    "sk-1234efghqrstuvwx1234efghqrstuvwx1234efgh",
    "sk-5678ijklmnopabcd5678ijklmnopabcd5678ijkl",
    "sk-abcd1234efgh5678abcd1234efgh5678abcd1234",
    "sk-ijklmnopqrstuvwxijklmnopqrstuvwxijklmnop",
]


def test_api_key(api_key: str) -> Tuple[str, bool, str]:
    """
    Test if an API key is valid by making a simple API call.
    
    Returns:
        Tuple of (api_key, is_valid, error_message)
    """
    try:
        client = OpenAI(api_key=api_key)
        # Try to list models - this is a lightweight operation
        # that will fail fast if the key is invalid
        client.models.list()
        return (api_key, True, "Valid")
    except AuthenticationError as e:
        return (api_key, False, f"Authentication failed: {str(e)}")
    except APIError as e:
        # Some API errors might still indicate the key is valid
        # (e.g., rate limit, insufficient credits)
        error_msg = str(e)
        if "insufficient_quota" in error_msg.lower() or "billing" in error_msg.lower():
            return (api_key, True, f"Valid but insufficient quota: {error_msg}")
        elif "rate_limit" in error_msg.lower():
            return (api_key, True, f"Valid but rate limited: {error_msg}")
        else:
            return (api_key, False, f"API error: {error_msg}")
    except Exception as e:
        return (api_key, False, f"Unexpected error: {str(e)}")


def test_all_keys(api_keys: List[str], max_workers: int = 10) -> List[Tuple[str, bool, str]]:
    """
    Test all API keys concurrently.
    
    Args:
        api_keys: List of API keys to test
        max_workers: Maximum number of concurrent workers
        
    Returns:
        List of tuples (api_key, is_valid, error_message)
    """
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_key = {executor.submit(test_api_key, key): key for key in api_keys}
        
        # Collect results as they complete
        for future in as_completed(future_to_key):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                key = future_to_key[future]
                results.append((key, False, f"Test failed: {str(e)}"))
    
    return results


def main():
    """Main function to run the API key tests."""
    print(f"Testing {len(API_KEYS)} OpenAI API keys...\n")
    print("=" * 80)
    
    results = test_all_keys(API_KEYS)
    
    # Separate valid and invalid keys
    valid_keys = [(key, msg) for key, is_valid, msg in results if is_valid]
    invalid_keys = [(key, msg) for key, is_valid, msg in results if not is_valid]
    
    # Print summary
    print(f"\n{'=' * 80}")
    print(f"SUMMARY:")
    print(f"  Total keys tested: {len(API_KEYS)}")
    print(f"  Valid keys: {len(valid_keys)}")
    print(f"  Invalid keys: {len(invalid_keys)}")
    print(f"{'=' * 80}\n")
    
    # Print valid keys
    if valid_keys:
        print("VALID API KEYS:")
        print("-" * 80)
        for i, (key, msg) in enumerate(valid_keys, 1):
            print(f"{i}. {key}")
            print(f"   Status: {msg}")
            print()
    
    # Print invalid keys
    if invalid_keys:
        print("\nINVALID API KEYS:")
        print("-" * 80)
        for i, (key, msg) in enumerate(invalid_keys, 1):
            print(f"{i}. {key}")
            print(f"   Error: {msg}")
            print()
    
    # Save results to file
    output_file = "api_key_test_results.txt"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"API Key Test Results\n")
        f.write(f"{'=' * 80}\n\n")
        f.write(f"Total keys tested: {len(API_KEYS)}\n")
        f.write(f"Valid keys: {len(valid_keys)}\n")
        f.write(f"Invalid keys: {len(invalid_keys)}\n\n")
        
        f.write(f"{'=' * 80}\n")
        f.write("VALID API KEYS:\n")
        f.write(f"{'-' * 80}\n")
        for key, msg in valid_keys:
            f.write(f"{key}\n")
            f.write(f"  Status: {msg}\n\n")
        
        f.write(f"\n{'=' * 80}\n")
        f.write("INVALID API KEYS:\n")
        f.write(f"{'-' * 80}\n")
        for key, msg in invalid_keys:
            f.write(f"{key}\n")
            f.write(f"  Error: {msg}\n\n")
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()

