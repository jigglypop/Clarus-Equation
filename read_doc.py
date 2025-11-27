import os
import sys

try:
    files = os.listdir('docs')
    print(f"Files in docs: {files}")
    for f in files:
        # Check for normalized or non-normalized forms if needed, but simple check first
        # We encode to utf-8 and decode to check if it matches roughly or just print it
        print(f"Checking: {f}")
        if '평탄화' in f or '공리' in f or '355' in  repr(f): # Check repr for escaped chars
            print(f"Found target file: {f}")
            path = os.path.join('docs', f)
            with open(path, 'r', encoding='utf-8') as file:
                print("--- CONTENT START ---")
                print(file.read())
                print("--- CONTENT END ---")
except Exception as e:
    print(f"Error: {e}")

