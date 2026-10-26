"""Extract Decision/Evidence from swift results into CodaBench JSONL."""
import argparse
import json
from pathlib import Path
import re
import sys

def parse_response(response):
    decisions = re.findall(r'<Decision>\s*(real|fake)\s*</Decision>', response, re.I)
    evidence = re.findall(r'<Evidence>(.*?)</Evidence>', response, re.I | re.S)
    if len(decisions) != 1 or len(evidence) != 1:
        raise ValueError('Expected exactly one Decision and Evidence field')
    text = re.sub(r'\s+', ' ', re.sub(r'<[^>]+>', ' ', evidence[0])).strip()
    if not text:
        raise ValueError('Empty Evidence field')
    return decisions[0].lower(), text

def read_rows(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def convert(input_path, raw_path):
    inputs = read_rows(input_path)
    by_audio = {}
    order = []
    for row in inputs:
        key = row['audios'][0]
        fid = row['id']
        if key in by_audio or fid in order:
            raise ValueError('Duplicate input audio or ID')
        by_audio[key] = fid
        order.append(fid)
    predictions = {}
    for row in read_rows(raw_path):
        fid = by_audio.get(row['audios'][0])
        if fid is None or fid in predictions:
            raise ValueError('Unknown or duplicate prediction audio')
        try:
            label, explanation = parse_response(row['response'])
        except ValueError as e:
            raise ValueError(f'{fid}: {e}') from e
        predictions[fid] = {'file_id': fid, 'label': label, 'explanation': explanation}
    if set(predictions) != set(order):
        raise ValueError('Prediction/input ID sets differ')
    return [predictions[fid] for fid in order]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=Path, required=True)
    p.add_argument('--raw', type=Path, required=True)
    p.add_argument('--output', type=Path, default=Path('prediction.jsonl'))
    args = p.parse_args()
    rows = convert(args.input, args.raw)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', encoding='utf-8', newline='\n') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'{args.output}: {len(rows)} samples')

if __name__ == '__main__':
    main()
