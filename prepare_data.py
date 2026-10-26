"""Prepare released annotations and audio for ms-swift."""
import argparse
import json
import re
from pathlib import Path

SYSTEM = 'You are a senior audio forensics scientist specializing in speech deepfake detection.'
USER = 'Analyze this audio for authenticity using observable acoustic evidence from the audio, and produce a structured XML forensic report covering spectral, prosodic, pronunciation, noise, and other acoustic anomalies, with a final decision of Real or Fake.'


def read_jsonl(path):
    return [json.loads(s) for s in path.read_text(encoding='utf-8-sig').splitlines() if s.strip()]

def messages():
    return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': USER}]

def training_records(data, split):
    rows = read_jsonl(data / 'labels' / f'{split}_labels.jsonl')
    ids = [r['file_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError(f'{split}: duplicate annotation IDs')
    records = []
    for row in rows:
        fid = row['file_id']
        audio = data / split / (fid + '.wav')
        if not audio.is_file():
            raise FileNotFoundError(audio)
        if row['label'] not in ('real', 'fake'):
            raise ValueError(f'Invalid label: {fid}')
        answer = row['explanation']
        decisions = re.findall(r'<Decision>\s*(real|fake)\s*</Decision>', answer, re.I)
        evidence = re.findall(r'<Evidence>(.*?)</Evidence>', answer, re.I | re.S)
        if len(decisions) != 1 or decisions[0].lower() != row['label'] or len(evidence) != 1:
            raise ValueError(f'Invalid XML reference: {fid}')
        records.append({'id': fid, 'messages': messages() + [
            {'role': 'assistant', 'content': answer}
        ], 'audios': [audio.as_posix()], 'label': row['label'], 'model': None})
    return records

def inference_records(audio_dir):
    files = sorted(audio_dir.glob('*.wav'))
    if not files:
        raise ValueError(f'No WAV files found: {audio_dir}')
    return [{'id': f.stem, 'messages': messages(), 'audios': [f.as_posix()]} for f in files]

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--audio-dir', type=Path)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    if args.audio_dir:
        if args.output is None:
            p.error('--output is required with --audio-dir')
        outputs = [(args.output, inference_records(args.audio_dir))]
    else:
        outputs = [(Path('data/swift') / f'{s}.jsonl', training_records(Path('data'), s))
                   for s in ('train', 'dev')]
    for path, rows in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows),
                        encoding='utf-8', newline='\n')
        print(f'{path}: {len(rows)} samples')

if __name__ == '__main__':
    main()
