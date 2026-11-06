"""Resolvability audit of the evidence that committed verification reports cite.

A verification report is only as good as the data behind it.  This script walks the
committed markdown under ``docs/``, extracts every *local artifact reference* (a
markdown link to ``.npz``/``.json``/``.png``/... or a backticked path that points at
one), and asks a single machine-answerable question per reference:

    can a reader of this commit still open the file the claim rests on?

Every reference lands in exactly one bucket:

``resolved``       the file exists next to the report (or at the repo root, for
                   reports that cite tree-root-relative paths);
``regenerable``    the path lives under a never-versioned output root
                   (``outputs/``, ``runs/``, ``artifacts/``, ``/tmp``) and is
                   therefore reproducible by re-running the documented command --
                   it is not *evidence in the commit*, and the ledger says so;
``allowlisted``    the path is declared missing on purpose in the allowlist, with a
                   reason attached (each entry is audit-visible, never silent);
``missing``        none of the above: the report cites data that this commit cannot
                   produce.  This is the WITHHELD case, and ``--check`` fails on it.

``--check`` is a *ratchet*: it compares the current scan with the committed baseline
(``docs/evidence-resolvability/audit.json``) and fails only on (a) a missing reference
that the baseline does not already record, (b) a stale allowlist entry.  The baseline is
the honest debt ledger -- existing gaps stay visible in the ledger, new ones cannot land.

The allowlist is deliberately glob-based *and* self-policed: an entry that no longer
matches anything in the tree is reported as ``stale_allowlist`` and fails ``--check``
as well, so the list cannot rot into a blanket waiver.

Usage::

    python scripts/audit_verification_evidence_resolvability.py \
        --output outputs/evidence-resolvability            # write JSON + ledger
    python scripts/audit_verification_evidence_resolvability.py --check   # gate
"""
from __future__ import annotations

import argparse
import fnmatch
import json
import re
from pathlib import Path

ARTIFACT_SUFFIXES = (
    '.npz', '.json', '.jsonl', '.csv', '.png', '.pdf', '.svg', '.dat', '.step',
    '.stp', '.log',
)
NEVER_VERSIONED_ROOTS = ('outputs/', 'runs/', 'artifacts/', '/tmp/', 'tmp/')
LINK = re.compile(r'\]\(\s*<?([^)\s>]+)>?\s*\)')
BACKTICK = re.compile(r'`([\w./+-]*(?:' + '|'.join(re.escape(s) for s in ARTIFACT_SUFFIXES) + r'))`')
EXTERNAL = ('http://', 'https://', 'mailto:', '#', 'ftp://')


def _is_artifact(reference: str) -> bool:
    return reference.lower().endswith(ARTIFACT_SUFFIXES)


def references_of(text: str):
    """Yield every artifact reference in one markdown document, in file order."""
    for match in LINK.finditer(text):
        target = match.group(1).split('#', 1)[0]
        if target and not target.startswith(EXTERNAL) and _is_artifact(target):
            yield target
    for match in BACKTICK.finditer(text):
        target = match.group(1)
        # A bare file name in prose ("result.json") is ambiguous; only accept
        # backticked *paths*, and only ones that are not already linked above.
        if '/' not in target.strip('/') and not target.startswith(NEVER_VERSIONED_ROOTS):
            continue
        if _is_artifact(target):
            yield target


def _normalise(reference: str, report: Path, repo: Path) -> str:
    """Repo-relative form of a reference, for reporting and allowlist matching."""
    raw = reference.strip()
    if raw.startswith('/tmp/') or raw.startswith('tmp/'):
        return raw
    candidate = (report.parent / raw).resolve()
    try:
        return candidate.relative_to(repo).as_posix()
    except ValueError:
        return raw.lstrip('./')


def _resolves(reference: str, report: Path, repo: Path) -> bool:
    if reference.startswith(('/', 'tmp/')) and not reference.startswith('/tmp/'):
        return False
    if reference.startswith('/tmp/'):
        return Path(reference).exists()
    if (report.parent / reference).exists():
        return True
    return (repo / reference).exists()


def _allowlisted(relative: str, report_relative: str, patterns):
    """Return the matching allowlist entry, or ``None``."""
    for entry in patterns:
        if fnmatch.fnmatchcase(relative, entry['glob']) and fnmatch.fnmatchcase(
            report_relative, entry.get('report', 'docs/**')
        ):
            return entry
    return None


def classify(reference: str, report: Path, repo: Path, patterns) -> tuple[str, str, str]:
    """Return ``(bucket, detail, allowlist_glob)``, resolving the reference where the report means it."""
    relative = _normalise(reference, report, repo)
    report_relative = report.resolve().relative_to(repo).as_posix()
    bare = reference.strip().lstrip('./')
    if bare.startswith(NEVER_VERSIONED_ROOTS):
        return 'regenerable', 'never-versioned output root', ''
    if _resolves(reference, report, repo):
        return 'resolved', '', ''
    entry = _allowlisted(relative, report_relative, patterns)
    if entry:
        return 'allowlisted', entry['reason'], entry['glob']
    return 'missing', '', ''


def scan(docs_root: Path, repo: Path, patterns):
    """Scan every markdown file under ``docs_root`` and bucket its references."""
    reports = []
    totals = dict(resolved=0, regenerable=0, allowlisted=0, missing=0)
    for document in sorted(docs_root.rglob('*.md')):
        text = document.read_text(encoding='utf-8', errors='replace')
        entries = []
        seen = set()
        for reference in references_of(text):
            relative = _normalise(reference, document, repo)
            if (reference, relative) in seen:
                continue
            seen.add((reference, relative))
            bucket, detail, glob = classify(reference, document, repo, patterns)
            totals[bucket] += 1
            entries.append(dict(reference=reference, path=relative, bucket=bucket, detail=detail, glob=glob))
        if entries:
            reports.append(dict(report=document.resolve().relative_to(repo).as_posix(), references=entries))
    used = {entry['glob'] for report in reports for entry in report['references'] if entry['glob']}
    stale = [entry['glob'] for entry in patterns if entry['glob'] not in used]
    return reports, totals, stale


def summarise(reports, totals, stale):
    missing = [dict(report=data['report'], path=entry['path'], reference=entry['reference'])
               for data in reports for entry in data['references'] if entry['bucket'] == 'missing']
    per_report = []
    for data in reports:
        counts = dict(resolved=0, regenerable=0, allowlisted=0, missing=0)
        for entry in data['references']:
            counts[entry['bucket']] += 1
        per_report.append(dict(report=data['report'], total=len(data['references']), **counts))
    per_report.sort(key=lambda row: (-row['missing'], -row['total'], row['report']))
    return dict(
        docs_scanned=len(reports),
        references=sum(totals.values()),
        buckets=totals,
        reports_with_missing=sum(1 for row in per_report if row['missing']),
        missing=missing,
        stale_allowlist=stale,
        per_report=per_report,
    )


def regressions(summary, baseline) -> list:
    """Missing references this scan found that the committed baseline did not record."""
    known = {(entry['report'], entry['path']) for entry in baseline.get('missing', [])}
    return [entry for entry in summary['missing'] if (entry['report'], entry['path']) not in known]


def ledger_markdown(summary) -> str:
    buckets = summary['buckets']
    lines = [
        '# Evidence resolvability ledger',
        '',
        'Generated by `scripts/audit_verification_evidence_resolvability.py`; regenerate, do not hand-edit.',
        '',
        f"- documents scanned: **{summary['docs_scanned']}**",
        f"- artifact references: **{summary['references']}**",
        f"- resolved in the commit: **{buckets['resolved']}**",
        f"- never-versioned output roots (reproducible, not evidence): **{buckets['regenerable']}**",
        f"- allowlisted as missing on purpose: **{buckets['allowlisted']}**",
        f"- unexplained missing (WITHHELD): **{buckets['missing']}** in "
        f"**{summary['reports_with_missing']}** reports",
        '',
        '## Reports citing evidence absent from the commit',
        '',
    ]
    rows = [row for row in summary['per_report'] if row['missing']]
    if not rows:
        lines.append('None: every cited artifact either exists, is declared, or is a never-versioned output.')
    else:
        lines += ['| report | refs | resolved | regenerable | allowlisted | missing |', '| --- | ---: | ---: | ---: | ---: | ---: |']
        lines += [f"| `{row['report']}` | {row['total']} | {row['resolved']} | {row['regenerable']} | "
                  f"{row['allowlisted']} | {row['missing']} |" for row in rows]
    lines += ['', '## Stale allowlist entries', '']
    lines += [f'- `{glob}` matches nothing in the tree' for glob in summary['stale_allowlist']] or ['None.']
    return '\n'.join(lines) + '\n'


def figure(summary, path: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    buckets = summary['buckets']
    order = ['resolved', 'regenerable', 'allowlisted', 'missing']
    colours = ['#2a7f62', '#4d7ea8', '#c08a2e', '#b3352b']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].bar(order, [buckets[key] for key in order], color=colours)
    axes[0].set_ylabel('artifact references')
    axes[0].set_title(f"{summary['references']} citations across {summary['docs_scanned']} documents")
    for index, key in enumerate(order):
        axes[0].text(index, buckets[key], str(buckets[key]), ha='center', va='bottom', fontsize=9)
    rows = [row for row in summary['per_report'] if row['missing']][:12]
    if rows:
        labels = [row['report'].replace('docs/', '').replace('/report.md', '') for row in rows][::-1]
        values = [row['missing'] for row in rows][::-1]
        axes[1].barh(labels, values, color=colours[3])
        axes[1].set_xlabel('unexplained missing references')
        axes[1].set_title('worst offenders (top 12)')
        axes[1].tick_params(labelsize=7)
    else:
        axes[1].text(.5, .5, 'no unexplained missing references', ha='center', va='center')
        axes[1].set_axis_off()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    repo = Path(__file__).resolve().parents[1]
    parser.add_argument('--docs', type=Path, default=repo / 'docs')
    parser.add_argument('--allowlist', type=Path,
                        default=repo / 'docs/evidence-resolvability-allowlist.json')
    parser.add_argument('--output', type=Path, default=None,
                        help='directory for audit.json / ledger.md / resolvability.png')
    parser.add_argument('--baseline', type=Path,
                        default=repo / 'docs/evidence-resolvability/audit.json',
                        help='committed debt ledger that --check ratchets against')
    parser.add_argument('--check', action='store_true',
                        help='exit 1 on any unexplained missing reference or stale allowlist entry')
    args = parser.parse_args(argv)
    patterns = []
    if args.allowlist.exists():
        patterns = json.loads(args.allowlist.read_text())['patterns']
    reports, totals, stale = scan(args.docs, repo, patterns)
    summary = summarise(reports, totals, stale)
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'audit.json').write_text(json.dumps(summary, indent=2) + '\n')
        (args.output / 'ledger.md').write_text(ledger_markdown(summary))
        figure(summary, args.output / 'resolvability.png')
    print(json.dumps({key: summary[key] for key in
                      ('docs_scanned', 'references', 'buckets', 'reports_with_missing', 'stale_allowlist')},
                     indent=2))
    if not args.check:
        return 0
    if not args.baseline.exists():
        print(f"baseline {args.baseline} missing: run without --check first to record it")
        return 1
    new = regressions(summary, json.loads(args.baseline.read_text()))
    for entry in new[:20]:
        print(f"NEW MISSING {entry['path']}  ({entry['report']})")
    for pattern in summary['stale_allowlist']:
        print(f"STALE ALLOWLIST {pattern}")
    return 1 if (new or summary['stale_allowlist']) else 0


if __name__ == '__main__':
    raise SystemExit(main())