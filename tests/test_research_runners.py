import hashlib
import json
from pathlib import Path

import pytest

from bench.data import load
from scripts import research_joint_mixture as research


def protocol():
    path = Path(__file__).resolve().parent.parent/'experiments/algorithm_research_protocol_2026_11_06.json'
    return json.loads(path.read_text())


@pytest.mark.parametrize('path,value', [
    (('min_history',), 500),  # would previously create empty/wrapped temporal folds
    (('primary_test_draws',), 0),
    (('primary_test_draws',), 2000),
    (('secondary_method',), 'static_joint_mixture'),  # would overwrite the primary result
    (('dense_windows',), [50, 50]),
    (('synthetic', 'replicates_per_scenario'), 0),
    (('synthetic', 'draws'), 200),
    (('synthetic', 'diagnostic_threshold'), 1),
    (('synthetic', 'scenarios', 0, 'multiplier'), 7),
    (('synthetic', 'scenarios', 0, 'bias_starts'), -1),
])
def test_protocol_rejects_misleading_or_empty_experiments(path, value):
    config = protocol()
    target = config
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    with pytest.raises(ValueError):
        research.validate_protocol(config, 1661)


def test_original_protocol_is_valid():
    research.validate_protocol(protocol(), 1661)


@pytest.mark.parametrize('source,destination', [
    ('protocol.json', 'results.npz'), ('protocol.json', 'protocol.json'), ('results.npz', 'results.json'),
])
def test_output_cannot_overwrite_array_artifact_or_protocol(tmp_path, monkeypatch, source, destination):
    path = tmp_path/source
    original = json.dumps(protocol())
    path.write_text(original)
    monkeypatch.setattr('sys.argv', ['research', '--protocol', str(path), '--out', str(tmp_path/destination)])
    with pytest.raises(SystemExit) as exc:
        research.main()
    assert exc.value.code == 2
    assert path.read_text() == original


def test_failed_controls_preserve_completed_report_and_arrays(tmp_path, monkeypatch):
    draws = load(clean_csv=None)
    config = protocol()
    archive = tmp_path/'input/Orig_IL_lotto.csv'
    archive.parent.mkdir()
    archive.write_bytes(b'test archive; parser replaced with a validated fixture')
    config['archive_sha256'] = hashlib.sha256(archive.read_bytes()).hexdigest()
    path = tmp_path/'protocol.json'
    path.write_text(json.dumps(config))
    report, arrays = tmp_path/'results.json', tmp_path/'results.npz'
    report.write_bytes(b'previous complete report')
    arrays.write_bytes(b'previous complete arrays')
    monkeypatch.setattr(research, 'ROOT', tmp_path)
    monkeypatch.setattr(research.data, 'load', lambda **kwargs: draws)

    def fake_comparison(draws, config, staged_arrays):
        assert staged_arrays != arrays
        staged_arrays.write_bytes(b'new provisional arrays')
        return {'test': 'provisional'}

    def failed_controls(config):
        raise RuntimeError('synthetic verification failed')

    monkeypatch.setattr(research, 'real_comparison', fake_comparison)
    monkeypatch.setattr(research, 'controls', failed_controls)
    monkeypatch.setattr('sys.argv', ['research', '--protocol', str(path), '--out', str(report)])
    with pytest.raises(RuntimeError, match='synthetic verification failed'):
        research.main()
    assert report.read_bytes() == b'previous complete report'
    assert arrays.read_bytes() == b'previous complete arrays'
    assert not list(tmp_path.glob('joint-research-*'))
