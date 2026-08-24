from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def test_real_eval_launchers_use_infinite_non_recording_base_mode() -> None:
    for relative_path in (
        "scripts/run_uploaded_act.sh",
        "scripts/run_uploaded_v11.sh",
    ):
        script = (REPO_ROOT / relative_path).read_text()
        assert "--strategy.type=base" in script
        assert "--duration=0" in script
        assert "--strategy.type=episodic" not in script
        assert "--dataset." not in script
        assert "press Ctrl+C after success" in script
