from pathlib import Path

REPO_ROOT = Path(__file__).parents[1]


def test_real_eval_launchers_use_official_episodic_rollout() -> None:
    for relative_path in (
        "scripts/run_uploaded_act.sh",
        "scripts/run_uploaded_v11.sh",
    ):
        script = (REPO_ROOT / relative_path).read_text()
        assert "--strategy.type=episodic" in script
        assert "--dataset.num_episodes=1" in script
        assert "--dataset.episode_time_s=30" in script
        assert "--return_to_initial_position=true" in script
        assert "--strategy.type=base" not in script
        assert "--duration=0" not in script
        assert "hardware_rollout_" not in script
