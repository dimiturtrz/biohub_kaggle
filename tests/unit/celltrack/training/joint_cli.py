from celltrack.training.joint_cli import JointCli


def test_build_parser():
    """The parser exposes the run's flags with the documented defaults — the interface the trainer fans out."""
    parser = JointCli.build_parser()

    args = parser.parse_args([])
    assert args.steps == 1500
    assert args.batch_size == 16  # batched by default (smallest batch that saturates the backbone); 1 = per-pair
    assert args.gpu_scene_fraction == 0.0  # generation off unless asked
    assert args.gpu_scene_detection is True  # honest full labels by default

    off = parser.parse_args(["--no-gpu-scene-detection"])
    assert off.gpu_scene_detection is False
