"""Compatibility entry point; evaluation implementation lives in research/evaluation.py."""
if __package__:
    from .research.evaluation import (
        add_shared_arguments, collect_logits, evaluation_temperature, main,
        prepare_evaluation, run_calibration, run_clinical, run_comparison,
        save_predictions, verify_checkpoint_split,
    )
else:
    from research.evaluation import (
        add_shared_arguments, collect_logits, evaluation_temperature, main,
        prepare_evaluation, run_calibration, run_clinical, run_comparison,
        save_predictions, verify_checkpoint_split,
    )

if __name__ == '__main__':
    main()
