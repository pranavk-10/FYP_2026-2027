"""Compatibility entry point; robustness implementation lives in research/robustness.py."""
if __package__:
    from .research.robustness import (
        CORRUPTIONS, CorruptedDataset, adversarial_attack, collect_adversarial_logits,
        collect_corrupted_logits, corrupt_image, hair_overlay, main,
        run_robustness, stable_seed, summarize_robustness,
    )
else:
    from research.robustness import (
        CORRUPTIONS, CorruptedDataset, adversarial_attack, collect_adversarial_logits,
        collect_corrupted_logits, corrupt_image, hair_overlay, main,
        run_robustness, stable_seed, summarize_robustness,
    )

if __name__ == '__main__':
    main()
