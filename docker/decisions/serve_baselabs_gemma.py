"""Serve the BaseLabs Gemma 4 checkpoints for text decisions."""

import argparse
import json
from pathlib import Path


SUPPORTED_OVERRIDES = {
    "attention": {"full_attention": "block_bidirectional", "sliding_attention": "causal"},
    "readout": "slot_lm",
    "head_dim": 256,
    "n_slots": 256,
}


def attention_masks(length, prefix_length, sliding_window, device):
    import torch

    positions = torch.arange(length, device=device)
    distance = positions[:, None] - positions[None, :]
    prefix = positions < prefix_length
    same_block = prefix[:, None] == prefix[None, :]
    # True means visible for SDPA. A suffix can read the prefix, never the reverse.
    return {
        "full_attention": ((distance >= 0) | same_block)[None, None],
        "sliding_attention": ((distance >= 0) & (distance < sliding_window))[None, None],
    }


def load_engine(checkpoint, device, max_tokens):
    import torch
    import transformers
    from qev.calibrate import Calibration
    from qev.engine import BranchResult, DecisionEngine, EngineInfo
    from qev.prompt import DEFAULT_STYLE
    from qev.readout import token_layout

    class GemmaDecisionEngine(DecisionEngine):
        def __init__(self):
            # Reuse Qev's text prompting and response assembly, with Gemma-specific loading and forward.
            self.model_path = str(checkpoint)
            self.device = torch.device(device)
            self.model_type = "gemma4"
            self.style = DEFAULT_STYLE
            self.template_kwargs = {}
            self.max_branch_tokens = self.max_request_tokens = max_tokens
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            self.slot_ids = self._verify_slots()
            self.max_slots = min(self.max_slots, SUPPORTED_OVERRIDES["n_slots"])
            calibration = checkpoint / "calibration.json"
            self.calibration = Calibration.load(calibration) if calibration.is_file() else Calibration()
            self.model = transformers.Gemma4ForConditionalGeneration.from_pretrained(
                checkpoint, local_files_only=True, dtype=torch.bfloat16,
                device_map={"": str(self.device)}, attn_implementation="sdpa",
            ).eval()
            self.sliding_window = self.model.config.get_text_config().sliding_window
            self.info = EngineInfo(
                model_path=self.model_path, model_type=self.model_type, dtype="bfloat16",
                device=str(self.device), fork_mode="sequential", fork_max_abs_diff=None,
            )

        @torch.inference_mode()
        def _run_sequential(self, state, rendered):
            results = []
            for question in rendered:
                prompt = self._render_prompt(state, question.suffix)
                ids, prefix_length, _ = token_layout(self.tokenizer, prompt, prompt.rindex(question.suffix), None)
                if len(ids) > self.max_branch_tokens or not 0 < prefix_length < len(ids):
                    raise ValueError("Invalid Gemma decision sequence length or block boundary")
                output = self.model(
                    input_ids=torch.tensor([ids], device=self.device),
                    attention_mask=attention_masks(len(ids), prefix_length, self.sliding_window, self.device),
                    use_cache=False, logits_to_keep=1, return_dict=True,
                )
                logits, mass = self._readout(output.logits[0, -1], question.n_slots)
                results.append(BranchResult(question.qid, question.kind, question.labels,
                                            question.descriptions, logits, mass, len(ids) - prefix_length))
            return results

    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    if config.get("model_type") != "gemma4" or config.get("baseten_modeling_overrides") != SUPPORTED_OVERRIDES:
        raise ValueError("Expected a BaseLabs Gemma 4 checkpoint with the supported attention and slot_lm overrides")
    return GemmaDecisionEngine()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="baselabs/bison-1", help="Hub model ID or mounted checkpoint directory")
    parser.add_argument("--revision", help="Hub checkpoint revision")
    parser.add_argument("--name", help="API model name (defaults to the last component of --model)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-tokens", type=int, default=131072)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 131072:
        parser.error("--max-tokens must be between 1 and 131072")

    import uvicorn
    from huggingface_hub import snapshot_download
    from qev.server import create_app

    checkpoint = Path(args.model)
    if checkpoint.is_dir():
        if args.revision:
            parser.error("--revision requires a Hub model ID")
    else:
        checkpoint = Path(snapshot_download(args.model, revision=args.revision))
    engine = load_engine(checkpoint, args.device, args.max_tokens)
    uvicorn.run(create_app(engine, model_name=args.name or Path(args.model).name), host=args.host, port=args.port)


if __name__ == "__main__":
    main()
