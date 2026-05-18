import argparse
import torch
from cs336_basics import Tokenizer
from cs336_basics.transformer import TransformerLM
from cs336_basics.utils import decode


def parse_args():
    p = argparse.ArgumentParser()
    # Model architecture (must match the checkpoint)
    p.add_argument("--vocab_size",     type=int,   default=10000)
    p.add_argument("--context_length", type=int,   default=256)
    p.add_argument("--d_model",        type=int,   default=512)
    p.add_argument("--num_heads",      type=int,   default=16)
    p.add_argument("--num_layers",     type=int,   default=4)
    p.add_argument("--d_ff",           type=int,   default=None)
    p.add_argument("--theta",          type=float, default=10000.0)
    # Files
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--vocab_file",  type=str, default="cs336_basics/tiny_stories_vocab.json")
    p.add_argument("--merges_file", type=str, default="cs336_basics/tiny_stories_merges.pkl")
    # Generation
    p.add_argument("--prompt",      type=str,   default="Once upon a time")
    p.add_argument("--max_tokens",  type=int,   default=200)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p",       type=float, default=0.9)
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()

    tokenizer = Tokenizer.from_files(args.vocab_file, args.merges_file)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        num_layers=args.num_layers,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        device=args.device,
    ).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["Model"])
    model.eval()

    prompt_ids = tokenizer.encode(args.prompt)
    print(f"Prompt: {args.prompt}")
    print(f"Generated:")

    generated_ids = decode(
        inputs=prompt_ids,
        model=model,
        max_context_window=args.context_length,
        temperature=args.temperature,
        p=args.top_p,
    )

    print(args.prompt + tokenizer.decode(generated_ids))


if __name__ == "__main__":
    main()
