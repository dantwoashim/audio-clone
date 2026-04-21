from __future__ import annotations

from pathlib import Path

import click

from f5_tts.studio.editing import align_transcript, build_text_edit_plan, render_speech_edit


@click.command()
@click.option("--audio", "audio_path", required=True, type=click.Path(exists=True, dir_okay=False))
@click.option("--transcript", default="", help="Confirmed transcript for the source audio.")
@click.option("--target", "target_text", required=True, help="Exact phrase to replace in the transcript.")
@click.option("--replacement", "replacement_text", required=True, help="Replacement phrase.")
@click.option("--occurrence", default=1, type=int, help="Which occurrence of the target phrase to replace.")
@click.option("--output_dir", default="tests", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--checkpoint", "checkpoint_path", default="", help="Optional finetuned checkpoint to use for the edit.")
@click.option("--use_ema/--no_use_ema", default=True, help="Use EMA weights when loading the checkpoint.")
@click.option("--preserve_timing/--free_timing", default=True, help="Keep the original phrase timing or allow it to expand.")
@click.option("--nfe_step", default=32, type=int, help="Number of denoising steps.")
@click.option("--render_spectrogram/--no_render_spectrogram", default=True, help="Render a spectrogram alongside the edited audio.")
def main(
    audio_path: str,
    transcript: str,
    target_text: str,
    replacement_text: str,
    occurrence: int,
    output_dir: str,
    checkpoint_path: str,
    use_ema: bool,
    preserve_timing: bool,
    nfe_step: int,
    render_spectrogram: bool,
):
    alignment = align_transcript(audio_path, transcript)
    plan = build_text_edit_plan(
        alignment["transcript"],
        alignment["words"],
        target_text,
        replacement_text,
        occurrence=occurrence,
        preserve_timing=preserve_timing,
    )

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    wav_path = output_root / "speech_edit_out.wav"
    spec_path = output_root / "speech_edit_out.png"

    result = render_speech_edit(
        audio_path,
        original_text=plan["original_text"],
        edited_text=plan["edited_text"],
        spans=plan["spans"],
        ckpt_file=checkpoint_path,
        use_ema=use_ema,
        nfe_step=nfe_step,
        output_wav_path=str(wav_path),
        output_spec_path=str(spec_path) if render_spectrogram else None,
        preserve_timing=preserve_timing,
    )

    click.echo(f"Transcript: {alignment['transcript']}")
    click.echo(f'Edited phrase: {plan["target_text"]} -> {plan["replacement_text"]}')
    click.echo(f"Audio written to: {result['audio_path']}")
    if result.get("spectrogram_path"):
        click.echo(f"Spectrogram written to: {result['spectrogram_path']}")


if __name__ == "__main__":
    main()
