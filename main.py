import os
import sys
import argparse
import signal
import faulthandler
from datetime import datetime
from utils import import_custom_class


def _register_signal_handlers():
    # In distributed runs, enabling custom signal handlers on every rank
    # can cause noisy teardown behavior. Keep it on rank0 by default.
    rank_env = os.environ.get("RANK")
    all_ranks = os.environ.get("GE_SIGNAL_HANDLER_ALL_RANKS", "0") == "1"
    if (not all_ranks) and (rank_env not in (None, "", "0")):
        return

    def _handle_signal(signum, _frame):
        try:
            sig_name = signal.Signals(signum).name
        except Exception:
            sig_name = str(signum)

        ts = datetime.now().isoformat(timespec="seconds")
        sys.stderr.write(
            f"\n[signal] {ts} pid={os.getpid()} received {sig_name} ({signum})\n"
        )
        try:
            faulthandler.dump_traceback(file=sys.stderr, all_threads=True)
        except Exception as exc:
            sys.stderr.write(f"[signal] failed to dump traceback: {exc}\n")
        sys.stderr.flush()

        # Restore default behavior and re-raise the signal to exit consistently.
        signal.signal(signum, signal.SIG_DFL)
        os.kill(os.getpid(), signum)

    # Enable faulthandler with the current stderr (after log redirection).
    try:
        faulthandler.enable(file=sys.stderr)
    except Exception:
        pass

    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGQUIT, signal.SIGUSR1, signal.SIGUSR2):
        try:
            signal.signal(sig, _handle_signal)
        except Exception:
            # Some signals may be unsupported depending on the platform.
            pass


def main():

    parser = argparse.ArgumentParser(
        description="Arguments for the main train program."
    )
    parser.add_argument('--config_file', type=str, required=True, help='Path for the config file')
    parser.add_argument('--runner_class_path', type=str, default="runner/ge_trainer_video_action_force_wm.py")
    parser.add_argument('--runner_class', type=str, default="Trainer")
    parser.add_argument('--mode', type=str, default="train")
    parser.add_argument('--checkpoint_path', type=str, default=None, help='Path to trained checkpoint, used in inference stage only')
    parser.add_argument('--n_validation', type=int, default=1, help='num of samples to predict, used in inference stage only')
    parser.add_argument('--n_chunk_action', type=int, default=1, help='num of action chunks to predict, used in action inference stage only')
    parser.add_argument('--output_path', type=str, default=None, help='Path to save outputs, used in inference stage only')
    parser.add_argument('--domain_name', type=str, default="agibotworld", help='Domain name of the validation dataset, used in inference stage only')

    args = parser.parse_args()
    Runner = import_custom_class(
        args.runner_class, args.runner_class_path, 
    )
    

    if args.mode == "train":
        ### Trainer
        runner = Runner(args.config_file)
        _register_signal_handlers()
        runner.prepare_dataset()
        runner.prepare_models()
        runner.prepare_trainable_parameters()
        runner.prepare_optimizer()
        runner.prepare_for_training()
        runner.prepare_trackers()
        runner.train()

    elif args.mode == "infer":
        ### Inference
        runner = Runner(args.config_file, output_dir=args.output_path)
        if args.checkpoint_path is not None:
            runner.args.load_weights = True
            runner.args.load_diffusion_model_weights = True
            runner.args.diffusion_model['model_path'] = args.checkpoint_path
        if hasattr(runner, "save_config_snapshot"):
            runner.save_config_snapshot()
        runner.prepare_val_dataset()
        runner.prepare_models()
        runner.infer(
            n_chunk_action=args.n_chunk_action,
            n_validation=args.n_validation,
            domain_name=args.domain_name
        )

    else:
        raise NotImplementedError



if __name__ == "__main__":
  
    main()
    
# bash scripts/train.sh main.py configs/ltx_model/policy_model_wipe_board_success_ft.yaml runner/ge_trainer_ft.py
