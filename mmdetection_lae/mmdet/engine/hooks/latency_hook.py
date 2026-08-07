# UPDATE: latency-per-block
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class LatencyLoggerHook(Hook):
    """Resets and reports the detector's per-block latency recorder.

    Only takes effect when the model was built with
    ``per_block_latency_measure=True`` (see
    ``mmdet.utils.latency_timer.BlockLatencyRecorder``); otherwise this
    hook is a no-op. Add it via ``custom_hooks`` in the config.
    """

    def before_test_epoch(self, runner: Runner) -> None:
        recorder = self._get_recorder(runner)
        if recorder is not None:
            recorder.reset()

    def after_test_epoch(self, runner: Runner, metrics=None) -> None:
        recorder = self._get_recorder(runner)
        if recorder is not None:
            recorder.sync_and_report(runner.logger)

    @staticmethod
    def _get_recorder(runner: Runner):
        model = runner.model
        if is_model_wrapper(model):
            model = model.module
        return getattr(model, 'latency_recorder', None)
