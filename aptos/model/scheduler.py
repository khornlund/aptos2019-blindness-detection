from torch.optim import lr_scheduler


class MultiScheduler:
    """
    Class to wrap schedulers for both the model and the loss function.
    """
    def __init__(self, optimizer, model_config, loss_config):
        # setup model scheduler
        self._model_slr = self._build(optimizer.model_opt, model_config)

        # setup loss scheduler
        self._loss_slr = self._build(optimizer.loss_opt, loss_config)

    def _build(self, optimizer, config):
        return getattr(lr_scheduler, config['type'])(optimizer, **config['args'])

    @property
    def model_slr(self):
        return self._model_slr

    @property
    def loss_slr(self):
        return self._loss_slr

    def step(self):
        self.model_slr.step()
        self.loss_slr.step()
